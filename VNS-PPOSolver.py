import os
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
tf.autograph.set_verbosity(0)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from NetworkGenerator import Scenario
from VNSSolver import VNSSolver

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(ActorNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(n_actions, activation='softmax')

    def call(self, state):
        data = self.fc1(state)
        data = self.fc2(data)
        actionProb = self.fc3(data)
        return actionProb
        
class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims):
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)
    
    def call(self, state):
        data = self.fc1(state)
        data = self.fc2(data)
        stateValue = self.q(data)
        return stateValue


class ADMEnvironment(VNSSolver):
    def __init__(self, S, seed):
        super().__init__(S, seed)
        self.flt2idx = {flt:idx+1 for idx, flt in enumerate(self.flts)}
        self.ap2idx = {ap:idx+1 for idx, ap in enumerate(self.S.airports)}
        self.skdPdist = self.getStringDistribution(self.skdPs) 
        self.skdQdist = self.getStringDistribution(self.skdQs)
        self.skdStrState = (self.skdPs, self.skdQs, self.skdPdist, self.skdQdist)
        self.stateShape = self.string2tensor(self.skdStrState).shape
    
        # prepare action_tuples and idx2action                
        pindpairs = itertools.combinations(range(len(self.skdPs)), 2)
        qindpairs = itertools.combinations(range(len(self.skdQs)), 2)
        combs = itertools.product(range(4), range(4), pindpairs, qindpairs)
        self.idx2action = {idx:comb for idx, comb in enumerate(combs)}
        self.n_actions = len(self.idx2action)        
    
    def reset(self):
        self.lastStrState = self.skdStrState
        self.lastObj = self.evaluate(*self.skdStrState[:2])[0]
        return self.string2tensor(self.skdStrState)

    def string2tensor(self, strState):        
        distcol = np.array([np.concatenate([strState[-2],strState[-1]])]).T        
        flightidx = [[self.flt2idx[flt] for flt in P[1:-1]] for P in strState[0]+strState[1]]
        return tf.concat([distcol,tf.ragged.constant(flightidx, dtype=tf.float32).to_tensor()],1)
        
    def step(self, action_idx):
        k1, k2, pindpair, qindpair = self.idx2action[action_idx]
        curPs, curQs = self.lastStrState[:2]
        curObj = self.lastObj
        
        for (nP1, nP2) in eval("self."+self.k2func[k1])(curPs[pindpair[0]], curPs[pindpair[1]]):
            nPs = curPs.copy()
            nPs[pindpair[0]], nPs[pindpair[1]] = nP1, nP2
            for (nQ1,nQ2) in eval("self."+self.k2func[k2])(curQs[qindpair[0]], curQs[qindpair[1]]):
                nQs = curQs.copy()
                nQs[qindpair[0]], nQs[qindpair[1]] = nQ1, nQ2
                nObj = self.evaluate(nPs, nQs)[0]
                if nObj < curObj:
                    curPs, curQs, curObj = nPs, nQs, nObj
        
        reward = self.lastObj - curObj
        curPdist = self.getStringDistribution(curPs) 
        curQdist = self.getStringDistribution(curQs)
        self.lastStrState = (curPs, curQs, curPdist, curQdist)
        self.lastObj = curObj
        return self.string2tensor(self.lastStrState), reward
    
class PPOMemory:
    def __init__(self, batch_size):
        self.states, self.probs, self.vals, self.actions, self.rewards = [], [], [], [], []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.probs),\
                np.array(self.vals), np.array(self.rewards), batches

    def store_memory(self, state, action, probs, vals, reward):
        self.states.append(state); self.actions.append(action); self.probs.append(probs)
        self.vals.append(vals); self.rewards.append(reward)

    def clear_memory(self):
        self.states, self.probs, self.vals, self.actions, self.rewards = [], [], [], [], []


class Agent:
    def __init__(self, n_actions, input_dims, fc1_dims, fc2_dims, gamma, alpha, \
                 gae_lambda, policy_clip, batch_size, n_epochs, chkpt_dir):
        self.n_actions = n_actions
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir
        
        self.actor = ActorNetwork(n_actions,fc1_dims, fc2_dims)
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic = CriticNetwork(fc1_dims, fc2_dims)
        self.critic.compile(optimizer=Adam(learning_rate=alpha))
        self.memory = PPOMemory(batch_size)

    def store_transition(self, state, action, probs, vals, reward):
        self.memory.store_memory(state, action, probs, vals, reward)

    def save_models(self):
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
        self.actor.save_weights(self.chkpt_dir + '/Actor')
        self.critic.save_weights(self.chkpt_dir + '/Critic')

    def choose_action(self, state):
        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)
        if action.numpy()[0] == self.n_actions:
            action = self.n_actions-1
        else:
            action = action.numpy()[0]

        value = value.numpy()[0]
        log_prob = log_prob.numpy()[0]
        return action, log_prob, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, batches = self.memory.generate_batches()
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])
                    probs = self.actor(states)
                    dist = tfp.distributions.Categorical(probs)
                    new_probs = dist.log_prob(actions)      # there seems to be a shape mismatch... how to debug this?
                    
                    critic_value = self.critic(states)
                    critic_value = tf.squeeze(critic_value, 1)
                    
                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs, weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    critic_loss = keras.losses.MSE(critic_value, returns)

                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_grads, critic_params))

        self.memory.clear_memory()
    


def train_and_test(config):
    tf.random.set_seed(seed = config["SEED"])
    S = Scenario(config["DATASET"], config["SCENARIO"], "PAX")
    env = ADMEnvironment(S, config["SEED"])
    agent = Agent(n_actions=env.n_actions, input_dims=env.stateShape, fc1_dims=config["FC1DIMS"], fc2_dims=config["FC2DIMS"], \
                  gamma=config["GAMMA"], alpha=config["ALPHA"], gae_lambda=config["GAELAMBDA"], policy_clip=config["POLICYCLIP"], \
                  batch_size=config["BATCHSIZE"], n_epochs=config["NEPOCH"], chkpt_dir="Results/"+config["SCENARIO"]+"/Policy_%d"%config["EPISODE"])                
    print('|Action| = ', env.n_actions)
    
    n_steps = 0
    episodeObjs = []
    for episode in range(config["EPISODE"]):
        state = env.reset()
        episodeReward = 0
        for i in range(config["TRAJLEN"]):
            action, prob, val = agent.choose_action(state)
            nextstate, reward = env.step(action)
            n_steps += 1
            episodeReward += reward
            agent.store_transition(state, action, prob, val, reward)
            if n_steps % config["LEARNFREQ"] == 0:
                agent.learn()
            state = nextstate

        episodeObjs.append(env.lastObj)
        print('episode: {:>3}'.format(episode),' reward: {:>5.1f} '.format(episodeReward), ' objective: {:>6.1f} '.format(env.lastObj))
        
    
    if config["SAVERESULT"]:
        np.savez_compressed('Results/%s/res_DRL'%config["SCENARIO"], res=episodeObjs)
    if config["SAVEPOLICY"]:
        agent.save_models()

if __name__ == '__main__':
    
    config = {"DATASET": "ACF7",
              "SCENARIO": "ACF7-SC1",
              "SEED": 1,
              "EPISODE": 2000,
              "TRAJLEN": 5,
              "FC1DIMS": 256,
              "FC2DIMS": 256,
              "ALPHA": 0.001,
              "GAMMA": 0.9,
              "GAELAMBDA": 0.95,
              "POLICYCLIP": 0.2,
              "BATCHSIZE": 64,
              "NEPOCH": 10,
              "LEARNFREQ": 20,                            
              "SAVERESULT": True,
              "SAVEPOLICY": True
              }

    train_and_test(config)








