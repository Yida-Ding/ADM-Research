import os
import itertools
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
tf.autograph.set_verbosity(0)
from NetworkGenerator import Scenario
from VNSSolver import VNSSolver


class PolicyGradientNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super().__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.pi = Dense(n_actions, activation='softmax')

    def call(self, state):      # feed-forward a state to the PGN
        value1 = self.fc1(state)
        value2 = self.fc2(value1)
        pi = self.pi(value2)
        return pi

class Agent:
    def __init__(self, n_actions, lr, gamma, fc1_dims, fc2_dims, policy=None):
        self.gamma = gamma
        self.lr = lr
        self.n_actions = n_actions
        self.state_memory, self.action_memory, self.reward_memory = [], [], []
        self.policy = PolicyGradientNetwork(n_actions, fc1_dims, fc2_dims)    # policy is keras.Model object, with compile function
        self.policy.compile(optimizer=Adam(learning_rate=self.lr))
        if policy != None:
            self.policy.load_weights('Results/%s/%s/%s'%(policy[0], policy[1], policy[1]))

    def choose_action(self, state):
        probs = self.policy(state)      # same as self.policy.call(state), i.e. feed-forward
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()
        if action.numpy()[0] == self.n_actions:
            return self.n_actions-1
        return action.numpy()[0]

    def store_transition(self, state, action, reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        rewards = np.array(self.reward_memory)
        
        # G: 1-D array, G[t]: sum of discounted future rewards from t
        G = np.zeros_like(rewards)          
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        
        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss += -g * tf.squeeze(log_prob)

        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))
        self.state_memory, self.action_memory, self.reward_memory = [], [], []
        
'''
state_string: (Ps, Qs, Pdist, Qdist) where Xdist is the distance distribution based on Xs
state_tensor: 2-D tensor, each row is flight string without airports with padding zeros, with left inserted column distance distribution
action_tuple: (i, j, (Pind1, Pind2), (Qind1, Qind2))  where i, j in {0-pass, 1-swap, 2-cut, 3-insert}
action_idx: scalar, from 0 to n_actions-1
'''

class ADMEnvironment(VNSSolver):
    def __init__(self, S, seed):
        super().__init__(S, seed)
        self.flt2idx = {flt:idx+1 for idx, flt in enumerate(self.flts)}
        self.ap2idx = {ap:idx+1 for idx, ap in enumerate(self.S.airports)}
        self.skdPdist = self.getStringDistribution(self.skdPs) 
        self.skdQdist = self.getStringDistribution(self.skdQs)
        self.skdStrState = (self.skdPs, self.skdQs, self.skdPdist, self.skdQdist)
    
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
        state2D = tf.concat([distcol,tf.ragged.constant(flightidx, dtype=tf.float32).to_tensor()],1)
        state1D = tf.reshape(state2D, [-1])
        return state2D
        
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
        

# EXISTPOLICY = None : training from pure randomness,
# EXISTPOLICY = (scenario, policy) : transfer learning with a pretrained policy network
def train_and_test(config):
    tf.random.set_seed(seed = config["SEED"])
    S = Scenario(config["DATASET"], config["SCENARIO"], "PAX")
    env = ADMEnvironment(S, config["SEED"])    
    agent = Agent(n_actions=env.n_actions, lr=config["ALPHA"], gamma=config["GAMMA"], \
                  fc1_dims=config["FC1DIMS"], fc2_dims=config["FC2DIMS"], policy=config["EXISTPOLICY"])
    print('|Action| = ', env.n_actions)
    
    episodeObjs = []
    for episode in range(config["EPISODE"]):
        state = env.reset()
        episodeReward = 0
        for i in range(config["TRAJLEN"]):
            action = agent.choose_action(state)
            nextstate, reward = env.step(action)
            agent.store_transition(state, action, reward)
            state = nextstate
            episodeReward += reward
            
        agent.learn()
        episodeObjs.append(env.lastObj)
        print('episode: {:>3}'.format(episode),' reward: {:>5.1f} '.format(episodeReward), ' objective: {:>6.1f} '.format(env.lastObj))
    
    if config["SAVERESULT"]:
        np.savez_compressed('Results/%s/res_RFC'%config["SCENARIO"], res=episodeObjs)
    if config["SAVEPOLICY"]:
        if not os.path.exists('Results/%s/Policy_%s'%(config["SCENARIO"], config["EPISODE"])):
            os.makedirs('Results/%s/Policy_%s'%(config["SCENARIO"], config["EPISODE"]))
        agent.policy.save_weights('Results/%s/Policy_%s/Policy_%s'%(config["SCENARIO"], config["EPISODE"], config["EPISODE"]))
        

if __name__ == '__main__':
    
    config = {"DATASET": "ACF7",
              "SCENARIO": "ACF7-SC1",
              "SEED": 0,
              "EPISODE": 2500,
              "TRAJLEN": 5,
              "ALPHA": 0.001,
              "GAMMA": 0.9,
              "FC1DIMS": 256,
              "FC2DIMS": 256,
              "EXISTPOLICY": None,
              "SAVERESULT": True,
              "SAVEPOLICY": False
              }
    
    train_and_test(config)
