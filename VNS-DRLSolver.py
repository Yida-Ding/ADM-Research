import os
import itertools
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from NetworkGenerator import Scenario
from VNSSolver import VNSSolver


'''
state_string: (Ps, Qs)
state_tensor: 2-D tensor, each row is flight string without airports, with padding zeros
action_tuple: (i, j, (Pind1, Pind2), (Qind1, Qind2))  # i, j in {0-pass, 1-swap, 2-cut, 3-insert}
action_idx: scalar, n_actions

'''

class PolicyGradientNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
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
    def __init__(self, n_actions, lr=0.003, gamma=0.99, layer1_size=256, layer2_size=256):
        self.gamma = gamma
        self.lr = lr
        self.n_actions = n_actions
        self.state_memory, self.action_memory, self.reward_memory = [], [], []
        self.policy = PolicyGradientNetwork(n_actions=n_actions)    # policy is keras.Model object, with compile function
        self.policy.compile(optimizer=Adam(learning_rate=self.lr))

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
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float64)
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
                state = tf.convert_to_tensor([state], dtype=tf.float64)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss += -g * tf.squeeze(log_prob)

        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))
        self.state_memory, self.action_memory, self.reward_memory = [], [], []

class ADMEnvironment(VNSSolver):
    def __init__(self, S, seed):
        super().__init__(S, seed)
        self.flt2idx = {flt:idx+1 for idx, flt in enumerate(self.flts)}
        self.ap2idx = {ap:idx+1 for idx, ap in enumerate(self.S.airports)}
        self.skdStrState = (self.skdPs, self.skdQs)
    
        # prepare action_tuples and idx2action                
        pindpairs = itertools.combinations(range(len(self.skdPs)), 2)
        qindpairs = itertools.combinations(range(len(self.skdQs)), 2)
        combs = itertools.product(range(4), range(4), pindpairs, qindpairs)
        self.idx2action = {idx:comb for idx, comb in enumerate(combs)}
        self.n_actions = len(self.idx2action)        
    
    def reset(self):
        self.lastStrState = self.skdStrState
        self.lastObj = self.evaluate(*self.skdStrState)[0]
        return self.string2tensor(self.skdStrState)

    def string2tensor(self, strState):
        res = [[self.flt2idx[flt] for flt in P[1:-1]] for P in strState[0]+strState[1]]
        return tf.ragged.constant(res).to_tensor()
        
    def step(self,action_idx):
        k1, k2, pindpair, qindpair = self.idx2action[action_idx]
        curPs, curQs = self.lastStrState
        curObj = self.lastObj
        
        for (nP1,nP2) in eval("self."+self.k2func[k1])(curPs[pindpair[0]], curPs[pindpair[1]]):
            nPs = curPs.copy()
            nPs[pindpair[0]], nPs[pindpair[1]] = nP1, nP2
            for (nQ1,nQ2) in eval("self."+self.k2func[k2])(curQs[qindpair[0]], curQs[qindpair[1]]):
                nQs = curQs.copy()
                nQs[qindpair[0]], nQs[qindpair[1]] = nQ1, nQ2
                nObj = self.evaluate(nPs, nQs)[0]
                if nObj < curObj:
                    curPs, curQs, curObj = nPs, nQs, nObj
        
        reward = self.lastObj - curObj
        self.lastStrState = (curPs, curQs)
        self.lastObj = curObj
        return self.string2tensor(self.lastStrState), reward
        
            

if __name__ == '__main__':
    S = Scenario("ACF7","ACF7-SC1","PAX")
    env = ADMEnvironment(S,0)
    agent = Agent(n_actions=env.n_actions, lr=0.001, gamma=0.9)
    print('number of actions: ', env.n_actions)
    
    res = []
    for episode in range(2200):
        state = env.reset()
        totalreward = 0
        for i in range(5):
            action = agent.choose_action(state)
            nextstate, reward = env.step(action)
            agent.store_transition(state, action, reward)
            state = nextstate
            totalreward += reward
            
        agent.learn()
        res.append(env.lastObj)
        print('episode: {:>3}'.format(episode),' reward: {:>5.1f} '.format(totalreward), ' objective: {:>6.1f} '.format(env.lastObj))
    
    np.savez_compressed('Results/ACF7-SC1/res_DRL',res=res)








