import os
import numpy as np
import itertools
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from NetworkGenerator import Scenario
from VNSSolver import VNSSolver
import time
import heapq

class ADMEnvironment(VNSSolver):
    def __init__(self, S, seed, npar):
        super().__init__(S, seed)
        self.flt2idx = {flt:idx+1 for idx, flt in enumerate(self.flts)}
        self.ap2idx = {ap:idx+1 for idx, ap in enumerate(self.S.airports)}
        self.skdStrState = (self.skdPs, self.skdQs)
        self.stateShape = self.string2tensor(self.skdStrState).shape
        Pinds = list(range(len(self.skdPs)))
        Qinds = list(range(len(self.skdQs)))
        
        if npar != None:
            Pdist = self.getStringDistribution(self.skdPs)
            Qdist = self.getStringDistribution(self.skdQs)
            Pinds = heapq.nlargest(npar[0], range(len(Pdist)), Pdist.take)
            Qinds = heapq.nlargest(npar[1], range(len(Qdist)), Qdist.take)
            
        pindpairs = itertools.combinations(Pinds, 2)
        qindpairs = itertools.combinations(Qinds, 2)
        combs = itertools.product(range(4), range(4), pindpairs, qindpairs)
        self.idx2action = {idx:comb for idx, comb in enumerate(combs)}
        self.n_actions = len(self.idx2action)
    
    def reset(self):
        self.lastStrState = self.skdStrState
        self.lastObj = self.evaluate(*self.skdStrState)[0]
        return self.string2tensor(self.skdStrState)

    def string2tensor(self, strState):
        flightidx = [[self.flt2idx[flt] for flt in P[1:-1]] for P in strState[0]+strState[1]]
        padarray = np.array([L+[0]*(10-len(L)) for L in flightidx]).flatten()
        return padarray
        
    def step(self, action_idx):
        k1, k2, pindpair, qindpair = self.idx2action[action_idx]
        curPs, curQs = self.lastStrState
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
        self.lastStrState = (curPs, curQs)
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

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims, fc2_dims, dire, existpolicy):
        super(ActorNetwork, self).__init__()
        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1))
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.dire = dire
        if existpolicy != None:
            self.load_state_dict(T.load("Results/"+existpolicy+"/Policy/actor.pt"))
            self.eval()

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist
    
    def save_model(self):
        T.save(self.state_dict(), self.dire+"/actor.pt")
        
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims, fc2_dims, dire, existpolicy):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.dire = dire
        if existpolicy != None:
            self.load_state_dict(T.load("Results/"+existpolicy+"/Policy/critic.pt"))
            self.eval()

    def forward(self, state):
        value = self.critic(state)
        return value    

    def save_model(self):
        T.save(self.state_dict(), self.dire+"/critic.pt")

class Agent:
    def __init__(self, n_actions, input_dims, fc1_dims, fc2_dims, gamma, alpha, \
                 gae_lambda, policy_clip, batch_size, n_epochs, dire, existpolicy):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.actor = ActorNetwork(n_actions, input_dims, alpha, fc1_dims, fc2_dims, dire, existpolicy)
        self.critic = CriticNetwork(input_dims, alpha, fc1_dims, fc2_dims, dire, existpolicy)            
        self.memory = PPOMemory(batch_size)        
        
    def choose_action(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        return action, probs, value
        
    def store_transition(self, state, action, probs, vals, reward):
        self.memory.store_memory(state, action, probs, vals, reward)

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, batches = \
                    self.memory.generate_batches()
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()             
    
def trainPPO(config):
    T.manual_seed(config["SEED"])
    S = Scenario(config["DATASET"], config["SCENARIO"], "PAX")
    env = ADMEnvironment(S, config["SEED"], config["NSTRING"])
    dire = "Results/"+config["SCENARIO"]+"/Policy"
    agent = Agent(n_actions=env.n_actions, input_dims=env.stateShape, fc1_dims=config["FC1DIMS"], fc2_dims=config["FC2DIMS"], \
                  gamma=config["GAMMA"], alpha=config["ALPHA"], gae_lambda=config["GAELAMBDA"], policy_clip=config["POLICYCLIP"], \
                  batch_size=config["BATCHSIZE"], n_epochs=config["NEPOCH"], dire=dire, existpolicy=config["EXISTPOLICY"])             
    print('|Action| = ', env.n_actions)
    
    episodeObjs = []
    times = []
    for episode in range(config["EPISODE"]):
        T1=time.time()
        state = env.reset()
        episodeReward = 0
        for i in range(config["TRAJLEN"]):
            action, prob, val = agent.choose_action(state)
            nextstate, reward = env.step(action)
            episodeReward += reward
            agent.store_transition(state, action, prob, val, reward)
            state = nextstate

        episodeObjs.append(env.lastObj)
        print(config["SCENARIO"], 'episode: {:>3}'.format(episode),' reward: {:>7.1f} '.format(episodeReward), ' objective: {:>7.1f} '.format(env.lastObj))
        agent.learn()
        T2=time.time()
        times.append(T2-T1)
    
    if config["SAVERESULT"]:
        if not os.path.exists("Results/"+config["SCENARIO"]):
            os.makedirs("Results/"+config["SCENARIO"])
        np.savez_compressed('Results/%s/res_PPO'%(config["SCENARIO"]), res=episodeObjs)
        np.savez_compressed('Results/%s/time_PPO'%(config["SCENARIO"]), res=times)
    if config["SAVEPOLICY"]:
        if not os.path.exists(dire):
            os.makedirs(dire)
        agent.actor.save_model()
        agent.critic.save_model()
    
    del agent.actor
    del agent.critic
    T.cuda.empty_cache() 


if __name__ == '__main__':
    

    config = {"DATASET": "ACF%d"%i,
              "SCENARIO": "ACF%d-SC%d"%(i,j),
              "SEED": 1,
              "EPISODE": 5000,
              "TRAJLEN": 5,
              "FC1DIMS": 256,
              "FC2DIMS": 256,
              "ALPHA": 0.00001, 
              "GAMMA": 0.9,
              "GAELAMBDA": 0.95,
              "POLICYCLIP": 0.8,
              "BATCHSIZE": 5,
              "NEPOCH": 3,
              "NSTRING": None,
              "SAVERESULT": True,
              "SAVEPOLICY": False,
              "EXISTPOLICY": None
              }
    
    trainPPO(config)
            
            
