import random
import io
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import copy
import json
import networkx as nx
from NetworkGenerator import Scenario
import multiprocessing
import os

class VNSSolver:
    def __init__(self,S,seed,baseline="distance",enumFlag=False):
        self.S=S
        random.seed(seed)
        np.random.seed(seed)
        self.node=S.name2FNode
        self.flts=list(S.flight2dict.keys())
        self.tails=list(S.tail2flights.keys())
        self.tail2cap=S.tail2capacity
        self.tail2srcdest={tail:[self.node[flts[0]].Ori,self.node[flts[-1]].Des] for tail,flts in S.tail2flights.items()}
        self.crews=list(S.crew2flights.keys())
        self.crew2srcdest={crew:[self.node[flts[0]].Ori,self.node[flts[-1]].Des] for crew,flts in S.crew2flights.items()}        
        self.skdPs=[[self.tail2srcdest[tail][0]]+flts+[self.tail2srcdest[tail][1]] for tail,flts in S.tail2flights.items()]   # e.g. P=[LAX,F00,F01,F02,ORD]; Ps=[P1,P2,P3,..]
        self.skdQs=[[self.crew2srcdest[crew][0]]+flts+[self.crew2srcdest[crew][1]] for crew,flts in S.crew2flights.items()]
        self.k2func={0:"pass_",1:"swap",2:"cut",3:"insert"}
        self.baseline=baseline
        self.enumFlag=enumFlag
        
        # three types of baseline VNS
        undirectGraph=self.S.connectableGraph.to_undirected()
        if baseline=="uniform": # uniform probability to be operated
            self.node2weight={self.S.FNode2name[node]:1 for node in self.S.FNodes}
        elif baseline=="degree": # larger degree indicates higher probability to be operated
            self.node2weight={self.S.FNode2name[node]:deg for node,deg in undirectGraph.degree()}
        elif baseline=="distance":  # smaller distance indicates higher probability to be operated
            self.node2weight={}
            for drpNode in self.S.drpFNodes:
                node2distance=nx.single_source_dijkstra_path_length(undirectGraph,drpNode,cutoff=None,weight='weight')
                for node,dis in node2distance.items():
                    if self.S.FNode2name[node] not in self.node2weight:
                        self.node2weight[self.S.FNode2name[node]]=-1*dis
                    elif self.node2weight[self.S.FNode2name[node]]<-1*dis:
                        self.node2weight[self.S.FNode2name[node]]=-1*dis
                        
    def checkConnections(self,pairs): #pairs=[(P1,P2)] or [(Q1,Q2)]
        flag=True
        for pair in pairs:
            if pair[0] not in self.flts and pair[1] not in self.flts:
                flag=False
            elif pair[0] not in self.flts and pair[0]!=self.node[pair[1]].Ori:
                flag=False
            elif pair[1] not in self.flts and pair[1]!=self.node[pair[0]].Des:
                flag=False
            elif pair[0] in self.flts and pair[1] in self.flts and (self.node[pair[0]].Des!=self.node[pair[1]].Ori or self.node[pair[1]].LDT<self.node[pair[0]].CT+self.node[pair[0]].EAT):
                flag=False
        return flag
        
    def pass_(self,X1,X2):
        return [(X1,X2)]
    
    def swap(self,X1,X2):
        pairs=[]
        for u in range(1,len(X1)-1):
            for v in range(1,len(X2)-1):
                if self.checkConnections([(X1[u-1],X2[v]),(X2[v-1],X1[u]),(X2[v],X1[u+1]),(X1[u],X2[v+1])]):
                    pairs.append((X1[:u]+[X2[v]]+X1[u+1:],X2[:v]+[X1[u]]+X2[v+1:]))
        return pairs
    
    def cut(self,X1,X2):
        pairs=[]
        for u1 in range(1,len(X1)-1):
            for u2 in range(u1,len(X1)-1):
                for v1 in range(1,len(X2)-1):
                    for v2 in range(v1,len(X2)-1):
                        if self.checkConnections([(X1[u1-1],X2[v1]),(X2[v1-1],X1[u1]),(X2[v2],X1[u2+1]),(X1[u2],X2[v2+1])]):
                            pairs.append((X1[:u1]+X2[v1:v2+1]+X1[u2+1:],X2[:v1]+X1[u1:u2+1]+X2[v2+1:]))
        return pairs
    
    def insert(self,X1,X2):
        pairs=[]
        for u in range(1,len(X1)-1):
            for v1 in range(1,len(X2)-1):
                for v2 in range(v1+1,len(X2)-1):
                    if self.checkConnections([(X1[u-1],X2[v1]),(X2[v2],X1[u]),(X2[v1-1],X2[v2+1])]):
                        pairs.append((X1[:u]+X2[v1:v2+1]+X1[u:],X2[:v1]+X2[v2+1:]))
        return pairs


    def evaluate(self,Ps,Qs):
        flt2tail={flt:self.tails[i] for i,P in enumerate(Ps) for flt in P[1:-1]}
        # find the father flight for each flight based on Qs and multiple hop schedule itin
        flt2father={}
        for Q in Qs:
            flts=Q[1:-1]
            flt2father[flts[0]]=None
            for i in range(1,len(flts)):
                flt2father[flts[i]]=flts[i-1]
        for fltleg in self.S.fltlegs2itin.keys():
            flts=fltleg.split('-')
            if len(flts)>1:
                for i in range(1,len(flts)):
                    flt2father[flts[i]]=flts[i-1]
                            
        # greedy assignment of dt&at for each flight based on the structure of Ps
        timeDict={flt:[0,0] for flt in self.flts}
        for P in Ps:
            for i in range(len(P)):
                if i==1:
                    timeDict[P[1]]=[self.node[P[1]].SDT,self.node[P[1]].SAT]
                elif i>1 and i<len(P)-1:
                    timeDict[P[i]][0]=max(timeDict[P[i-1]][1]+self.S.config["ACMINCONTIME"],self.node[P[i]].SDT)
                    timeDict[P[i]][1]=timeDict[P[i]][0]+self.node[P[i]].SFT
        
        # repair timeDict based on the structure of Qs
        for flt,father in flt2father.items():
            if father!=None:
                fatherAt=timeDict[father][1]
                timeDict[flt][0]=max(timeDict[flt][0],fatherAt+self.S.config["CREWMINCONTIME"])
                timeDict[flt][1]=timeDict[flt][0]+self.node[flt].SFT
                
        # feasibility check for crews: check conflict/unconnected flights based on current timeDict and Qs
        for Q in Qs:
            curArrTime=self.S.config["STARTTIME"]
            for flt in Q[1:-1]:
                if curArrTime+self.node[flt].CT>timeDict[flt][0]:
                    return np.inf,None,None,None
                else:
                    curArrTime=timeDict[flt][1]
        
        # the delay of itin is the actual arrival time of itin minus the schedule arrival time of itin
        itinDelay=[(itin,timeDict[self.S.itin2flights[itin][-1]][1]-self.S.itin2skdtime[itin][1]) for itin in self.S.itin2flights.keys()]
        itinDelay=sorted(itinDelay,key=lambda xx:xx[1],reverse=True)
        itin2pax=self.S.itin2pax.copy()
        bothItin2pax=defaultdict(int); itin2flowin=defaultdict(int)
        for itin1, delay1 in itinDelay:
            # Case1: not rerouted
            minCost = self.S.config["DELAYCOST"]*itin2pax[itin1]*delay1
            minflt2 = None            
            
            flts1 = self.S.itin2flights[itin1]
            for flt2 in self.flts:
                # condition for replaceable flt2: current arrival time of flt2 should be earlier than current arrival time of itin1
                if self.S.itin2origin[itin1] == self.node[flt2].Ori and self.S.itin2destination[itin1] == self.node[flt2].Des and timeDict[flt2][1] < timeDict[flts1[-1]][1]:
                    paxFlt2 = sum([itin2pax[itin] for itin in self.S.flt2skditins[flt2]])
                    remain2 = self.tail2cap[flt2tail[flt2]]-paxFlt2
                    leave = min(remain2,itin2pax[itin1])
                    remain1 = itin2pax[itin1]-leave                    
                    # Case2: reroute with a late-depart flight. The replaceable flt2 has later departure time than the schedule departure time of itin1, so the leave in itin1 can be sure to catch the flt2, but they will experience arrival delay than schedule 
                    if timeDict[flt2][0] > self.S.itin2skdtime[itin1][0]:
                        # the timing of flt2 remains unchanged
                        timeFlt2 = [timeDict[flt2][0],timeDict[flt2][1]] 
                        # sign operation: if positive keep it, if negative set it to 0
                        cost = self.S.config["DELAYCOST"]*leave*max(timeFlt2[1]-self.S.itin2skdtime[itin1][1],0) \
                             + self.S.config["DELAYCOST"]*remain1*delay1
                    # Case3: reroute with an early flight: # The replaceable flt2 has earlier departure time than the schedule departure time of itin1, so we need to change the time of flight2 to schedule departure time of itin1
                    else: 
                        # the timing of flt2 is aligned to the schedule departure time of itin1
                        timeFlt2 = [self.S.itin2skdtime[itin1][0],self.S.itin2skdtime[itin1][0]+self.node[flt2].SFT] 
                        cost = self.S.config["DELAYCOST"]*remain1*delay1 \
                             + self.S.config["DELAYCOST"]*paxFlt2*max(timeFlt2[1]-self.node[flt2].ScheduleAT,0) \
                             + self.S.config["DELAYCOST"]*leave*max(timeFlt2[1]-self.S.itin2skdtime[itin1][1],0)
                             
                    if cost < minCost:
                        minCost = cost
                        minflt2 = flt2
                        minTimeFlt2 = timeFlt2
                        minLeave = leave
                            
            if minflt2 != None:
                itin2 = self.S.fltlegs2itin[minflt2]
                timeDict[minflt2] = minTimeFlt2
                itin2flowin[itin2] += minLeave
                itin2pax[itin1] -= minLeave
                itin2pax[itin2] += minLeave
                bothItin2pax[(itin2,itin1)] = minLeave
            
            # feasibility check for itin1: check conflict or unconnected flights 
            if itin2pax[itin1]>0:
                curAt=self.S.config["STARTTIME"]
                for flt in flts1:
                    if curAt+self.node[flt].CT>timeDict[flt][0]:
                        return np.inf,None,None,None
                    else:
                        curAt=timeDict[flt][1]
                
            bothItin2pax[(itin1,itin1)]=itin2pax[itin1]-itin2flowin.get(itin1,0)
        
        # compute paxDict
        paxDict=defaultdict(int)
        for recItin,pax in itin2pax.items():
            for flt in self.S.itin2flights[recItin]:
                paxDict[flt]+=pax         
                
        # feasibility check for tail capacity
        for flt,tail in flt2tail.items():
            if paxDict[flt]>self.tail2cap[tail]:
                return np.inf,None,None,None
        
        # compute delay cost
        delayCost=0
        for bothItin,pax in bothItin2pax.items():
            recItin,skdItin=bothItin
            delay=timeDict[self.S.itin2flights[recItin][-1]][1]-self.S.itin2skdtime[skdItin][1]
            if delay<0:
                delay=0
            delayCost+=self.S.config["DELAYCOST"]*delay*pax
        
        # compute follow gain
        followCost=0
        for i,skdFlts in enumerate(self.S.tail2flights.values()):
            recFlts=Ps[i][1:-1]
            followCost+=self.S.config["FOLLOWSCHEDULECOST"]
            for j in range(len(skdFlts)):
                if j<len(recFlts) and skdFlts[j]==recFlts[j]:
                    followCost+=self.S.config["FOLLOWSCHEDULECOST"]
        for i,skdFlts in enumerate(self.S.crew2flights.values()):
            recFlts=Qs[i][1:-1]
            followCost+=self.S.config["FOLLOWSCHEDULECOST"]
            for j in range(len(skdFlts)):
                if j<len(recFlts) and skdFlts[j]==recFlts[j]:
                    followCost+=self.S.config["FOLLOWSCHEDULECOST"]
        for bothItin,pax in bothItin2pax.items():
            recItin,skdItin=bothItin
            if recItin==skdItin:
                followCost+=self.S.config["FOLLOWSCHEDULECOSTPAX"]*pax*(len(self.S.itin2flights[skdItin])+1)            
        
        objective=delayCost+followCost
        return objective,timeDict,bothItin2pax,paxDict,delayCost,followCost
    
    def generateVNSRecoveryPlan(self,minPs,minQs,minRes):
        objective,timeDict,bothItin2pax,paxDict,delayCost,followCost=minRes
        flt2crew={flt:self.crews[i] for i,Q in enumerate(minQs) for flt in Q[1:-1]}
        
        resd=defaultdict(list)
        for i,P in enumerate(minPs):
            for flight in P[1:-1]:
                resd["Tail"].append(self.tails[i])
                resd["Flight"].append(flight)
                resd["Crew"].append(flt2crew[flight])
                resd["From"].append(self.node[flight].Ori)
                resd["To"].append(self.node[flight].Des)
                resd["RDT"].append(timeDict[flight][0])
                resd["RAT"].append(timeDict[flight][1])
                resd["Flight_time"].append(timeDict[flight][1]-timeDict[flight][0])
                resd["Capacity"].append(self.tail2cap[self.tails[i]])
                resd["Pax"].append(paxDict[flight])
                resd["Timestring"].append(self.S.getTimeString(timeDict[flight][0])+" -> "+self.S.getTimeString(timeDict[flight][1]))
                arrdelay=round(resd["RAT"][-1]-self.S.flight2scheduleAT[flight])
                resd["Delay"].append(self.S.getTimeString(arrdelay) if arrdelay>0 else '')
                if flight in self.S.disruptedFlights:
                    resd["DelayType"].append(1)
                elif arrdelay>0:
                    resd["DelayType"].append(2)
                else:
                    resd["DelayType"].append(0)
                    
        resdItin=defaultdict(list)
        for bothItin,pax in bothItin2pax.items():
            recItin,skdItin=bothItin
            recFlts=self.S.itin2flights[recItin]
            resdItin["Rec_itin"].append(recItin)
            resdItin["Skd_itin"].append(skdItin)
            resdItin["Rec_flights"].append('-'.join(recFlts))
            resdItin["Skd_flights"].append('-'.join(self.S.itin2flights[skdItin]))
            resdItin["Pax"].append(pax)
            resdItin["From"].append(self.S.itin2origin[skdItin])
            resdItin["To"].append(self.S.itin2destination[skdItin])
            resdItin["RDT"].append(timeDict[recFlts[0]][0])
            resdItin["RAT"].append(timeDict[recFlts[-1]][1])
            delay=timeDict[recFlts[-1]][1]-self.S.itin2skdtime[skdItin][1]
            if delay<0:
                delay=0
            resdItin["Arr_delay"].append(delay)
            resdItin["Cost"].append("%.1f"%(self.S.config["DELAYCOST"]*pax*delay))
            resdItin["Rec_string"].append(self.S.getTimeString(timeDict[recFlts[0]][0])+' -> '+self.S.getTimeString(timeDict[recFlts[-1]][1]))
                    
        dfrecovery=pd.DataFrame(resd)
        dfrecovery.to_csv("Results/"+self.S.scname+"/RecoveryVNS.csv",index=None)
        dfitin=pd.DataFrame(resdItin).sort_values(by=["Rec_itin","Skd_itin"])
        dfitin.to_csv("Results/"+self.S.scname+"/ItineraryVNS.csv",index=None)
        costDict={"DelayCost":delayCost,"ExtraFuelCost":0.,"CancelCost":0.,"FollowGain":followCost,"Objective":objective}
        with open("Results/%s/CostVNS.json"%(self.S.scname), "w") as outfile:
            json.dump(costDict, outfile, indent = 4)
    
    def getStringDistribution(self,Xs):
        Xweights=np.array([np.mean([self.node2weight[fnode] for fnode in x[1:-1]]) for x in Xs])
        if self.baseline!="uniform":
           Xweights=(Xweights-Xweights.min())/(Xweights.max()-Xweights.min())
        Xdist=np.exp(Xweights)/np.exp(Xweights).sum()
        return Xdist
    
    def transformStrings(self,k,curXs,Xdist):
        Xind1,Xind2=np.random.choice(np.arange(len(curXs)),size=2,replace=False,p=Xdist)
        Xpairs=eval("self."+self.k2func[k])(curXs[Xind1],curXs[Xind2])
        if len(Xpairs)>=1:
            nX1,nX2=random.choice(Xpairs)
            nXs=curXs.copy()
            nXs[Xind1],nXs[Xind2]=nX1,nX2
        else:
            nXs=curXs
        return nXs
        
    def exploreNeighborK(self,k,curPs,curQs,Pdist,Qdist):
        # sample the pair based on metrics, without enumeration through operating choices
        if self.enumFlag==False:            
            nPs=self.transformStrings(k,curPs,Pdist)
            nQs=self.transformStrings(k,curQs,Qdist)
            nRes=self.evaluate(nPs,nQs)
            return nPs,nQs,nRes

        # random sample the pair + full enumeration through operating choices
        else:
            curRes=self.evaluate(curPs,curQs)
            Pind1,Pind2=random.sample(range(len(curPs)),2)
            for (nP1,nP2) in eval("self."+self.k2func[k])(curPs[Pind1],curPs[Pind2]):
                nPs=curPs.copy()
                nPs[Pind1],nPs[Pind2]=nP1,nP2
                for i in range(len(curQs)):
                    Qind1,Qind2=random.sample(range(len(curQs)),2)        
                    for (nQ1,nQ2) in eval("self."+self.k2func[k])(curQs[Qind1],curQs[Qind2]):
                        nQs=curQs.copy()
                        nQs[Qind1],nQs[Qind2]=nQ1,nQ2
                        nRes=self.evaluate(nPs,nQs)
                        if nRes[0]<curRes[0]:
                            curPs,curQs,curRes=nPs,nQs,nRes
            return curPs,curQs,curRes    

    def VNS(self,trajLen):
        minPs,minQs=self.skdPs,self.skdQs
        minRes=self.evaluate(minPs,minQs)
        for i in range(trajLen):
            k=0
            while k<=3:
                Pdist=self.getStringDistribution(minPs)
                Qdist=self.getStringDistribution(minQs)
                curPs,curQs,curRes=self.exploreNeighborK(k,minPs,minQs,Pdist,Qdist)
                if curRes[0]<minRes[0]:
                    minPs,minQs,minRes=curPs,curQs,curRes
                else:
                    k+=1
        return minPs,minQs,minRes

def runVNSWithEnumSaveSolution(config):
    S=Scenario(config["DATASET"],config["SCENARIO"],"PAX")
    solver=VNSSolver(S,0,config["BASELINE"],config["ENUMFLAG"])
    res=solver.VNS(config["TRAJLEN"])
    print(res[2][0])
    solver.generateVNSRecoveryPlan(*res)
    
def runVNS(par):
    
    config={"DATASET": "ACF%d"%par[0],
            "SCENARIO": "ACF%d-SC%d"%(par[0],par[1]),
            "BASELINE": par[2],
            "TRAJLEN": 10,
            "ENUMFLAG": False,
            "EPISODES": 100
            }
    
    if not os.path.exists("Results"):
        os.makedirs("Results")
    if not os.path.exists("Results/%s"%config["SCENARIO"]):
        os.makedirs("Results/%s"%config["SCENARIO"])
        
    res=[]
    times=[]
    S=Scenario(config["DATASET"],config["SCENARIO"],"PAX")
    for seed in range(config["EPISODES"]):
        T1=time.time()
        solver=VNSSolver(S,seed,config["BASELINE"],config["ENUMFLAG"])
        objective=solver.VNS(config["TRAJLEN"])[2][0]
        T2=time.time()
        res.append(objective)
        times.append(T2-T1)
        print(config["SCENARIO"],'episode: {:>3}'.format(seed), ' objective: {:>6.1f} '.format(objective),'best:',"%.1f"%min(res))

    np.savez_compressed('Results/%s/res_%s'%(config["SCENARIO"],config["BASELINE"]),res=res)
    np.savez_compressed('Results/%s/time_%s'%(config["SCENARIO"],config["BASELINE"]),res=times)

if __name__ == '__main__':
    
    p=multiprocessing.Pool()
    todo=[(i,typ,m) for i in range(5,25,5) for typ in [0,1,2] for m in ["degree","uniform","distance"]]
    t1=time.time()
    p=multiprocessing.Pool()
    for i,res in enumerate(p.imap_unordered(runVNS,todo),1):
        t2=time.time()
        print("%0.2f%% done, time to finish: ~%0.1f minutes"%(i*100/len(todo),((t2-t1)/(60*(i+1)))*len(todo)-((t2-t1)/60)))
            
