import random
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import copy
from NetworkGenerator import Scenario


class VNSSolver:
    def __init__(self,S,seed):
        self.S=S
        random.seed(seed)
        self.node=S.name2FNode
        self.flts=list(S.flight2dict.keys())
        self.tails=list(S.tail2flights.keys())
        self.tail2cap=S.tail2capacity
        self.tail2srcdest={tail:[self.node[flts[0]].Ori,self.node[flts[-1]].Des] for tail,flts in S.tail2flights.items()}
        self.crews=list(S.crew2flights.keys())
        self.crew2srcdest={crew:[self.node[flts[0]].Ori,self.node[flts[-1]].Des] for crew,flts in S.crew2flights.items()}        
        self.skdPs=[[self.tail2srcdest[tail][0]]+flts+[self.tail2srcdest[tail][1]] for tail,flts in S.tail2flights.items()]   # e.g. P=[LAX,F00,F01,F02,ORD]; Ps=[P1,P2,P3,..]
        self.skdQs=[[self.crew2srcdest[crew][0]]+flts+[self.crew2srcdest[crew][1]] for crew,flts in S.crew2flights.items()]
        self.k2func={1:"swap",2:"cross",3:"insert"}
        
        # process dfitinerary
        self.itins=self.S.dfitinerary["Itinerary"].tolist()
        self.itin2skdflts,self.itin2od,self.itin2skdtime,self.itin2skdpax,self.flt2skditins={},{},{},{},defaultdict(list)
        for row in self.S.dfitinerary.itertuples():
            self.itin2skdflts[row.Itinerary]=row.Flight_legs.split('-')
            self.itin2od[row.Itinerary]=(row.From,row.To)
            self.itin2skdtime[row.Itinerary]=(row.SDT,row.SAT)
            self.itin2skdpax[row.Itinerary]=row.Pax
            for flt in row.Flight_legs.split('-'):
                self.flt2skditins[flt].append(row.Itinerary)
            
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
    
    def swap(self,X1,X2):
        pairs=[]
        for u in range(1,len(X1)-1):
            for v in range(1,len(X2)-1):
                if self.checkConnections([(X1[u-1],X2[v]),(X2[v-1],X1[u]),(X2[v],X1[u+1]),(X1[u],X2[v+1])]):
                    pairs.append((X1[:u]+[X2[v]]+X1[u+1:],X2[:v]+[X1[u]]+X2[v+1:]))
        return pairs
    
    def cross(self,X1,X2):
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

    def exploreNeighborK(self,k,curPs,curQs):
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

    def VNS(self,numIt=10):
        minPs,minQs=self.skdPs,self.skdQs
        minRes=self.evaluate(minPs,minQs)
        for i in range(numIt):
            k=1
            while k<=3:
                curPs,curQs,curRes=self.exploreNeighborK(k,minPs,minQs)
                if curRes[0]<minRes[0]:
                    minPs,minQs,minRes=curPs,curQs,curRes
                else:
                    k+=1
        
        print("The minimum delay cost:",minRes[0])
        print("Tail assignment:",minPs)
        print("Crew assignment:",minQs)
        return minPs,minQs,minRes
    
    def generateVNSRecoveryPlan(self,minPs,minQs,minRes):
        resd=defaultdict(list)
        obj,flt2time,paxDict=minRes
        flt2crew={flt:self.crews[i] for i,Q in enumerate(minQs) for flt in Q[1:-1]}
        for i,P in enumerate(minPs):
            for flight in P[1:-1]:
                resd["Tail"].append(self.tails[i])
                resd["Flight"].append(flight)
                resd["Crew"].append(flt2crew[flight])
                resd["From"].append(self.node[flight].Ori)
                resd["To"].append(self.node[flight].Des)
                resd["RDT"].append(flt2time[flight][0])
                resd["RAT"].append(flt2time[flight][1])
                resd["Flight_time"].append(flt2time[flight][1]-flt2time[flight][0])
                resd["Capacity"].append(self.tail2cap[self.tails[i]])
                resd["Pax"].append(paxDict[flight])
                resd["Timestring"].append(self.S.getTimeString(flt2time[flight][0])+" -> "+self.S.getTimeString(flt2time[flight][1]))
                arrdelay=round(resd["RAT"][-1]-self.S.flight2scheduleAT[flight])
                resd["Delay"].append(self.S.getTimeString(arrdelay) if arrdelay>0 else '')
                if flight in self.S.disruptedFlights:
                    resd["DelayType"].append(1)
                elif arrdelay>0:
                    resd["DelayType"].append(2)
                else:
                    resd["DelayType"].append(0)
        
        dfrecovery=pd.DataFrame(resd)
        dfrecovery.to_csv("Results/"+self.S.scname+"/RecoveryVNS.csv",index=None)
        resdcost=defaultdict(list)
        resdcost["Cost"].append("delay_cost")
        resdcost["Value"].append(round(obj))
        dfcost=pd.DataFrame(resdcost)
        dfcost.to_csv("Results/"+self.S.scname+"/CostVNS.csv",index=None)
      
    def evaluate(self,Ps,Qs):
        flt2tail={flt:self.tails[i] for i,P in enumerate(Ps) for flt in P[1:-1]}
        # greedy assignment of dt&at for each flight based on the structure of Ps
        timeDict={flt:[0,0] for flt in self.flts}
        for P in Ps:
            for i in range(len(P)):
                if i==1:
                    timeDict[P[1]]=[self.node[P[1]].SDT,self.node[P[1]].SAT]
                elif i>1 and i<len(P)-1:
                    timeDict[P[i]][0]=max(timeDict[P[i-1]][1]+self.S.config["ACMINCONTIME"],self.node[P[i]].SDT)
                    timeDict[P[i]][1]=timeDict[P[i]][0]+self.node[P[i]].SFT
        timeDictCopy=timeDict.copy()
        
        # feasibility check for crews: check conflict/unconnected flights based on current flt2time and Qs
        for Q in Qs:
            curArrTime=self.S.config["STARTTIME"]
            for flt in Q[1:-1]:
                if curArrTime+self.node[flt].CT>timeDict[flt][0]:
                    return np.inf,None,None
                else:
                    curArrTime=timeDict[flt][1]
          
        # allocate passengers to minimize the actual delay cost.
        paxDict={flt:self.S.flt2pax[flt] for flt in self.flts}
        fltDelays=sorted([(flt,timeDict[flt][1]-self.node[flt].ScheduleAT) for flt in self.flts],key=lambda item:item[1],reverse=True)
        paxDelays=[]; flt2flowin=defaultdict(int)
        for flt1,delay1 in fltDelays:
            if delay1==0:
                break

            # not reroute
            minCost=self.S.config["DELAYCOST"]*paxDict[flt1]*(timeDict[flt1][1]-self.node[flt1].ScheduleAT)
            minflt2=None            
            for flt2 in self.flts:
                if flt2tail[flt1]!=flt2tail[flt2] and self.node[flt1].Ori==self.node[flt2].Ori and self.node[flt1].Des==self.node[flt2].Des and timeDict[flt1][1]>timeDict[flt2][1]:
                    remain2=self.tail2cap[flt2tail[flt2]]-paxDict[flt2]
                    leave=min(remain2,paxDict[flt1])
                    remain1=paxDict[flt1]-leave
                    if self.node[flt1].ScheduleDT<=timeDict[flt2][0]:
                        nflt2SDT,nflt2SAT=timeDict[flt2][0],timeDict[flt2][1]
                        cost=self.S.config["DELAYCOST"]*leave*(timeDict[flt2][1]-self.node[flt1].ScheduleAT)+self.S.config["DELAYCOST"]*remain1*(timeDict[flt1][1]-self.node[flt1].ScheduleAT)
                    else:
                        nflt2SDT,nflt2SAT=self.node[flt1].ScheduleDT,self.node[flt2].SAT-self.node[flt2].SDT+self.node[flt1].ScheduleDT
                        cost=self.S.config["DELAYCOST"]*leave*(nflt2SAT-self.node[flt1].ScheduleAT)+self.S.config["DELAYCOST"]*remain1*(timeDict[flt1][1]-self.node[flt1].ScheduleAT)+self.S.config["DELAYCOST"]*paxDict[flt2]*(nflt2SAT-self.node[flt2].ScheduleAT)
                    if cost<minCost:
                        minCost=cost
                        minflt2=flt2
                        minflt2SDT,minflt2SAT=nflt2SDT,nflt2SAT

            if minflt2!=None:
                timeDict[minflt2]=(minflt2SDT,minflt2SAT)
                paxDict[flt1]-=leave
                paxDict[minflt2]+=leave
                paxDelays.append((leave,timeDict[minflt2][1]-self.node[flt1].ScheduleAT,"%s-%s"%(flt1,minflt2)))
                flt2flowin[minflt2]+=leave
                
            paxDelays.append((paxDict[flt1]-flt2flowin.get(flt1,0)-self.S.flt2nonepax.get(flt1,0),delay1,flt1))
            
        # the cost related to modification of timeDict
        for flt in self.flts:
            delta=timeDict[flt][1]-timeDictCopy[flt][1]
            if delta!=0:
                paxDelays.append((paxDict[flt]-flt2flowin[flt],delta,flt))
    
        # compute the actual delay cost based on paxDelays
        delayCost=0
        for nPax,delay,info in paxDelays:
            delayCost+=self.S.config["DELAYCOST"]*nPax*delay
        return delayCost,timeDict,paxDict

deltas=[]
for dataset in ["ACF4","ACF5"]:
    for i in range(1,10):
        S=Scenario(dataset,dataset+"-SC%d"%i,"PAX")
        t1=time.time()
        solver=VNSSolver(S,0)
        solver.generateVNSRecoveryPlan(*solver.VNS(100))
        t2=time.time()
        deltas.append(t2-t1)

df=pd.read_csv("Results/Stats.csv",na_filter=None)
df["VNS_time"]=deltas
df.to_csv("Results/Stats.csv",index=None)


