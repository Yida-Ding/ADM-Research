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
            
    def visTimes(self,timeDict):
        for flt,times in timeDict.items():
            print(flt,self.S.getTimeString(times[0]),self.S.getTimeString(times[1]))
            
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
        
        print("The minimum cost:",minRes[0])
        print("Tail assignment:",minPs)
        print("Crew assignment:",minQs)
        return minPs,minQs,minRes
    
        
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
        
        itinDelay=sorted([(itin,timeDict[self.S.itin2flights[itin][-1]][1]-self.S.itin2skdtime[itin][1]) for itin in self.S.itin2flights.keys()],key=lambda xx:xx[1],reverse=True)
        itin2pax=self.S.itin2pax.copy()
        bothItin2pax=defaultdict(int); itin2flowin=defaultdict(int)
        for itin1,delay1 in itinDelay:
            # Case1: not rerouted
            minCost=self.S.config["DELAYCOST"]*itin2pax[itin1]*delay1
            minflt2=None            
            
            flts1=self.S.itin2flights[itin1]
            for flt2 in self.flts:
                if self.S.itin2origin[itin1]==self.node[flt2].Ori and self.S.itin2destination[itin1]==self.node[flt2].Des and timeDict[flt2][1]<timeDict[flts1[-1]][1]:# arrive earlier than current
                    paxFlt2=sum([itin2pax[itin] for itin in self.S.flt2skditins[flt2]])
                    remain2=self.tail2cap[flt2tail[flt2]]-paxFlt2
                    leave=min(remain2,itin2pax[itin1])
                    remain1=itin2pax[itin1]-leave                    
                    # Case2: reroute with a late flight
                    if timeDict[flt2][0]>self.S.itin2skdtime[itin1][0]: # depart later than schedule, then arrive must later than schedule 
                        timeFlt2=[timeDict[flt2][0],timeDict[flt2][1]]
                        cost=self.S.config["DELAYCOST"]*leave*(timeDict[flt2][1]-self.S.itin2skdtime[itin1][1])+self.S.config["DELAYCOST"]*remain1*delay1                    
                    # Case3: reroute with an early flight
                    else: # depart earlier than schedule, change the time of flight2 to schedule
                        timeFlt2=[self.S.itin2skdtime[itin1][0],self.S.itin2skdtime[itin1][0]+self.node[flt2].SFT]
                        cost=self.S.config["DELAYCOST"]*remain1*delay1+self.S.config["DELAYCOST"]*paxFlt2*(timeFlt2[1]-self.node[flt2].ScheduleAT)                        
                    if cost<minCost:
                        minCost=cost
                        minflt2=flt2
                        minTimeFlt2=timeFlt2
                        minLeave=leave
                            
            if minflt2!=None:
                itin2=self.S.fltlegs2itin[minflt2]
                timeDict[minflt2]=minTimeFlt2
                itin2flowin[itin2]+=minLeave
                itin2pax[itin1]-=minLeave
                itin2pax[itin2]+=minLeave
                bothItin2pax[(itin2,itin1)]=minLeave
            
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

      
for i in range(10):
    S=Scenario("ACF6","ACF6-SC%d"%i,"PAX")
    solver=VNSSolver(S,0)
    solver.generateVNSRecoveryPlan(*solver.VNS(100))


