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
        self.seed=seed
        self.flts=list(S.flight2dict.keys())
        self.tails=list(S.tail2flights.keys())
        self.tail2cap=S.tail2capacity
        self.node=S.name2FNode
        self.tail2srcdest={tail:[self.node[flts[0]].Ori,self.node[flts[-1]].Des] for tail,flts in S.tail2flights.items()}
        self.crew2srcdest={crew:[self.node[flts[0]].Ori,self.node[flts[-1]].Des] for crew,flts in S.crew2flights.items()}        
        self.schedulePs=[[self.tail2srcdest[tail][0]]+flts+[self.tail2srcdest[tail][1]] for tail,flts in S.tail2flights.items()]   # e.g. P=[LAX,F00,F01,F02,ORD]; Ps=[P1,P2,P3,..]
        self.scheduleCrewDict={flt:crew for crew,flts in S.crew2flights.items() for flt in flts}
        
    def checkConnections(self,pairs):
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
        
    def swap(self,P1,P2):
        res=[]
        for u in range(1,len(P1)-1):
            for v in range(1,len(P2)-1):
                if self.checkConnections([(P1[u-1],P2[v]),(P2[v-1],P1[u]),(P2[v],P1[u+1]),(P1[u],P2[v+1])]):
                    pair,loc=(P1[:u]+[P2[v]]+P1[u+1:],P2[:v]+[P1[u]]+P2[v+1:]),(u,v)
                    res.append((pair,loc))
        return res
    
    def cross(self,P1,P2):
        res=[]
        for u1 in range(1,len(P1)-1):
            for u2 in range(u1+1,len(P1)-1):
                for v1 in range(1,len(P2)-1):
                    for v2 in range(v1+1,len(P2)-1):
                        if self.checkConnections([(P1[u1-1],P2[v1]),(P2[v1-1],P1[u1]),(P2[v2],P1[u2+1]),(P1[u2],P2[v2+1])]):
                            pair,loc=(P1[:u1]+P2[v1:v2+1]+P1[u2+1:],P2[:v1]+P1[u1:u2+1]+P2[v2+1:]),(u1,u2,v1,v2)
                            res.append((pair,loc))
        return res
    
    def insert(self,P1,P2):
        res=[]
        for u in range(1,len(P1)-1):
            for v1 in range(1,len(P2)-1):
                for v2 in range(v1+1,len(P2)-1):
                    if self.checkConnections([(P1[u-1],P2[v1]),(P2[v2],P1[u]),(P2[v1-1],P2[v2+1])]):
                        pair,loc=(P1[:u]+P2[v1:v2+1]+P1[u:],P2[:v1]+P2[v2+1:]),(u,v1,v2)
                        res.append((pair,loc))
        return res
            
    def exploreNeighborK(self,k,Ps,crewDict):
        k2func={1:"swap",2:"cross",3:"insert"}
        ind1,ind2=random.sample(range(len(Ps)),2)
        curPs,curLoc=Ps,None
        curObj,curTimeDict,curPaxDict=self.evaluate(Ps)
        for (nP1,nP2),nloc in eval("self."+k2func[k])(Ps[ind1],Ps[ind2]):
            nPs=copy.deepcopy(Ps)
            nPs[ind1],nPs[ind2]=nP1,nP2
            nObj,nTimeDict,nPaxDict=self.evaluate(nPs)
            if nObj<curObj:
                curPs,curObj,curLoc,curTimeDict,curPaxDict=nPs,nObj,nloc,nTimeDict,nPaxDict

        # repair crewDict according to curLoc and k; use the assignment of crews to check whether curPs is a feasible solution    
        # if swap is executed on curPs, then we can just interchange the crews of the two flights, to ensure crews not to miss the connection
        if k==1 and curLoc!=None:
            flt1,flt2=curPs[ind1][curLoc[0]],curPs[ind2][curLoc[1]]
            crewDict[flt1],crewDict[flt2]=crewDict[flt2],crewDict[flt1]
        # TODO: think about how to adjust crewDict when cross or insert happens to curPs.
        elif k==2 and curLoc!=None:
            pass
        elif k==3 and curLoc!=None:
            pass
        return curPs,curObj,crewDict,curTimeDict,curPaxDict
    
    def evaluate(self,Ps):
        flt2tail={flt:self.tails[i] for i,P in enumerate(Ps) for flt in P[1:-1]}
        # step1: estimate the earliest possible dt&at for each flight based on the structure of Ps
        timeDict={flt:[0,0] for flt in self.flts}
        for P in Ps:
            for i in range(len(P)):
                if i==1:
                    timeDict[P[1]]=[self.node[P[1]].SDT,self.node[P[1]].SAT]
                elif i>1 and i<len(P)-1:
                    timeDict[P[i]][0]=max(timeDict[P[i-1]][1]+self.S.config["ACMINCONTIME"],self.node[P[i]].SDT)
                    timeDict[P[i]][1]=timeDict[P[i]][0]+self.node[P[i]].SFT
                                        
        # step2: allocate passengers to minimize the actual delay cost. TODO: improve the allocation of passengers when considering two-leg/three-leg flights
        paxDict={flt:self.S.flt2pax[flt] for flt in self.flts}
        fltDelays=sorted([(flt,timeDict[flt][1]-self.node[flt].ScheduleAT) for flt in self.flts],key=lambda item:item[1],reverse=True) # [(flt1,delay1),(flt2,delay2)]
        paxDelays=[]                # [(nPax1,delay1),(nPax2,delay2)]
        for i,(flt1,delay1) in enumerate(fltDelays):
            if delay1==0:
                break
            for j in range(i+1,len(fltDelays)):
                flt2=fltDelays[j][0]
                if self.node[flt1].Ori==self.node[flt2].Ori and self.node[flt1].Des==self.node[flt2].Des and self.node[flt1].ScheduleDT<=self.node[flt2].SDT and self.node[flt1].SAT>self.node[flt2].SAT:
                    remain2=self.tail2cap[flt2tail[flt2]]-paxDict[flt2]
                    leave=min(remain2,paxDict[flt1])
                    paxDict[flt1]-=leave
                    paxDict[flt2]+=leave
                    paxDelays.append((leave,self.node[flt2].SAT-self.node[flt1].ScheduleAT))
                    
            paxDelays.append((paxDict[flt1],delay1))

        # step3: compute the actual delay cost based on paxDelays
        delayCost=0
        for nPax,delay in paxDelays:
            delayCost+=self.S.config["DELAYCOST"]*nPax*delay                
        return delayCost,timeDict,paxDict
    
    def VNS(self,numIt=10):
        minPs=self.schedulePs
        minObj,minTimeDict,minPaxDict=self.evaluate(minPs)
        minCrewDict=self.scheduleCrewDict
        for i in range(numIt):
            k=1
            while k<=3:
                curPs,curObj,curCrewDict,curTimeDict,curPaxDict=self.exploreNeighborK(k,minPs,minCrewDict)
                if curObj<minObj:
                    minPs,minObj,minCrewDict,minTimeDict,minPaxDict=curPs,curObj,curCrewDict,curTimeDict,curPaxDict
                else:
                    k+=1
        
        print("The minimum delay cost:",minObj)
        print("The flight assignment of tails:",minPs)
        crew2flts={}
        for k,v in minCrewDict.items():
            crew2flts[v]=crew2flts.get(v,[])+[k]
        print("The flight assignment of crews:",crew2flts)
        return minObj,minPs,crew2flts,minTimeDict,minPaxDict
    
    def generateVNSRecoveryPlan(self,minObj,Ps,crew2flts,timeDict,paxDict):
        resd=defaultdict(list)        
        flt2crew={flt:crew for crew,flts in crew2flts.items() for flt in flts}
        for i,P in enumerate(Ps):
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
        
        dfrecovery=pd.DataFrame(resd)
        dfrecovery.to_csv("Results/"+self.S.scname+"/RecoveryVNS.csv",index=None)
        
        resdcost=defaultdict(list)
        resdcost["Cost"].append("delay_cost")
        resdcost["Value"].append(round(minObj))
        dfcost=pd.DataFrame(resdcost)
        dfcost.to_csv("Results/"+self.S.scname+"/CostVNS.csv",index=None)
       
S=Scenario("ACF2","ACF2-F00d3h","PAX")
solver=VNSSolver(S,0)
solver.generateVNSRecoveryPlan(*solver.VNS(10))
        
#S=Scenario("ACF3","ACF3-F05d5h","PAX")
#solver=VNSSolver(S,0)
#solver.generateVNSRecoveryPlan(*solver.VNS(10))

