import random
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import haversine
import json
import os
import cplex

from NetworkVisualizer import Visualizer
from util import ArcFeasibilityChecker,DisruptionHelper,Node
 

class Scenario:
    def __init__(self,direname,scname):
        self.direname=direname
        self.scname=scname
        with open("Datasets/"+direname+"/Config.json", "r") as outfile:
            self.config=json.load(outfile)
        with open("Scenarios/"+scname+"/DisruptionScenario.json", "r") as outfile:
            self.disruption=json.load(outfile)
        
        self.dfitinerary=pd.read_csv("Datasets/"+direname+"/Itinerary.csv",na_filter=None)
        self.dfduration=pd.read_csv("Datasets/"+direname+"/Duration.csv",na_filter=None)
        self.dfschedule=pd.read_csv("Datasets/"+direname+"/Schedule.csv",na_filter=None)
        self.flight2dict={dic['Flight']:dic for dic in self.dfschedule.to_dict(orient='record')}
        self.type2mincontime={"ACF":self.config["ACMINCONTIME"],"CRW":self.config["CREWMINCONTIME"],"PAX":self.config["PAXMINCONTIME"]}

        drpHelper=DisruptionHelper(self,self.dfschedule)
        self.dfschedule,self.FNodes,self.entity2delayTime=drpHelper.processDisruption(self.disruption)
        
        self.name2FNode={node.name:node for node in self.FNodes}
        self.tail2flights={tail:df_cur.sort_values(by='SDT')["Flight"].tolist() for tail,df_cur in self.dfschedule.groupby("Tail")}
        self.crew2flights={crew:df_cur.sort_values(by='SDT')["Flight"].tolist() for crew,df_cur in self.dfschedule.groupby("Crew")}
        self.itin2flights={row.Itinerary:row.Flight_legs.split('-') for row in self.dfitinerary.itertuples()}
        self.type2flightdict={"ACF":self.tail2flights,"CRW":self.crew2flights,"PAX":self.itin2flights}
        self.itin2pax={row.Itinerary:row.Pax for row in self.dfitinerary.itertuples()}
        self.appair2duration={(row.From,row.To):row.Duration for row in self.dfduration.itertuples()}
        self.checker=ArcFeasibilityChecker(self)
        
        self.problem=cplex.Cplex()
        self.problem.objective.set_sense(self.problem.objective.sense.minimize)
        
        
class Entity:
    def __init__(self,S,name,etype):
        self.S=S
        self.name=name
        self.etype=etype
        self.flightPath=[]
        self.F=S.type2flightdict[etype][name]
        self.CT=S.type2mincontime[etype]
        self.Ori=S.flight2dict[self.F[0]]["From"]
        self.Des=S.flight2dict[self.F[-1]]["To"]
        self.EDT=S.config["STARTTIME"]+self.S.entity2delayTime.get(name,0)
        self.LAT=S.config["ENDTIME"]
        self.SNode=Node(S,"SNode","S-"+name,(None,self.Ori,self.EDT,None,0,-1))
        self.TNode=Node(S,"TNode","T-"+name,(self.Des,None,None,self.LAT,0,1))
        self.name2Node=self.S.name2FNode
        self.name2Node.update({"S-"+name:self.SNode,"T-"+name:self.TNode})
        self.string="Type: %s | ID: %s | OD: %s --> %s"%(etype,name,self.Ori,self.Des)
        self.PNGA()
        
    def PNGA(self):
        self.NodeSet,self.ArcSet=set(),set()
        tempPath=[self.SNode]
        self.generatePath(tempPath)
        
    def generatePath(self,tempPath):
        lastNode=tempPath[-1]
        if lastNode.Des==self.Des:
            return
        else:
            nextNodes=[node for node in self.S.FNodes if self.S.checker.checkConnectionArc(lastNode,node)]
            for nextNode in nextNodes:
                tempPathc=tempPath.copy()+[nextNode]
                if nextNode in self.NodeSet:
                    self.insert(tempPathc)
                else:
                    if nextNode.Des==self.Des:
                        self.insert(tempPathc+[self.TNode])
                    self.generatePath(tempPathc)
                    
    def insert(self,tempPath):
        namePath=[node.name for node in tempPath]
        self.NodeSet=self.NodeSet|set(namePath)
        for i in range(len(tempPath)-1):
            self.ArcSet=self.ArcSet|{(tempPath[i].name,tempPath[i+1].name)}
        self.flightPath+=[namePath]
        
    def plotPathOnBaseMap(self,path,ax,title):
        vis=Visualizer(self.S.direname)
        visflights=[(S.flight2dict[flight]["From"],S.flight2dict[flight]["To"]) for flight in path[1:-1]]
        vis.plotTrajectoryOnBasemap(visflights,ax)
        ax.set_title(title)
        
class ConstraintHelper:
    def __init__(self,S):
        self.S=S
        
    def setFlowBalanceConstraint(self,entity):
        head="x^%s_"%entity.name
        flowvar=[head+"(%s,%s)"%(arc[0],arc[1]) for arc in entity.ArcSet]
        self.S.problem.variables.add(names=flowvar,types=['B']*len(flowvar))
        LHS,RHS=[],[]
        for node1 in entity.NodeSet:
            postive,negative=[],[]
            for node2 in entity.NodeSet:
                if (node2,node1) in entity.ArcSet:
                    postive.append(head+"(%s,%s)"%(node2,node1))
                elif (node1,node2) in entity.ArcSet:
                    negative.append(head+"(%s,%s)"%(node1,node2))
                    
            LHS.append([postive+negative,[1]*len(postive)+[-1]*len(negative)])
            RHS.append(entity.name2Node[node1].Demand)
        self.S.problem.linear_constraints.add(lin_expr=LHS,rhs=RHS,senses=['E']*len(RHS),names=["FLOWBALANCE%d"%i for i in range(len(RHS))])
        print(self.S.problem.linear_constraints.get_names())
    
        
    
random.seed(0)
S=Scenario("ACF2","ACF2-SC0")
E=Entity(S,"T00","ACF")

C=ConstraintHelper(S)
C.setFlowBalanceConstraint(E)


## Visualize path:
#fig,axes=plt.subplots(3,3,figsize=(15,10),dpi=100)
#selpaths=random.sample(entity.flightPath,9)
#for i,ax in enumerate(axes.flat):
#    entity.plotPathOnBaseMap(selpaths[i],ax,"Path%d"%i)
#plt.suptitle(entity.string,fontsize=20)


    




