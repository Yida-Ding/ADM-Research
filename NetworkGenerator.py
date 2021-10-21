import random
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import haversine
import json
import os
import cplex
import itertools

from NetworkVisualizer import Visualizer
from util import ArcFeasibilityChecker,DisruptionHelper,Node,ConstraintHelper
 

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
        self.pax2flights={row.Itinerary+"P%02d"%i:row.Flight_legs.split('-') for row in self.dfitinerary.itertuples() for i in range(row.Pax)}
        
        self.type2flightdict={"ACF":self.tail2flights,"CRW":self.crew2flights,"PAX":self.pax2flights}
        self.itin2pax={row.Itinerary:row.Pax for row in self.dfitinerary.itertuples()}
        self.itin2destination={itin:self.name2FNode[self.itin2flights[itin][-1]].Des for itin in self.itin2flights.keys()}
        self.appair2duration={(row.From,row.To):row.Duration for row in self.dfduration.itertuples()}
        self.tail2capacity={tail:df_cur["Capacity"].tolist()[0] for tail,df_cur in self.dfschedule.groupby("Tail")}
        self.checker=ArcFeasibilityChecker(self)
        
        self.problem=cplex.Cplex()
        self.problem.objective.set_sense(self.problem.objective.sense.minimize)
        
        
class Entity:
    def __init__(self,S,name,etype):
        self.S=S
        self.name=name
        self.etype=etype
        self.F=S.type2flightdict[etype][name]
        self.CT=S.type2mincontime[etype]
        self.Ori=S.flight2dict[self.F[0]]["From"]
        self.Des=S.flight2dict[self.F[-1]]["To"]
        self.EDT=S.config["STARTTIME"]+self.S.entity2delayTime.get(name,0)
        self.LAT=S.config["ENDTIME"]
        self.SNode=Node(S,"SNode","S-"+name,(None,self.Ori,self.EDT,None,0,-1))
        self.TNode=Node(S,"TNode","T-"+name,(self.Des,None,None,self.LAT,0,1))
        self.name2Node=self.S.name2FNode.copy()
        self.name2Node.update({"S-"+name:self.SNode,"T-"+name:self.TNode})
        self.string="Type: %s | ID: %s | OD: %s --> %s"%(etype,name,self.Ori,self.Des)
        self.flightPath=[]
        self.PNGA()
        self.graph=nx.DiGraph(self.ArcSet)
        
    def PNGA(self):
        self.NodeSet,self.ArcSet=set(),set()
        tempPath=[self.SNode]
        self.generatePath(tempPath,True)        
        
    def generatePath(self,tempPath,initflag):
        lastNode=tempPath[-1]
        if lastNode.Des==self.Des and initflag==False:
            return
        else:
            nextNodes=[node for node in self.S.FNodes if self.S.checker.checkConnectionArc(lastNode,node,tempPath)]
            for nextNode in nextNodes:
                tempPathc=tempPath.copy()+[nextNode]
                if nextNode in self.NodeSet:
                    self.insert(tempPathc)
                else:
                    if nextNode.Des==self.Des:
                        self.insert(tempPathc+[self.TNode])
                    self.generatePath(tempPathc,False)
                    
    def insert(self,tempPath):
        namePath=[node.name for node in tempPath]
        self.NodeSet=self.NodeSet|set(namePath)
        for i in range(len(tempPath)-1):
            self.ArcSet=self.ArcSet|{(tempPath[i].name,tempPath[i+1].name)}
        self.flightPath+=[namePath]
        
    def plotPathOnBaseMap(self,path,ax,title):
        vis=Visualizer(self.S.direname)
        flights=[(S.flight2dict[flight]["From"],S.flight2dict[flight]["To"],flight) for flight in path[1:-1]]
        vis.plotTrajectoryOnBasemap(flights,ax)
        ax.set_title(title)
        
    
    
random.seed(0)
S=Scenario("ACF2","ACF2-SC0")

#create all entities
type2entity=defaultdict(list)
for tname in S.tail2flights:
    type2entity["ACF"].append(Entity(S,tname,"ACF"))
for cname in S.crew2flights:
    type2entity["CRW"].append(Entity(S,cname,"CRW"))
for pname in S.pax2flights:
    type2entity["PAX"].append(Entity(S,pname,"PAX"))

C=ConstraintHelper(S,type2entity)
#C.setSpeedCompressionConstraint()
#C.setCruiseSpeedConstraint()
#C.setFlightTimeConstraint
#C.setFlowBalanceConstraint()
#C.setIntermediateArcConstraint()
C.setNodeClosureConstraint()
C.setSeatCapacityConstraint()
#C.setSinkArcConstraint()
#C.setSourceArcConstraint()
C.addFlightCancellationCost()
C.addFuelCost()
C.addActualDelayCost()
#C.addFollowScheduleCost()

S.problem.solve()
print(S.problem.solution.get_values())
## Visualize path:m
#print(E.name)
#print(E.ArcSet)
#fig,axes=plt.subplots(3,3,figsize=(15,10),dpi=100)
#selpaths=random.sample(E.flightPath,min(len(E.flightPath),9))
#for i,ax in enumerate(axes.flat[:len(selpaths)]):
#    E.plotPathOnBaseMap(selpaths[i],ax,"Path%d"%i)
#    print(selpaths[i])
#plt.suptitle(E.string,fontsize=20)

    




