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
        self.pax2flights={row.Itinerary+"P%02d"%i:row.Flight_legs.split('-') for row in self.dfitinerary.itertuples() for i in range(row.Pax)}
        
        self.type2flightdict={"ACF":self.tail2flights,"CRW":self.crew2flights,"PAX":self.pax2flights}
        self.itin2pax={row.Itinerary:row.Pax for row in self.dfitinerary.itertuples()}
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
        
    def PNGA(self):
        self.NodeSet,self.ArcSet=set(),set()
        tempPath=[self.SNode]
        self.generatePath(tempPath)
        self.graph=nx.DiGraph(self.ArcSet)
        
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
        flights=[(S.flight2dict[flight]["From"],S.flight2dict[flight]["To"],flight) for flight in path[1:-1]]
        vis.plotTrajectoryOnBasemap(flights,ax)
        ax.set_title(title)
        
class ConstraintHelper:
    def __init__(self,S,type2entity):
        self.S=S
        self.type2entity=type2entity
        self.entities=list(itertools.chain.from_iterable(type2entity.values()))
        
        for entity in self.entities:
            head="x^%s_"%entity.name
            variables=[head+"(%s,%s)"%(arc[0],arc[1]) for arc in entity.ArcSet]
            self.S.problem.variables.add(names=variables,types=['B']*len(variables))

        for f,node in self.S.name2FNode.items():
            self.S.problem.variables.add(names=["z_%s"%f],types=['B'])
            self.S.problem.variables.add(names=["dt_%s"%f,"at_%s"%f],lb=[node.SDT,node.EAT],ub=[node.LDT,node.LAT])
            self.S.problem.variables.add(names=["deltat_%s"%f],lb=[0],ub=[node.CrsTimeComp])
            for entity in self.type2entity["ACF"]:
                self.S.problem.variables.add(names=["y_%s_%s"%(f,entity.name)],types=['B'])
                self.S.problem.variables.add(names=["v_%s_%s"%(f,entity.name),"crt_%s_%s"%(f,entity.name),"fc_%s_%s"%(f,entity.name),"tau1_%s_%s"%(f,entity.name),"tau3_%s_%s"%(f,entity.name),"tau4_%s_%s"%(f,entity.name),"w_%s_%s"%(f,entity.name)],lb=[0,0,0,0,0,0,0])
                
                
    def setFlowBalanceConstraint(self):
        for entity in self.entities:
            head="x^%s_"%entity.name
            for node1 in entity.NodeSet:
                positive,negative=[head+"(%s,%s)"%(node2,node1) for node2 in entity.graph.predecessors(node1)],[head+"(%s,%s)"%(node1,node2) for node2 in entity.graph.successors(node1)]
                self.S.problem.linear_constraints.add(lin_expr=[[positive+negative,[1]*len(positive)+[-1]*len(negative)]],rhs=[entity.name2Node[node1].Demand],senses=['E'],names=["FLOWBALANCE_%s"%(node1)])    
    
    def setNodeClosureConstraint(self):
        for f in self.S.name2FNode:
            for typ in self.type2entity:
                if typ=="ACF" or typ=="CRW":
                    variables=["z_%s"%f]
                    for entity in self.type2entity[typ]:
                        if f in entity.NodeSet:
                            variables+=["x^%s_(%s,%s)"%(entity.name,f,g) for g in entity.graph.successors(f)]
                    self.S.problem.linear_constraints.add(lin_expr=[[variables,[1]*(len(variables))]],rhs=[1],senses=['E'],names=["NODECLOSURE_%s_%s"%(typ,f)])
                elif typ=="PAX":
                    for entity in self.type2entity[typ]:
                        if f in entity.NodeSet:
                            variables=["z_%s"%f]+["x^%s_(%s,%s)"%(entity.name,f,g) for g in entity.graph.successors(f)]
                            self.S.problem.linear_constraints.add(lin_expr=[[variables,[1]*(len(variables))]],rhs=[1],senses=['L'],names=["NODECLOSURE_%s_%s"%(typ,f)])
                
    def setFlightTimeConstraint(self):
        for f,node in self.S.name2FNode.items():
            variables=["at_%s"%f,"dt_%s"%f,"deltat_%s"%f]
            for entity in self.type2entity["ACF"]:
                if f in entity.NodeSet:
                    variables+=["x^%s_(%s,%s)"%(entity.name,f,g) for g in entity.graph.successors(f)]
            self.S.problem.linear_constraints.add(lin_expr=[[variables,[-1,1,-1]+[node.SFT]*(len(variables)-3)]],rhs=[0],senses=['E'],names=["FLIGHTTIME_%s"%(f)])
    
    def setSourceArcConstraint(self):
        for entity in self.entities:
            if entity.SNode.name in entity.graph.nodes:
                for g in entity.graph.successors(entity.SNode.name):
                    if entity.name2Node[g].SDT<entity.EDT:
                        variables=["dt_%s"%g,"x^%s_(%s,%s)"%(entity.name,entity.SNode.name,g)]
                        self.S.problem.linear_constraints.add(lin_expr=[[variables,[1,-1*entity.EDT]]],rhs=[0],senses=['G'],names=["SOURCEARC_(%s,%s)"%(entity.SNode.name,g)])
                        
    def setSinkArcConstraint(self):
        for entity in self.entities:
            if entity.TNode.name in entity.graph.nodes:
                for f in entity.graph.predecessors(entity.TNode.name):
                    timedelta=entity.name2Node[f].LAT-entity.LAT
                    if timedelta>0:
                        variables=["at_%s"%f,"x^%s_(%s,%s)"%(entity.name,f,entity.TNode.name)]
                        self.S.problem.linear_constraints.add(lin_expr=[[variables,[1,timedelta]]],rhs=[entity.name2Node[f].LAT],senses=['L'],names=["SINKARC_(%s,%s)"%(f,entity.TNode.name)])
    
    def setIntermediateArcConstraint(self):
        for entity in self.entities:
            for f,g in entity.ArcSet:
                if f!=entity.SNode.name and g!=entity.TNode.name and entity.name2Node[f].LAT+entity.CT>entity.name2Node[g].SDT:
                    variables=["at_%s"%f,"x^%s_(%s,%s)"%(entity.name,f,g),"dt_%s"%g]
                    self.S.problem.linear_constraints.add(lin_expr=[[variables,[1,(entity.CT+entity.name2Node[f].LAT),-1]]],rhs=[entity.name2Node[f].LAT],senses=['L'],names=["INTERMEDIATEARC_(%s,%s)"%(f,g)])
    
    def setSeatCapacityConstraint(self):
        for f,node in self.S.name2FNode.items():
            variables,coeffs=[],[]
            for entity in self.type2entity["PAX"]:
                if f in entity.NodeSet:
                    variables+=["x^%s_(%s,%s)"%(entity.name,f,g) for g in entity.graph.successors(f)]
                    coeffs+=[1]*len(variables)
            for entity in self.type2entity["ACF"]:
                if f in entity.NodeSet:
                    variables+=["x^%s_(%s,%s)"%(entity.name,f,g) for g in entity.graph.successors(f)]
                    coeffs+=[-1*self.S.tail2capacity[entity.name]]*len(variables) 
            self.S.problem.linear_constraints.add(lin_expr=[[variables,coeffs]],rhs=[0],senses=['L'],names=["SEATCAPACITY_%s"%(f)])
    
    def setCruiseSpeedConstraint(self):
        for f,node in self.S.name2FNode.items():
            variables=["deltat_%s"%f]
            for entity in self.type2entity["ACF"]:
                if f in entity.NodeSet:
                    variables+=["x^%s_(%s,%s)"%(entity.name,f,g) for g in entity.graph.successors(f)]
            self.S.problem.linear_constraints.add(lin_expr=[[variables,[-1]+[entity.name2Node[f].CrsTimeComp]*(len(variables)-1)]],rhs=[0],senses=['G'],names=["CRUISESPEED_%s"%(f)])
    
    #units: v:km/min t:min d:km
    def setSpeedCompressionConstraint(self):
        for f,node in self.S.name2FNode.items():
            variables2=["deltat_%s"%f]
            for entity in self.type2entity["ACF"]:
                suffix="_%s_%s"%(f,entity.name)
                variables1=["y"+suffix]
                variables2+=["crt"+suffix]
                if f in entity.NodeSet:
                    variables1+=["x^%s_(%s,%s)"%(entity.name,f,g) for g in entity.graph.successors(f)]
                self.S.problem.linear_constraints.add(lin_expr=[[variables1,[-1]+[1]*(len(variables1)-1)]],rhs=[0],senses=['E'])    
                self.S.problem.linear_constraints.add(lin_expr=[[["y"+suffix,"v"+suffix],[self.S.config["ACTAVGSPEED"]/60,-1]]],rhs=[0],senses=['L']) 
                self.S.problem.linear_constraints.add(lin_expr=[[["y"+suffix,"v"+suffix],[self.S.config["ACTAVGSPEED"]/60*1.1,-1]]],rhs=[0],senses=['G'])
                self.S.problem.linear_constraints.add(lin_expr=[[["fc"+suffix,"tau1"+suffix,"v"+suffix,"tau3"+suffix,"tau4"+suffix],[-1/self.S.flight2dict[f]["Distance"],0.01,0.16,0.74,2200]]],rhs=[0],senses=['L'])
                
                q1=cplex.SparseTriple(ind1=["v"+suffix,"tau1"+suffix],ind2=["v"+suffix,"y"+suffix],val=[1,-1])
                q2=cplex.SparseTriple(ind1=["y"+suffix,"w"+suffix],ind2=["y"+suffix,"v"+suffix],val=[1,-1])
                q3=cplex.SparseTriple(ind1=["w"+suffix,"tau3"+suffix],ind2=["w"+suffix,"y"+suffix],val=[1,-1])
                q4=cplex.SparseTriple(ind1=["w"+suffix,"tau4"+suffix],ind2=["w"+suffix,"v"+suffix],val=[1,-1])
                q5=cplex.SparseTriple(ind1=["y"+suffix,"v"+suffix],ind2=["y"+suffix,"crt"+suffix],val=[self.S.flight2dict[f]["Distance"],-1])
                
                qs=[q1,q2,q3,q4,q5]
                for i in range(len(qs)):
                    self.S.problem.quadratic_constraints.add(quad_expr=qs[i],rhs=0,sense='L')
                    
            self.S.problem.linear_constraints.add(lin_expr=[[variables2,[1]*len(variables2)]],rhs=[self.S.flight2dict[f]["Cruise_time"]],senses=['E'])
                
    
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
C.setSpeedCompressionConstraint()


## Visualize path:m
#fig,axes=plt.subplots(3,3,figsize=(15,10),dpi=100)
#selpaths=random.sample(E.flightPath,min(len(E.flightPath),9))
#for i,ax in enumerate(axes.flat[:len(selpaths)]):
#    E.plotPathOnBaseMap(selpaths[i],ax,"Path%d"%i)
#plt.suptitle(E.string,fontsize=20)

    




