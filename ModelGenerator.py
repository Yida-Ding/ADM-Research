import random
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import haversine
import json
import os
import cplex
import numpy as np
import itertools

class MIPModel:
    def __init__(self,S,type2entity):
        self.S=S
        self.type2entity=type2entity
        self.entities=list(itertools.chain.from_iterable(type2entity.values()))
        
        for entity in self.entities:
            variables=["x_%s_%s_%s"%(entity.name,arc[0],arc[1]) for arc in entity.ArcSet]
            self.S.problem.variables.add(names=variables,types=['B']*len(variables))

        for f,node in self.S.name2FNode.items():
            self.S.problem.variables.add(names=["z_%s"%f],types=['B'])
            self.S.problem.variables.add(names=["dt_%s"%f,"at_%s"%f],lb=[node.SDT,node.EAT],ub=[node.LDT,node.LAT],types=['C','C'])
            self.S.problem.variables.add(names=["deltat_%s"%f],lb=[0],ub=[node.CrsTimeComp],types=['C'])
            self.S.problem.variables.add(names=["delay_%s"%f],lb=[0],types=['C'])
            for entity in self.type2entity["ACF"]:
                self.S.problem.variables.add(names=["y_%s_%s"%(f,entity.name)],types=['B'])
                self.S.problem.variables.add(names=["v_%s_%s"%(f,entity.name),"crt_%s_%s"%(f,entity.name),"fc_%s_%s"%(f,entity.name),"tau1_%s_%s"%(f,entity.name),"tau3_%s_%s"%(f,entity.name),"tau4_%s_%s"%(f,entity.name),"w_%s_%s"%(f,entity.name)],lb=[0]*7,types=['C']*7)
                
        for entity in self.type2entity["PAX"]:
            self.S.problem.variables.add(names=["delay_%s"%entity.name],lb=[0],types=['C'])
        
    def setFlowBalanceConstraint(self):
        print("Initiate Flow Balance Constraint")
        for entity in self.entities:
            head="x_%s_"%entity.name
            for node1 in entity.NodeSet:
                positive,negative=[head+"%s_%s"%(node2,node1) for node2 in entity.graph.predecessors(node1)],[head+"%s_%s"%(node1,node2) for node2 in entity.graph.successors(node1)]
                self.S.problem.linear_constraints.add(lin_expr=[[positive+negative,[1]*len(positive)+[-1]*len(negative)]],rhs=[entity.name2Node[node1].Demand],senses=['E'],names=["FLOWBALANCE_%s"%(node1)])    
        
    def setNodeClosureConstraint(self):
        print("Initiate Node Closure Constraint ")
        for f in self.S.name2FNode:
            for typ in self.type2entity:
                if typ=="ACF" or typ=="CRW":
                    variables=["z_%s"%f]
                    for entity in self.type2entity[typ]:
                        if f in entity.NodeSet:
                            variables+=["x_%s_%s_%s"%(entity.name,f,g) for g in entity.graph.successors(f)]
                    self.S.problem.linear_constraints.add(lin_expr=[[variables,[1]*(len(variables))]],rhs=[1],senses=['E'],names=["NODECLOSURE_%s_%s"%(typ,f)])
                elif typ=="PAX":
                    for entity in self.type2entity[typ]:
                        if f in entity.NodeSet:
                            variables=["z_%s"%f]+["x_%s_%s_%s"%(entity.name,f,g) for g in entity.graph.successors(f)]
                            self.S.problem.linear_constraints.add(lin_expr=[[variables,[1]*(len(variables))]],rhs=[1],senses=['L'],names=["NODECLOSURE_%s_%s"%(typ,f)])
        
    def setFlightTimeConstraint(self):
        print("Initiate Flight Time Constraint ")
        for f,node in self.S.name2FNode.items():
            variables=["at_%s"%f,"dt_%s"%f,"deltat_%s"%f]
            for entity in self.type2entity["ACF"]:
                if f in entity.NodeSet:
                    variables+=["x_%s_%s_%s"%(entity.name,f,g) for g in entity.graph.successors(f)]
            self.S.problem.linear_constraints.add(lin_expr=[[variables,[-1,1,-1]+[node.SFT]*(len(variables)-3)]],rhs=[0],senses=['E'],names=["FLIGHTTIME_%s"%(f)])
    
        
    def setSourceArcConstraint(self):
        print("Initiate Source Arc Constraint ")
        for entity in self.entities:
            if entity.SNode.name in entity.graph.nodes: # in case of the SNode is not in the graph, e.g. an empty ArcSet 
                for g in entity.graph.successors(entity.SNode.name):
                    if entity.name2Node[g].SDT<entity.EDT:
                        variables=["dt_%s"%g,"x_%s_%s_%s"%(entity.name,entity.SNode.name,g)]
                        self.S.problem.linear_constraints.add(lin_expr=[[variables,[1,-1*entity.EDT]]],rhs=[0],senses=['G'],names=["SOURCEARC_%s_%s"%(entity.SNode.name,g)])
                       
    def setSinkArcConstraint(self):
        print("Initiate Sink Arc Constraint ")
        for entity in self.entities:
            if entity.TNode.name in entity.graph.nodes:
                for f in entity.graph.predecessors(entity.TNode.name):
                    timedelta=entity.name2Node[f].LAT-entity.LAT
                    if timedelta>0:
                        variables=["at_%s"%f,"x_%s_%s_%s"%(entity.name,f,entity.TNode.name)]
                        self.S.problem.linear_constraints.add(lin_expr=[[variables,[1,timedelta]]],rhs=[entity.name2Node[f].LAT],senses=['L'],names=["SINKARC_%s_%s"%(f,entity.TNode.name)])
    
    def setIntermediateArcConstraint(self):
        print("Initiate Intermediate Arc Constraint ")
        for entity in self.entities:
            for f,g in entity.ArcSet:
                if f!=entity.SNode.name and g!=entity.TNode.name and entity.name2Node[f].LAT+entity.CT>entity.name2Node[g].SDT:
                    variables=["at_%s"%f,"x_%s_%s_%s"%(entity.name,f,g),"dt_%s"%g]
                    self.S.problem.linear_constraints.add(lin_expr=[[variables,[1,(entity.CT+entity.name2Node[f].LAT),-1]]],rhs=[entity.name2Node[f].LAT],senses=['L'],names=["INTERMEDIATEARC_%s_%s"%(f,g)])
    
    def setSeatCapacityConstraint(self):
        print("Initiate Seat Capacity Constraint ")
        for f,node in self.S.name2FNode.items():
            variables,coeffs=[],[]
            for entity in self.type2entity["PAX"]:
                if f in entity.NodeSet:
                    temp=["x_%s_%s_%s"%(entity.name,f,g) for g in entity.graph.successors(f)]
                    variables+=temp
                    coeffs+=[1]*len(temp)
            for entity in self.type2entity["ACF"]:
                if f in entity.NodeSet:
                    temp=["x_%s_%s_%s"%(entity.name,f,g) for g in entity.graph.successors(f)]
                    variables+=temp
                    coeffs+=[-1*self.S.tail2capacity[entity.name]]*len(temp) 
            self.S.problem.linear_constraints.add(lin_expr=[[variables,coeffs]],rhs=[0],senses=['L'],names=["SEATCAPACITY_%s"%(f)])
    
    def setCruiseSpeedConstraint(self):
        print("Initiate Cruise Speed Constraint ")
        for f,node in self.S.name2FNode.items():
            variables=["deltat_%s"%f]
            for entity in self.type2entity["ACF"]:
                if f in entity.NodeSet:
                    variables+=["x_%s_%s_%s"%(entity.name,f,g) for g in entity.graph.successors(f)]
            self.S.problem.linear_constraints.add(lin_expr=[[variables,[-1]+[entity.name2Node[f].CrsTimeComp]*(len(variables)-1)]],rhs=[0],senses=['G'],names=["CRUISESPEED_%s"%(f)])
    
    #units: v:km/min t:min d:km
    def setSpeedCompressionConstraint(self):
        print("Initiate Speed Compression Constraint ")
        for f,node in self.S.name2FNode.items():
            variables2=["deltat_%s"%f]
            for entity in self.type2entity["ACF"]:
                suffix="_%s_%s"%(f,entity.name)
                variables1=["y"+suffix]
                variables2+=["crt"+suffix]
                if f in entity.NodeSet:
                    variables1+=["x_%s_%s_%s"%(entity.name,f,g) for g in entity.graph.successors(f)]
                self.S.problem.linear_constraints.add(lin_expr=[[variables1,[-1]+[1]*(len(variables1)-1)]],rhs=[0],senses=['E'])    
                self.S.problem.linear_constraints.add(lin_expr=[[["y"+suffix,"v"+suffix],[self.S.config["SCHEDULECRUISESPEED"],-1]]],rhs=[0],senses=['L']) 
                self.S.problem.linear_constraints.add(lin_expr=[[["y"+suffix,"v"+suffix],[self.S.config["MAXCRUISESPEED"],-1]]],rhs=[0],senses=['G'])
                self.S.problem.linear_constraints.add(lin_expr=[[["fc"+suffix,"tau1"+suffix,"v"+suffix,"tau3"+suffix,"tau4"+suffix],[-1/self.S.flight2dict[f]["Distance"]]+self.S.config["FUELCONSUMPPARA"]]],rhs=[0],senses=['L'])
                
                q1=cplex.SparseTriple(ind1=["v"+suffix,"tau1"+suffix],ind2=["v"+suffix,"y"+suffix],val=[1,-1])
                q2=cplex.SparseTriple(ind1=["y"+suffix,"w"+suffix],ind2=["y"+suffix,"v"+suffix],val=[1,-1])
                q3=cplex.SparseTriple(ind1=["w"+suffix,"tau3"+suffix],ind2=["w"+suffix,"y"+suffix],val=[1,-1])
                q4=cplex.SparseTriple(ind1=["w"+suffix,"tau4"+suffix],ind2=["w"+suffix,"v"+suffix],val=[1,-1])
                q5=cplex.SparseTriple(ind1=["y"+suffix,"v"+suffix],ind2=["y"+suffix,"crt"+suffix],val=[self.S.flight2dict[f]["Distance"],-1])
                
                qs=[q1,q2,q3,q4,q5]
                for i in range(len(qs)):
                    self.S.problem.quadratic_constraints.add(quad_expr=qs[i],rhs=0,sense='L')
                    
            self.S.problem.linear_constraints.add(lin_expr=[[variables2,[1]*len(variables2)]],rhs=[self.S.flight2dict[f]["Distance"]/self.S.config["SCHEDULECRUISESPEED"]],senses=['E'])

    def addFlightCancellationCost(self):
        print("Initiate Flight Cancellation Cost")
        self.S.problem.objective.set_linear([("z_%s"%f,self.S.config["FLIGHTCANCELCOST"]) for f in self.S.name2FNode.keys()])
    
    def addFuelCost(self):
        print("Initiate Fuel Cost")
        for f,node in self.S.name2FNode.items():
            for entity in self.type2entity["ACF"]:
                if f in entity.NodeSet:
                    self.S.problem.objective.set_linear([("fc_%s_%s"%(f,entity.name),self.S.config["FUELCOSTPERKG"])])
    
    #use linear function with flight delay approximation in 3.10.1
    def addApproximatedDelayCost(self):
        print("Initiate Delay Cost")
        for f,node in self.S.name2FNode.items():
            self.S.problem.linear_constraints.add(lin_expr=[[["at_%s"%f,"delay_%s"%f],[1,-1]]],rhs=[node.SAT],senses=['L'])
            arrivalPax=sum([self.S.itin2pax[itin] for itin,dest in self.S.itin2destination.items() if dest==node.Des])
            self.S.problem.objective.set_linear([("delay_%s"%f,arrivalPax*self.S.config["DELAYCOST"])])
            
    def addActualDelayCost(self):
        print("Initiate Delay Cost")
        for entity in self.type2entity["PAX"]:
            SATpax=entity.name2Node[entity.F[-1]].SAT
            for f in entity.graph.predecessors(entity.TNode.name):
                LATf=entity.name2Node[f].LAT
                self.S.problem.linear_constraints.add(lin_expr=[[["at_%s"%f,"delay_%s"%entity.name,"x_%s_%s_%s"%(entity.name,f,entity.TNode.name)],[1,-1,LATf-SATpax]]],rhs=[LATf],senses=['L'])

            self.S.problem.objective.set_linear([("delay_%s"%entity.name,self.S.config["DELAYCOST"])])
            
    def addFollowScheduleCost(self):
        for entity in self.type2entity["ACF"]:
            flights=self.S.tail2flights[entity.name]
            for i in range(len(flights)-1):
                if (flights[i],flights[i+1]) in entity.ArcSet:
                    self.S.problem.objective.set_linear([("x_%s_%s_%s"%(entity.name,flights[i],flights[i+1]),self.S.config["FOLLOWSCHEDULECOST"])])
        
        for entity in self.type2entity["CRW"]:
            flights=self.S.crew2flights[entity.name]
            for i in range(len(flights)-1):
                if (flights[i],flights[i+1]) in entity.ArcSet:
                    self.S.problem.objective.set_linear([("x_%s_%s_%s"%(entity.name,flights[i],flights[i+1]),self.S.config["FOLLOWSCHEDULECOST"])])
            
        for entity in self.type2entity["PAX"]:
            flights=self.S.pax2flights[entity.name]
            for i in range(len(flights)-1):
                if (flights[i],flights[i+1]) in entity.ArcSet:
                    self.S.problem.objective.set_linear([("x_%s_%s_%s"%(entity.name,flights[i],flights[i+1]),self.S.config["FOLLOWSCHEDULECOSTPAX"])])

