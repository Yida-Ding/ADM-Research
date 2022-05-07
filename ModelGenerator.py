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
from NetworkGenerator import Scenario,Entity

class MIPModel:
    def __init__(self,S,type2entity):
        self.S=S
        self.type2entity=type2entity
        self.entities=list(itertools.chain.from_iterable(type2entity.values()))
        self.problem=cplex.Cplex()
        self.problem.objective.set_sense(self.problem.objective.sense.minimize)
        self.constraintData,self.objectiveData,self.itinVariables=[],[],[]
        
        for node in self.S.FNodes:
            #z_f, dt_f, at_f, deltat_f
            self.problem.variables.add(names=["z_%s"%node.name,"dt_%s"%node.name,"at_%s"%node.name,"deltat_%s"%node.name],types=['B','C','C','C'],lb=[0,node.SDT,node.EAT,0],ub=[1,node.LDT,node.LAT,node.CrsTimeComp])            
            #y_f_r, v_f_r, crt_f_r, fc_f_r, tau1_f_r, tau3_f_r, tau4_f_r, w_f_r
            for entity in self.type2entity["ACF"]:
                suffix="_%s_%s"%(node.name,entity.name)
                self.problem.variables.add(names=["y"+suffix,"v"+suffix,"crt"+suffix,"fc"+suffix,"tau1"+suffix,"tau3"+suffix,"tau4"+suffix,'w'+suffix],types=['B']+['C']*7,lb=[0]*8)            
        
        if "ITIN" in self.type2entity:
            #x_r_f_g
            variables=["x_%s_%s_%s"%(entity.name,arc[0].name,arc[1].name) for entity in self.entities for arc in entity.partialGraph.edges]
            self.problem.variables.add(names=variables,types=['I']*len(variables),lb=[0]*len(variables),ub=[entity.Nb for entity in self.entities for arc in entity.partialGraph.edges])

        elif "PAX" in self.type2entity:
            #x_r_f_g
            variables=["x_%s_%s_%s"%(entity.name,arc[0].name,arc[1].name) for entity in self.entities for arc in entity.partialGraph.edges]
            self.problem.variables.add(names=variables,types=['B']*len(variables))
        
    def passConstraintsToCplex(self):
        if "ITIN" in self.type2entity:
            self.problem.variables.add(names=self.itinVariables,types=['B']*len(self.itinVariables))
            
        lin_exp,rhs,senses,names=zip(*self.constraintData)
        lin_exp,rhs,senses,names=list(lin_exp),list(rhs),list(senses),list(names)
        self.problem.linear_constraints.add(lin_expr=lin_exp,rhs=rhs,senses=senses,names=names)   
        self.problem.objective.set_linear(self.objectiveData)
        self.problem.objective.set_offset(-sum([node.ScheFuelConsump for node in self.S.FNodes])*self.S.config["FUELCOSTPERKG"])        
        self.problem.set_problem_name("MIP")
        self.problem.set_log_stream(None)
        self.problem.set_warning_stream(None)
    
    def setFlowBalanceConstraint(self):
        print("Initiate Flow Balance Constraint")
        for entity in self.entities:
            head="x_%s_"%entity.name
            for node1 in entity.partialGraph.nodes:
                positive,negative=[head+"%s_%s"%(node2.name,node1.name) for node2 in entity.partialGraph.predecessors(node1)],[head+"%s_%s"%(node1.name,node2.name) for node2 in entity.partialGraph.successors(node1)]
                self.constraintData.append(([positive+negative,[1]*len(positive)+[-1]*len(negative)],node1.Demand,'E',"FLOWBALANCE_%s_%s"%(entity.name,node1.name)))
        
    def setNodeClosureConstraint(self):
        print("Initiate Node Closure Constraint")
        for node1 in self.S.FNodes:
            for typ in self.type2entity:
                if typ=="ACF" or typ=="CRW":
                    variables=["z_%s"%node1.name]
                    for entity in self.type2entity[typ]:
                        if node1 in entity.partialGraph.nodes:
                            variables+=["x_%s_%s_%s"%(entity.name,node1.name,node2.name) for node2 in entity.partialGraph.successors(node1)]
                    self.constraintData.append(([variables,[1]*(len(variables))],1,'E',"NODECLOSURE_%s_%s"%(typ,node1.name)))
                elif typ=="PAX" or typ=="ITIN":
                    for entity in self.type2entity[typ]:
                        if node1 in entity.partialGraph.nodes:
                            variables=["z_%s"%node1.name]+["x_%s_%s_%s"%(entity.name,node1.name,node2.name) for node2 in entity.partialGraph.successors(node1)]
                            self.constraintData.append(([variables,[entity.Nb]+[1]*(len(variables)-1)],entity.Nb,'L',"NODECLOSURE_%s_%s"%(typ,node1.name)))
        
    def setFlightTimeConstraint(self):
        print("Initiate Flight Time Constraint")
        for node1 in self.S.FNodes:
            variables=["at_%s"%node1.name,"dt_%s"%node1.name,"deltat_%s"%node1.name]
            for entity in self.type2entity["ACF"]:
                if node1 in entity.partialGraph.nodes:
                    variables+=["x_%s_%s_%s"%(entity.name,node1.name,node2.name) for node2 in entity.partialGraph.successors(node1)]
            self.constraintData.append(([variables,[-1,1,-1]+[node1.SFT]*(len(variables)-3)],0,'E',"FLIGHTTIME_%s"%(node1.name)))
        
    def setSourceArcConstraint(self):
        print("Initiate Source Arc Constraint")
        for entity in self.entities:
            for node in entity.partialGraph.successors(entity.SNode):
                if node.SDT<entity.EDT:
                    if entity.etype=="ITIN":
                        self.itinVariables.append("ff_%s_%s"%(entity.name,node.name))
                        variables=["ff_%s_%s"%(entity.name,node.name),"x_%s_%s_%s"%(entity.name,entity.SNode.name,node.name)]
                        self.constraintData.append(([variables,[entity.Nb,-1]],0,'G',"SOURCEARCITIN_%s_%s"%(entity.SNode.name,node.name)))
                        variables=["dt_%s"%node.name,"ff_%s_%s"%(entity.name,node.name)]
                        self.constraintData.append(([variables,[1,-1*entity.EDT]],0,'G',"SOURCEARC_%s_%s"%(entity.SNode.name,node.name)))
                    else:
                        variables=["dt_%s"%node.name,"x_%s_%s_%s"%(entity.name,entity.SNode.name,node.name)]
                        self.constraintData.append(([variables,[1,-1*entity.EDT]],0,'G',"SOURCEARC_%s_%s"%(entity.SNode.name,node.name)))
                        
    def setSinkArcConstraint(self):
        print("Initiate Sink Arc Constraint")
        for entity in self.entities:
            for node in entity.partialGraph.predecessors(entity.TNode):
                timedelta=node.LAT-entity.LAT
                if timedelta>0:
                    if entity.etype=="ITIN":
                        self.itinVariables.append("lf_%s_%s"%(entity.name,node.name))
                        variables=["lf_%s_%s"%(entity.name,node.name),"x_%s_%s_%s"%(entity.name,node.name,entity.TNode.name)]
                        self.constraintData.append(([variables,[entity.Nb,-1]],0,'G',"SINKARCITIN_%s_%s"%(node.name,entity.TNode.name)))
                        variables=["at_%s"%node.name,"lf_%s_%s"%(entity.name,node.name)]
                        self.constraintData.append(([variables,[1,timedelta]],node.LAT,'L',"SINKARC_%s_%s"%(node.name,entity.TNode.name)))
                    else:
                        variables=["at_%s"%node.name,"x_%s_%s_%s"%(entity.name,node.name,entity.TNode.name)]
                        self.constraintData.append(([variables,[1,timedelta]],node.LAT,'L',"SINKARC_%s_%s"%(node.name,entity.TNode.name)))
                        
    def setIntermediateArcConstraint(self):
        print("Initiate Intermediate Arc Constraint")
        for entity in self.entities:
            for node1,node2 in entity.partialGraph.edges:
                if node1!=entity.SNode and node2!=entity.TNode and node1.LAT+entity.CT>node2.SDT:
                    if entity.etype=="ITIN":
                        self.itinVariables.append("con_%s_%s_%s"%(entity.name,node1.name,node2.name))
                        variables=["con_%s_%s_%s"%(entity.name,node1.name,node2.name),"x_%s_%s_%s"%(entity.name,node1.name,node2.name)]
                        self.constraintData.append(([variables,[entity.Nb,-1]],0,'G',"INTERMEDIATEARCITIN_%s_%s"%(node1.name,node2.name)))
                        variables=["at_%s"%node1.name,"con_%s_%s_%s"%(entity.name,node1.name,node2.name),"dt_%s"%node2.name]
                        self.constraintData.append(([variables,[1,(entity.CT+node1.LAT),-1]],node1.LAT,'L',"INTERMEDIATEARC_%s_%s"%(node1.name,node2.name)))
                    else:
                        variables=["at_%s"%node1.name,"x_%s_%s_%s"%(entity.name,node1.name,node2.name),"dt_%s"%node2.name]
                        self.constraintData.append(([variables,[1,(entity.CT+node1.LAT),-1]],node1.LAT,'L',"INTERMEDIATEARC_%s_%s"%(node1.name,node2.name)))
    
    def setSeatCapacityConstraint(self):
        print("Initiate Seat Capacity Constraint")
        for node1 in self.S.FNodes:
            variables,coeffs=[],[]
            for entity in self.type2entity.get("PAX",[])+self.type2entity.get("ITIN",[]):
                if node1 in entity.partialGraph.nodes:
                    temp=["x_%s_%s_%s"%(entity.name,node1.name,node2.name) for node2 in entity.partialGraph.successors(node1)]
                    variables+=temp
                    coeffs+=[1]*len(temp)
            for entity in self.type2entity["ACF"]:
                if node1 in entity.partialGraph.nodes:
                    temp=["x_%s_%s_%s"%(entity.name,node1.name,node2.name) for node2 in entity.partialGraph.successors(node1)]
                    variables+=temp
                    coeffs+=[-1*self.S.tail2capacity[entity.name]]*len(temp) 
            self.constraintData.append(([variables,coeffs],0,'L',"SEATCAPACITY_%s"%(node1.name)))
            
    def setCruiseSpeedConstraint(self):
        print("Initiate Cruise Speed Constraint")
        for node1 in self.S.FNodes:
            variables=["deltat_%s"%node1.name]
            for entity in self.type2entity["ACF"]:
                if node1 in entity.partialGraph.nodes:
                    variables+=["x_%s_%s_%s"%(entity.name,node1.name,node2.name) for node2 in entity.partialGraph.successors(node1)]
            self.constraintData.append(([variables,[-1]+[node1.CrsTimeComp]*(len(variables)-1)],0,'G',"CRUISESPEED_%s"%(node1.name)))
            
    def setCrsTimeCompConstraint(self,crstimecomp):
        print("Initiate Cruise Time Compression Constraint")
        for node1 in self.S.FNodes:
            variables2=["deltat_%s"%node1.name]
            for entity in self.type2entity["ACF"]:
                suffix="_%s_%s"%(node1.name,entity.name)
                variables1=["y"+suffix]
                variables2+=["crt"+suffix]
                if node1 in entity.partialGraph.nodes:
                    variables1+=["x_%s_%s_%s"%(entity.name,node1.name,node2.name) for node2 in entity.partialGraph.successors(node1)]
                
                self.constraintData.append(([variables1,[-1]+[1]*(len(variables1)-1)],0,'E',"CRSTIMECOMP1_%s_%s"%(node1.name,entity.name))) #Eqn16
                self.constraintData.append(([["y"+suffix,"v"+suffix],[node1.ScheCrsSpeed,-1]],0,'L',"CRSTIMECOMP2_%s_%s"%(node1.name,entity.name))) #Eqn17
                self.constraintData.append(([["y"+suffix,"v"+suffix],[node1.MaxCrsSpeed,-1]],0,'G',"CRSTIMECOMP3_%s_%s"%(node1.name,entity.name))) #Eqn17
                self.constraintData.append(([["fc"+suffix,"tau1"+suffix,"v"+suffix,"tau3"+suffix,"tau4"+suffix],[-1/node1.CrsStageDistance]+self.S.config["FUELCONSUMPPARA"]],0,'L',"CRSTIMECOMP4_%s_%s"%(node1.name,entity.name))) #Eqn34                                
                q1=cplex.SparseTriple(ind1=["v"+suffix,"tau1"+suffix],ind2=["v"+suffix,"y"+suffix],val=[1,-1]) #Eqn35
                q2=cplex.SparseTriple(ind1=["y"+suffix,"w"+suffix],ind2=["y"+suffix,"v"+suffix],val=[1,-1])  #Eqn36
                q3=cplex.SparseTriple(ind1=["w"+suffix,"tau3"+suffix],ind2=["w"+suffix,"y"+suffix],val=[1,-1])  #Eqn37
                q4=cplex.SparseTriple(ind1=["w"+suffix,"tau4"+suffix],ind2=["w"+suffix,"v"+suffix],val=[1,-1])  #Eqn38
                q5=cplex.SparseTriple(ind1=["y"+suffix,"v"+suffix],ind2=["y"+suffix,"crt"+suffix],val=[node1.CrsStageDistance,-1]) #Eqn39                
                qs=[q1,q2,q3,q4,q5]
                for i in range(len(qs)):
                    self.problem.quadratic_constraints.add(quad_expr=qs[i],rhs=0,sense='L')
                    
            self.constraintData.append(([variables2,[1]*len(variables2)],node1.ScheCrsTime,'E',"CRSTIMECOMP5_%s"%node1.name)) #Eqn20
            if not crstimecomp:
                self.constraintData.append(([["deltat_%s"%node1.name],[1]],0,'E',"CRSTIMECOMP6_%s"%node1.name))
                
    def addFlightCancellationCost(self):
        print("Initiate Flight Cancellation Cost")
        self.objectiveData+=[("z_%s"%node1.name,self.S.config["FLIGHTCANCELCOST"]) for node1 in self.S.FNodes]
    
    def addFuelCost(self):
        print("Initiate Fuel Cost")
        for node1 in self.S.FNodes:
            for entity in self.type2entity["ACF"]:
                self.objectiveData+=[("fc_%s_%s"%(node1.name,entity.name),self.S.config["FUELCOSTPERKG"])]
    
    def addDelayCost(self,delaytype):
        print("Initiate Delay Cost")
        if delaytype=="approx":
            #delay_f (approximate passenger delay)
            variables=["delay_%s"%node1.name for node1 in self.S.FNodes]
            self.problem.variables.add(names=variables,lb=[0]*(len(variables)),types=['C']*(len(variables)))                        
            for node1 in self.S.FNodes:
                self.constraintData.append(([["at_%s"%node1.name,"delay_%s"%node1.name],[1,-1]],node1.ScheduleAT,'L',"ApproxDelay_%s"%node1.name))
                arrivalPax=sum([self.S.itin2pax[itin] for itin,dest in self.S.itin2destination.items() if dest==node1.Des])
                self.objectiveData+=[("delay_%s"%node1.name,arrivalPax*self.S.config["DELAYCOST"])]

        elif delaytype=="actual":
            #delay_r (indiviual passenger delay)
            variables=["delay_%s"%entity.name for entity in self.type2entity["PAX"]]
            self.problem.variables.add(names=variables,lb=[0]*(len(variables)),types=['C']*(len(variables)))                        
            for entity in self.type2entity["PAX"]:
                for node1 in entity.partialGraph.predecessors(entity.TNode):
                    LATf=node1.LAT
                    self.constraintData.append(([["at_%s"%node1.name,"delay_%s"%entity.name,"x_%s_%s_%s"%(entity.name,node1.name,entity.TNode.name)],[1,-1,LATf-entity.scheduleAT]],LATf,'L',"ActualDelay_%s"%node1.name))
                self.objectiveData+=[("delay_%s"%entity.name,self.S.config["DELAYCOST"])]
              
    def addFollowScheduleCost(self):
        print("Initiate Follow Schedule Cost")
        for entity in self.type2entity["ACF"]:
            flights=[entity.SNode.name]+self.S.tail2flights[entity.name]+[entity.TNode.name]
            for i in range(len(flights)-1):
                if (entity.name2Node[flights[i]],entity.name2Node[flights[i+1]]) in entity.partialGraph.edges:
                    self.objectiveData+=[("x_%s_%s_%s"%(entity.name,flights[i],flights[i+1]),self.S.config["FOLLOWSCHEDULECOST"])]
        
        for entity in self.type2entity["CRW"]:
            flights=[entity.SNode.name]+self.S.crew2flights[entity.name]+[entity.TNode.name]
            for i in range(len(flights)-1):
                if (entity.name2Node[flights[i]],entity.name2Node[flights[i+1]]) in entity.partialGraph.edges:
                    self.objectiveData+=[("x_%s_%s_%s"%(entity.name,flights[i],flights[i+1]),self.S.config["FOLLOWSCHEDULECOST"])]
            
        for entity in self.type2entity.get("PAX",[]):
            flights=[entity.SNode.name]+self.S.paxname2flights[entity.name]+[entity.TNode.name]
            for i in range(len(flights)-1):
                if (entity.name2Node[flights[i]],entity.name2Node[flights[i+1]]) in entity.partialGraph.edges:
                    self.objectiveData+=[("x_%s_%s_%s"%(entity.name,flights[i],flights[i+1]),self.S.config["FOLLOWSCHEDULECOSTPAX"])]

        for entity in self.type2entity.get("ITIN",[]):
            flights=[entity.SNode.name]+self.S.itin2flights[entity.name]+[entity.TNode.name]
            for i in range(len(flights)-1):
                if (entity.name2Node[flights[i]],entity.name2Node[flights[i+1]]) in entity.partialGraph.edges:
                    self.objectiveData+=[("x_%s_%s_%s"%(entity.name,flights[i],flights[i+1]),entity.Nb*self.S.config["FOLLOWSCHEDULECOSTPAX"])]







