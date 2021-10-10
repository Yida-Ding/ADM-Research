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

class ArcFeasibilityChecker:
    def __init__(self,S):
        self.S=S
    
    def checkConnectionArc(self,node1,node2):
        if (node1.ntype,node2.ntype) in [("SNode","FNode"),("FNode","TNode")]:
            if node1.Des==node2.Ori and node2.LDT>=node1.EAT:
                return True
        elif (node1.ntype,node2.ntype)==("FNode","FNode"):
            if node1.Des==node2.Ori and node2.LDT>=node1.EAT+node1.CT:
                return True            
        return False
        
    def checkExternalArc(self,node1,node2):
        if not self.checkConnectionArc(node1,node2) and node1.SAT+self.appair2duration[(node1.Des,node2.Ori)]<=node1.LAT:
            return True
        return False


class Node:
    def __init__(self,S,ntype="FNode",name=None,info=None):
        self.ntype=ntype
        self.name=name
        if ntype=="FNode":
            self.Ori=S.flight2dict[name]["From"]
            self.Des=S.flight2dict[name]["To"]
            self.SDT=S.flight2dict[name]["SDT"]
            self.SAT=S.flight2dict[name]["SAT"]
            self.SFT=S.flight2dict[name]["Flight_time"]
            self.LDT=self.SDT+S.config["MAXHOLDTIME"]
            self.LAT=self.SAT+S.config["MAXHOLDTIME"]
            self.CrsTimeComp=int(S.flight2dict[name]["Cruise_time"]*S.config["CRSTIMECOMPPCT"]) #TODO: Consider the compression limit by different types of aircraft in the future, and modify EAT
            self.EDT=self.SDT
            self.EAT=self.SDT+self.SFT-self.CrsTimeComp
            self.CT=min(S.type2mincontime.values())
            self.Demand=0
            
        elif ntype=="SNode" or ntype=="TNode":
            self.Ori,self.Des,self.EAT,self.LDT,self.CT,self.Demand=info
            self.SDT=self.SAT=self.LAT=self.EDT=self.CrsTimeComp=self.SFT=None
        #TODO: generate data for schedule maintenance and construct must-nodes       

        
class DisruptionHelper:
    def __init__(self,S,dfschedule):
        self.S=S
        self.dfschedule=dfschedule

    def processDisruption(self,disruption):
        #FlightDepartureDelay,FlightCancellation,DelayedReadyTime
        for flight,delayTime in disruption.get("FlightDepartureDelay",[]):
            self.dfschedule.loc[self.dfschedule['Flight']==flight,["SDT"]]+=delayTime
        self.dfschedule.drop(self.dfschedule[self.dfschedule.Flight.isin(disruption.get("FlightCancellation",[]))].index,inplace=True)
        entity2delayTime={name:time for (name,time) in disruption.get("DelayedReadyTime",[])}        
        
        #AirportClosure
        FNodes=[]
        orig2FNode,dest2FNode=defaultdict(list),defaultdict(list)
        for flight in self.dfschedule["Flight"].tolist():
            node=Node(self.S,ntype="FNode",name=flight)
            FNodes.append(node)
            orig2FNode[node.Ori].append(node)
            dest2FNode[node.Des].append(node)
                    
        for ap,startTime,endTime in disruption.get("AirportClosure",[]):
            for node in orig2FNode.get(ap,[]):
                if node.SDT>startTime and node.LDT<endTime:
                    self.dfschedule.drop(self.dfschedule[self.dfschedule.Flight==node.name].index,inplace=True)
                    FNodes.remove(node)
                elif node.SDT<startTime and node.LDT>startTime:
                    node.LDT=startTime
                elif node.SDT<endTime and node.LDT>endTime:
                    node.SDT=endTime
            for node in dest2FNode.get(ap,[]):
                if node.EAT>startTime and node.LAT<endTime:
                    self.dfschedule.drop(self.dfschedule[self.dfschedule.Flight==node.name].index,inplace=True)
                    FNodes.remove(node)
                elif node.EAT<startTime and node.LAT>startTime:
                    node.LAT=startTime
                elif node.EAT<endTime and node.LAT>endTime:
                    node.EAT=endTime
        return self.dfschedule,FNodes,entity2delayTime
    
class CplexHelper:
    def __init__(self,sense):
        self.problem=cplex.Cplex()
        if sense=='MAX':
            self.problem.objective.set_sense(self.problem.objective.sense.maximize)
        elif sense=='MIN':
            self.problem.objective.set_sense(self.problem.objective.sense.minimize)
    
    def addVariables(self,names,lowbounds=None,uppbounds=None,coeffs=None,types=None):
        self.problem.variables.add(names=names,lb=lowbounds,ub=uppbounds,obj=coeffs)
    
    def addLinearConstraint(self,name,LHS,RHS,sense): #LHS:[['x','y'],[c1,c2]] RHS:float
        self.problem.linear_constraints.add(lin_expr=[LHS],rhs=[RHS],senses=[sense],names=[name])
    
    def getVariablesAndConstraints(self):
        return [self.problem.variables.get_names(),self.problem.linear_constraints.get_names()]

    def solveProblem(self):
        self.problem.solve()
        return self.problem.solution.get_values()

#cphp=CplexHelper("MAX")
#cphp.addVariables(['x','y','z'],[0.0,0.0,0.0],[100,1000,cplex.infinity],[5.0,2.0,-1.0])
#cphp.addLinearConstraint("c1",[["x","y","z"],[3.0,1.0,-1.0]],75.0,"L")
#cphp.addLinearConstraint("c2",[["x","y","z"],[3.0,4.0,4.0]],160.0,"L")
#res1=cphp.getVariablesAndConstraints()
#res2=cphp.solveProblem()
#print(res1)
    
    




