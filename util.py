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


class CplexHelper:
    def __init__(self,sense):
        self.problem=cplex.Cplex()
        if sense=='MAX':
            self.problem.objective.set_sense(self.problem.objective.sense.maximize)
        elif sense=='MIN':
            self.problem.objective.set_sense(self.problem.objective.sense.minimize)
    
    def addVariables(self,names,lowbounds,uppbounds,coeffs=None):
        self.problem.variables.add(names=names,lb=lowbounds,ub=uppbounds,obj=coeffs)
    
    def addLinearConstraint(self,name,LHS,RHS,sense): #LHS:[['x','y'],[c1,c2]] RHS:float
        self.problem.linear_constraints.add(lin_expr=[LHS],rhs=[RHS],senses=[sense],names=[name])
    
    def getVariablesAndConstraints(self):
        return [self.problem.variables.get_names(),self.problem.linear_constraints.get_names()]

    def solveProblem(self):
        self.problem.solve()
        return self.problem.solution.get_values()

        
class DisruptionHelper:
    def __init__(self,disruption):
        self.disruption=disruption
    
    def processDisruptionType12(self,dfschedule):
        for flightName,delayTime in self.disruption["FlightDepartureDelay"]:
            dfschedule.loc[dfschedule['Flight']==flightName,["SDT"]]+=delayTime
        dfschedule.drop(dfschedule[dfschedule.Flight.isin(self.disruption["FlightCancellation"])].index,inplace=True)
        return dfschedule
        
    def processDisruptionType3(self):
        entityName2delayTime={name:time for (name,time) in self.disruption["DelayedReadyTime"]}
        return entityName2delayTime
        
    def processDisruptionType4(self):
        #TODO: implement airport closure
    

#cphp=CplexHelper("MAX")
#cphp.addVariables(['x','y','z'],[0.0,0.0,0.0],[100,1000,cplex.infinity],[5.0,2.0,-1.0])
#cphp.addLinearConstraint("c1",[["x","y","z"],[3.0,1.0,-1.0]],75.0,"L")
#cphp.addLinearConstraint("c2",[["x","y","z"],[3.0,4.0,4.0]],160.0,"L")
#res1=cphp.getVariablesAndConstraints()
#res2=cphp.solveProblem()
#print(res1)
    
    




