import random
import networkx as nx
from collections import defaultdict
import pandas as pd
import haversine
import json
import os
import numpy as np
import time
from ModelGenerator import MIPModel
from NetworkGenerator import Scenario,Entity,PSCAHelper


def executeModel(dataset,scenario,mode):
    if not os.path.exists("Results/%s"%scenario):
        os.makedirs("Results/%s"%scenario)
        
    T1=time.time()
    S=Scenario(dataset,scenario,mode["PAXTYPE"])
    type2entity={}
    for etype in ["ACF","CRW",mode["PAXTYPE"]]:
        pscahelper=PSCAHelper(S,etype,mode["BOUNDETYPES"][etype],mode["SIZEBOUND"])
        type2entity[etype]=pscahelper.entities
        pscahelper.getGraphReductionStat(mode["MODEID"])

    T2=time.time()
    model=MIPModel(S,type2entity)
    model.setFlowBalanceConstraint()
    model.setNodeClosureConstraint()
    model.setFlightTimeConstraint()
    model.setSourceArcConstraint()
    model.setSinkArcConstraint()
    model.setIntermediateArcConstraint()
    model.setSeatCapacityConstraint()
    model.setCruiseSpeedConstraint()
    model.setCrsTimeCompConstraint(mode["CRSTIMECOMP"])
    model.addFlightCancellationCost()
    model.addFuelCost()
    model.addFollowScheduleCost()
    model.addDelayCost(mode["DELAYTYPE"])
    model.passConstraintsToCplex()
    
    model.problem.parameters.timelimit.set(mode["TIMELIMIT"])
    model.problem.parameters.mip.tolerances.mipgap.set(mode["MIPTOLERANCE"])
    print(model.problem.get_stats())
    model.problem.solve()
    T3=time.time()
    
    resdire="Results/%s/"%(scenario)
    model.problem.solution.write(resdire+"ModelSolution.sol")
    variables,values,coeffs,offset,gap=model.problem.variables.get_names(),model.problem.solution.get_values(),model.problem.objective.get_linear(),model.problem.objective.get_offset(),model.problem.solution.MIP.get_mip_relative_gap()
    variable2coeff={variables[i]:coeffs[i] for i in range(len(variables))}
    variable2value={variables[i]:values[i] for i in range(len(values))}
    variable2value.update({"offset":offset,"prepareTime":T2-T1,"cplexTime":T3-T2,"optimalityGap":gap})
    
    with open(resdire+"Variables.json","w") as outfile:
        json.dump(variable2value,outfile,indent=4)
    with open(resdire+"Coefficients.json","w") as outfile:
        json.dump(variable2coeff,outfile,indent=4)        
    with open(resdire+"Mode.json","w") as outfile:
        json.dump(mode,outfile,indent=4)
    
    print("Gap:",gap," prepareTime:",T2-T1," cplexTime",T3-T2)


def mainModelExecutor(dataset,scenario):
    mode={"MODEID":"Mode1",     # the directory name of mode setting
          "PAXTYPE":"PAX",      # (ITIN/PAX) aggregate the passengers of the same itinarery as an entity, or leave the passengers as individual entities
          "DELAYTYPE":"actual", # (approx/actual) calculate the delay cost by approximation method regarding the delay of flight (section 3.10.1), or by actual method regarding delay of passenger (section 3.10.3); Note that the combination ("ITIN","actual") is not allowed
          "CRSTIMECOMP":1,      # (0/1) allowed to compress the cruise time (1) or not (0) 
          "BOUNDETYPES":{
              "ACF":0,
              "CRW":0,
              "ITIN":0,
              "PAX":0   },      # (0/1) bound the size of partial network of each entity type (1) or not (0)
          "SIZEBOUND":1000,       # the upper bound of the number of arcs in the partial network according to PSCA algorithm, which is intended to control the size of partial network
          "MIPTOLERANCE":0.0,  # the relative mip tolerance of optimality gap
          "TIMELIMIT":1000,      # the limit of duration in seconds for cplex computation
          }
    
    executeModel(dataset,scenario,mode)
        



