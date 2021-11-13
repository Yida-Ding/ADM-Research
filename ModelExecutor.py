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
    if not os.path.exists("Results/%s/%s"%(scenario,mode["MODEID"])):
        os.makedirs("Results/%s/%s"%(scenario,mode["MODEID"]))

    Tstart=time.time()
    S=Scenario(dataset,scenario)
    type2entity={}
    for etype in ["ACF","CRW",mode["PAXTYPE"]]:
        pscahelper=PSCAHelper(S,etype,mode["BOUNDETYPES"][etype],mode["SIZEBOUND"])
        type2entity[etype]=pscahelper.entities
        pscahelper.getGraphReductionStat(mode["MODEID"])

    model=MIPModel(S,type2entity)
    model.problem.solve()
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
    print(model.problem.get_stats())
    model.problem.solve()
    Tfinal=time.time()
    
    resdire="Results/%s/%s/"%(scenario,mode["MODEID"])
    model.problem.write(resdire+"ModelProblem.lp")
    model.problem.solution.write(resdire+"ModelSolution.sol")
    variables,values,coeffs,offset=model.problem.variables.get_names(),model.problem.solution.get_values(),model.problem.objective.get_linear(),model.problem.objective.get_offset()
    variable2coeff={variables[i]:coeffs[i] for i in range(len(variables))}
    variable2value={variables[i]:values[i] for i in range(len(values))}
    variable2value.update({"offset":offset,"runtime":Tfinal-Tstart})
    with open(resdire+"Variables.json","w") as outfile:
        json.dump(variable2value,outfile,indent=4)
    with open(resdire+"Coefficients.json","w") as outfile:
        json.dump(variable2coeff,outfile,indent=4)        
    with open(resdire+"Mode.json","w") as outfile:
        json.dump(mode,outfile,indent=4)
    
    
mode={"MODEID":"Mode0",  # the directory name of mode setting
      "PAXTYPE":"PAX",  # (ITIN/PAX) aggregate the passengers of the same itinarery or leave the passengers as individuals
      "DELAYTYPE":"actual",  # (approx/actual) calculate the delay cost by approximation (3.10.1) or by actual delay (3.10.3); Note that the combination ("ITIN","actual") is not allowed
      "CRSTIMECOMP":1,  # (0/1) allowed to compress the cruise time or not 
      "SIZEBOUND":50,  # the upper bound of the number of arcs in the partial network according to PSCA algorithm, which is intended to control the size of partial network
      "BOUNDETYPES":{"ACF":1,"CRW":1,"ITIN":1,"PAX":1},   # (0/1) the entity types that are bounded by PSCA by sizebound, by default all of the entities will be bounded by PSCA     
      "TIMELIMIT":120   # the duration in seconds for cplex computation
      }

executeModel("ACF10","ACF10-SC1",mode)









