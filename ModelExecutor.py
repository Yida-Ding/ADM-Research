import random
import networkx as nx
from collections import defaultdict
import pandas as pd
import haversine
import json
import os

from ModelGenerator import MIPModel
from NetworkGenerator import Scenario,Entity,PSCAHelper

'''
PARAMETERS:
    paxtype="ITIN"/"PAX" : aggregate the passengers of the same itinarery or leave the passengers as individuals
    delaytype="APPROX"/"ACTUAL" : calculate the delay cost by approximation (3.10.1) or by actual delay (3.10.3); Note that the combination ("ITIN","ACTUAL") is not allowed
    timelimit=integer : the duration in seconds for cplex computation
    sizebound=integer : the upper bound of the number of arcs in the partial network according to PSCA algorithm, which is intended to control the size of partial network
'''

def executeModel(dataset,scenario,paxtype="ITIN",delaytype="APPROX",timelimit=60,sizebound=float("inf")):
    S=Scenario(dataset,scenario)
    type2entity={}
    type2entity["ACF"]=PSCAHelper(S,"ACF",sizebound=sizebound).entities
    type2entity["CRW"]=PSCAHelper(S,"CRW",sizebound=sizebound).entities
    type2entity[paxtype]=PSCAHelper(S,paxtype,sizebound=None).entities
    
    model=MIPModel(S,type2entity)
    model.setFlowBalanceConstraint()
    model.setNodeClosureConstraint()
    model.setFlightTimeConstraint()
    model.setSourceArcConstraint()
    model.setSinkArcConstraint()
    model.setIntermediateArcConstraint()
    model.setSeatCapacityConstraint()
    model.setCruiseSpeedConstraint()
    model.setSpeedCompressionConstraint()
    model.addFlightCancellationCost()
    model.addFuelCost()
    model.addFollowScheduleCost()
    if delaytype=="APPROX":
        model.addApproximatedDelayCost()    #for passenger aggregation
    else:
        model.addActualDelayCost()     #for individual passenger
    model.problem.parameters.timelimit.set(timelimit)
    model.solveProblem()
    
    if not os.path.exists("Results"):
        os.makedirs("Results")
    if not os.path.exists("Results/%s"%scenario):
        os.makedirs("Results/%s"%scenario)
    if not os.path.exists("Results/%s/%s-%s"%(scenario,paxtype,delaytype)):
        os.makedirs("Results/%s/%s-%s"%(scenario,paxtype,delaytype))
        
    model.problem.solution.write("Results/%s/%s-%s/ModelSolution.sol"%(scenario,paxtype,delaytype))
    variables=model.problem.variables.get_names()
    values=model.problem.solution.get_values()
    variable2value={variables[i]:values[i] for i in range(len(values))}
    variable2value.update({"paxtype":paxtype,"delaytype":delaytype})
    with open("Results/%s/%s-%s/Variables.json"%(scenario,paxtype,delaytype),"w") as outfile:
        json.dump(variable2value,outfile,indent=4)
    

executeModel("ACF10","ACF10-SC1",paxtype="ITIN",delaytype="APPROX",timelimit=120,sizebound=60)
executeModel("ACF10","ACF10-SC1",paxtype="PAX",delaytype="ACTUAL",timelimit=120,sizebound=60)


