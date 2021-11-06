import random
import networkx as nx
from collections import defaultdict
import pandas as pd
import haversine
import json
import os

from ModelGenerator import MIPModel
from NetworkGenerator import Scenario,Entity,PSCAHelper


# sizebound: The upper bound of the number of arcs in the partial network according to PSCA algorithm on page 19
def executeModel(dataset,scenario,timelimit=60,sizebound=float("inf")):
    S=Scenario(dataset,scenario)
    pscaACF=PSCAHelper(S,"ACF",sizebound)
    pscaCRW=PSCAHelper(S,"CRW",sizebound)
    
    type2entity={}
    type2entity["ACF"]=pscaACF.entities
    type2entity["CRW"]=pscaCRW.entities
    type2entity["PAX"]=[Entity(S,pname,"PAX") for pname in S.paxname2flights]
    
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
#    model.addActualDelayCost()     #for individual passenger
    model.addApproximatedDelayCost()    #for passenger aggregation
    model.addFollowScheduleCost()
        
    model.problem.parameters.timelimit.set(timelimit)
    model.solveProblem()
    
    if not os.path.exists("Results"):
        os.makedirs("Results")
    if not os.path.exists("Results/%s"%scenario):
        os.makedirs("Results/%s"%scenario)
        
    model.problem.solution.write("Results/%s/ModelSolution.sol"%scenario)
    variables=model.problem.variables.get_names()
    values=model.problem.solution.get_values()
    variable2value={variables[i]:values[i] for i in range(len(values))}
    with open("Results/%s/Variables.json"%scenario,"w") as outfile:
        json.dump(variable2value,outfile,indent=4)
    

dataset="ACF2"
executeModel(dataset,dataset+"-SC1",120)


