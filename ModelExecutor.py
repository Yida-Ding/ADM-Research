import random
import networkx as nx
from collections import defaultdict
import pandas as pd
import haversine
import json
import os

from ModelGenerator import MIPModel
from NetworkGenerator import Scenario,Entity


def executeModel(dataset,scenario):
    S=Scenario(dataset,scenario)
    type2entity={}
    type2entity["ACF"]=[Entity(S,tname,"ACF") for tname in S.tail2flights]
    type2entity["CRW"]=[Entity(S,cname,"CRW") for cname in S.crew2flights]
    type2entity["PAX"]=[Entity(S,pname,"PAX") for pname in S.pax2flights]
    
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
    model.addActualDelayCost()
#    model.addFollowScheduleCost()
    S.problem.solve()
    
    if not os.path.exists("Results"):
        os.makedirs("Results")
    if not os.path.exists("Results/%s"%scenario):
        os.makedirs("Results/%s"%scenario)
    S.problem.solution.write("Results/%s/ModelSolution.sol"%scenario)
    
    variables=S.problem.variables.get_names()
    values=S.problem.solution.get_values()
    variable2value={variables[i]:values[i] for i in range(len(values))}
    with open("Results/%s/Variables.json"%scenario,"w") as outfile:
        json.dump(variable2value,outfile,indent=4)


executeModel("ACF2","ACF2-SC0")




