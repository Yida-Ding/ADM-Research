import pandas as pd
import networkx as nx
import json
import cplex
from collections import defaultdict
from lxml import etree
from NetworkGenerator import Scenario

class Analyzer:
    def __init__(self,dataset,scenario,modeid):
        self.S=Scenario(dataset,scenario)
        self.scenario=scenario
        self.dataset=dataset
        self.modeid=modeid
        with open("Results/%s/%s/Mode.json"%(self.scenario,self.modeid),"r") as outfile:
            self.mode=json.load(outfile)
        with open("Results/%s/%s/Variables.json"%(self.scenario,self.modeid),"r") as outfile:
            self.variable2value=json.load(outfile)            
        with open("Results/%s/%s/Coefficients.json"%(self.scenario,self.modeid),"r") as outfile:
            self.variable2coeff=json.load(outfile)            
        tree=etree.parse("Results/%s/%s/ModelSolution.sol"%(self.scenario,self.modeid))
        for x in tree.xpath("//header"):
            self.objective=float(x.attrib["objectiveValue"])
            
for i in range(6):
    analyzer=Analyzer("ACF80","ACF80-SC1","Mode%d"%i)
    time=analyzer.variable2value["runtime"]
    print(time)
    
