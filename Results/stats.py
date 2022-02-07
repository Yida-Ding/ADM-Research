import pandas as pd
import os
from collections import defaultdict

resd=defaultdict(list)
for dire in os.listdir():
    if os.path.isdir(dire):
        with open(dire+"/CostCPLEX.json", "r") as outfile:
            costDict=json.load(outfile)
            costCPLEX=costDict["Objective"]
        with open(dire+"/CostVNS.json", "r") as outfile:
            costDict=json.load(outfile)
            costVNS=costDict["Objective"]
        resd["Scenario"].append(dire)
        resd["CPLEX"].append(costCPLEX)
        resd["VNS"].append(costVNS)
        resd["Gap"].append("{0:.0%}".format((costVNS-costCPLEX)/costCPLEX))
pd.DataFrame(resd).to_csv("Stats.csv",index=None)
