import pandas as pd
import os
from collections import defaultdict

resd=defaultdict(list)
for dire in os.listdir():
    if os.path.isdir(dire):
        dfcostCPLEX=pd.read_csv(dire+'/Cost.csv',na_filter=None)
        dfcostVNS=pd.read_csv(dire+'/CostVNS.csv',na_filter=None)
        costCPLEX=dfcostCPLEX["Value"].tolist()[1]
        costVNS=dfcostVNS["Value"].tolist()[0]
        resd["Scenario"].append(dire)
        resd["CPLEX"].append(costCPLEX)
        resd["VNS"].append(costVNS)
        resd["Gap"].append("{0:.0%}".format((costVNS-costCPLEX)/costCPLEX))
pd.DataFrame(resd).to_csv("Stats.csv",index=None)
