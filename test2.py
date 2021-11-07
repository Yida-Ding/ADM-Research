import cplex
import numpy as np
from collections import defaultdict


flights=["F1"]
dicta={"F1":[("I0",5),("I1",3)]}
dictb={"F1":[("I1",5),("I2",3)]}
for flight in flights:
    d1=dict(dicta[flight])
    d2=dict(dictb[flight])
    indict=defaultdict(int)
    outdict=defaultdict(int)
    for itin in set(d1.keys())|set(d2.keys()):
        indict[itin]+=d2.get(itin,0)-d1.get(itin,0)
    print(indict)
    print(d1)
    print(d2)











