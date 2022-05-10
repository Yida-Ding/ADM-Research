import pandas as pd
import numpy as np
from ResultAnalyzer import getDatasetSummary,getScenarioSummary
import json

#data=[]
#for size in range(5,35,5):
#    row=["AC%02d"%size]
#    row+=getDatasetSummary("ACF%d"%size)
#    row+=getScenarioSummary("ACF%d"%size,"ACF%d-SCm"%size)
#    row+=getScenarioSummary("ACF%d"%size,"ACF%d-SCp"%size)
#    data.append(row)
#
#columns=["Dataset","|F|","|A|","|Uac|","|Ucr|","|Uit|","|Ups|","<2h",">2h","total","<2h",">2h","total"]
#df=pd.DataFrame(data,columns=columns,index=None)
#df.set_index('Dataset',inplace=True)
#res=df.to_latex()
#print(res)

def getDataRow(dataset,scenario):
    res=[]
    times=[]
    for method in ["PPO","uniform","distance","degree"]:
        ys=np.load('Results/%s/res_%s.npz'%(scenario,method))['res']
        ts=np.load('Results/%s/time_%s.npz'%(scenario,method))['res']
        obj=np.mean(ys[-100:])
        time=np.mean(ts[-100:])
        res.append(obj)
        times.append(time)
        
    with open("Results/%s/CostCPLEX.json"%scenario,"r") as outfile:
        cplexobj=json.load(outfile)["Objective"]
        res.append(cplexobj)
        times.append(1000)
    
    pct=[" ({0:.2%})".format(abs((r-res[-1])/res[-1])) for r in res[:-1]]+[""]
    res=["%.1f"%res[i]+pct[i] for i in range(len(res))]
    times=["%.3f"%t for t in times]
    
    ans=[]
    for v in range(len(res)):
        ans.append(res[v])
        ans.append(times[v])
    return ans

def getRowDown(dataset,scenario):
    res=[]
    times=[]
    with open("Results/%s/CostVNS.json"%scenario,"r") as outfile:
        ppoobj=json.load(outfile)["Objective"]
        res.append(ppoobj)
        times.append(999)
        
    for method in ["uniform","distance","degree"]:
        ys=np.load('Results/%s/res_%s.npz'%(scenario,method))['res']
        ts=np.load('Results/%s/time_%s.npz'%(scenario,method))['res']
        obj=np.mean(ys[-100:])
        time=np.mean(ts[-100:])
        res.append(obj)
        times.append(time)
        
    with open("Results/%s/CostCPLEX.json"%scenario,"r") as outfile:
        cplexobj=json.load(outfile)["Objective"]
        res.append(cplexobj)
        times.append(1000)

    pct=[" ({0:.2%})".format(abs((r-res[-1])/res[-1])) for r in res[:-1]]+[""]
    res=["%.1f"%res[i]+pct[i] for i in range(len(res))]
    times=["%.3f"%t for t in times]
    
    ans=[]
    for v in range(len(res)):
        ans.append(res[v])
        ans.append(times[v])
    return ans



data=[]
for i in range(5,20,5):
    for typ in ['m','p']:
        data.append(["ACF%d-SC%s"%(i,typ)]+getDataRow("ACF%d"%i,"ACF%d-SC%s"%(i,typ)))
        
for i in range(20,35,5):
    for typ in ['m','p']:
        data.append(["ACF%d-SC%s"%(i,typ)]+getRowDown("ACF%d"%i,"ACF%d-SC%s"%(i,typ)))

columns=["Instance","PPO-VNS","U-VNS","DT-VNS","DG-VNS","CPLEX","PPO-VNS","U-VNS","DT-VNS","DG-VNS","CPLEX"]
df=pd.DataFrame(data,columns=columns,index=None)
df.set_index('Instance',inplace=True)
res=df.to_latex()
print(res)
        
        
        
        
        
        



