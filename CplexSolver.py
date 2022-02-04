import shutil
import time
import pandas as pd
from Scenarios.ScenarioGenerator import ScenarioGenerator
from ModelExecutor import mainModelExecutor
from ResultAnalyzer import mainResultAnalyzer

def delete(dataset,scenario):
    shutil.rmtree("Scenarios/"+scenario)
    shutil.rmtree("Results/"+scenario)
    
def main(dataset,scenario,seed=0):
    SC=ScenarioGenerator(dataset,scenario,seed)
    delayinfo=SC.getRandomFlightDelay(3)
    SC.setFlightDepartureDelay(delayinfo)
    SC.setDelayedReadyTime({})
    try:
        mainModelExecutor(dataset,scenario)
        mainResultAnalyzer(dataset,scenario)
    except: # cplex no solution error
        shutil.rmtree("Scenarios/"+scenario)
        shutil.rmtree("Results/"+scenario)
        return False
    return True

def runCPLEX(dataset,scenario):
    t1=time.time()
    try:
        mainModelExecutor(dataset,scenario)
        mainResultAnalyzer(dataset,scenario)
    except: # cplex no solution error
        shutil.rmtree("Scenarios/"+scenario)
        shutil.rmtree("Results/"+scenario)
    t2=time.time()
    return t2-t1

def runMain(dataset):
    seed=0
    for i in range(1,10):
        while True:
            flag=main(dataset,dataset+"-SC%d"%i,seed)
            if flag:
                seed+=1
                break
            else:
                seed+=1

deltas=[]
for dataset in ["ACF4","ACF5"]:
    for i in range(1,10):
        delta=runCPLEX(dataset,dataset+"-SC%d"%i)
        deltas.append(delta)

df=pd.read_csv("Results/Stats.csv",na_filter=None)
df["CPLEX_time"]=deltas
df.to_csv("Results/Stats.csv",index=None)






  