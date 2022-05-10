import shutil
import time
import pandas as pd
from Scenarios.ScenarioGenerator import ScenarioGenerator
from ModelExecutor import mainModelExecutor
from ResultAnalyzer import mainResultAnalyzer

def createScenario(dataset,scenario,k,seed=0):
    SC=ScenarioGenerator(dataset,scenario,seed)
    delayinfo=SC.getRandomFlightDelay(k)
    SC.setFlightDepartureDelay(delayinfo)
    SC.setDelayedReadyTime({})

def main(dataset,scenario,k,seed=0):
    SC=ScenarioGenerator(dataset,scenario,seed)
    delayinfo=SC.getRandomFlightDelay(k)
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

def mainWithoutExcept(dataset,scenario,seed=0):
    SC=ScenarioGenerator(dataset,scenario,seed)
    delayinfo=SC.getRandomFlightDelay(2)
    SC.setFlightDepartureDelay(delayinfo)
    SC.setDelayedReadyTime({})
    mainModelExecutor(dataset,scenario)
    mainResultAnalyzer(dataset,scenario)

def runCPLEX(dataset,scenario):
    mainModelExecutor(dataset,scenario)
    mainResultAnalyzer(dataset,scenario)

def runMain(dataset):
    seed=0
    for i in range(5):
        while True:
            flag=main(dataset,dataset+"-SC%d"%i,seed)
            if flag:
                seed+=1
                break
            else:
                seed+=1

if __name__ == '__main__':
    
    for i in range(100,400,50):
        for typ in ["m","p"]:
            if typ=='m':
                createScenario("ACF%d"%i,"ACF%d-SC%s"%(i,typ),0.1,0)
            else:
                createScenario("ACF%d"%i,"ACF%d-SC%s"%(i,typ),0.3,0)



