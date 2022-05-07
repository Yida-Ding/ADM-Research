import shutil
import time
import pandas as pd
from Scenarios.ScenarioGenerator import ScenarioGenerator
from ModelExecutor import mainModelExecutor
from ResultAnalyzer import mainResultAnalyzer

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
    
    main("ACF5","ACF5-SC1",12)
        




