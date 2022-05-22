import shutil
import time
import pandas as pd
from Scenarios.ScenarioGenerator import ScenarioGenerator
from ModelExecutor import mainModelExecutor
from ResultAnalyzer import mainResultAnalyzer
import random

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


#    main("ACF5","ACF5-SCm",0.1,17)
#    main("ACF25","ACF25-SCm",0.1,3)
    
if __name__ == '__main__':
    createScenario("ACF25","ACF25-SCp",0.3,114)
#    main("ACF25","ACF25-SCp",0.3,114)

