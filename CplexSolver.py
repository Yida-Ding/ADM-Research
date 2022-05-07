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
    seedL=[[7,0],[0,7],[1,6],[1,2],[9,18],[2,58]]
    i=0
    for size in range(5,35,5):
        createScenario("ACF%d"%size,"ACF%d-SCm"%size,0.1,seedL[i][0])
        createScenario("ACF%d"%size,"ACF%d-SCp"%size,0.3,seedL[i][1])
        i+=1
    
#    main("ACF30","ACF30-SCp",0.3,58)
#    createScenario("ACF5","ACF5-SCm",0.1,7)
#    runCPLEX("ACF5","ACF5-SCm")




