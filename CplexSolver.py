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

def main(dataset,scenario,k,mode,seed=0):
    SC=ScenarioGenerator(dataset,scenario,seed)
    delayinfo=SC.getRandomFlightDelay(k)
    SC.setFlightDepartureDelay(delayinfo)
    SC.setDelayedReadyTime({})
    try:
        mainModelExecutor(dataset,scenario,mode)
        mainResultAnalyzer(dataset,scenario)
    except: # cplex no solution error
        shutil.rmtree("Scenarios/"+scenario)
        shutil.rmtree("Results/"+scenario)
        return False
    return True

def runCPLEX(dataset,scenario,mode):
    try:
        mainModelExecutor(dataset,scenario,mode)
        mainResultAnalyzer(dataset,scenario)
    except: # cplex no solution error
        shutil.rmtree("Results/"+scenario)
        return False
    return True


if __name__ == '__main__':
    
    mode={"MODEID":"Mode1",     # the directory name of mode setting
          "PAXTYPE":"PAX",      # (ITIN/PAX) aggregate the passengers of the same itinarery as an entity, or leave the passengers as individual entities
          "DELAYTYPE":"actual", # (approx/actual) calculate the delay cost by approximation method regarding the delay of flight (section 3.10.1), or by actual method regarding delay of passenger (section 3.10.3); Note that the combination ("ITIN","actual") is not allowed
          "CRSTIMECOMP":0,      # (0/1) allowed to compress the cruise time (1) or not (0) 
          "BOUNDETYPES":{
              "ACF":0,
              "CRW":0,
              "ITIN":0,
              "PAX":0   },      # (0/1) bound the size of partial network of each entity type (1) or not (0)
          "SIZEBOUND":1000,       # the upper bound of the number of arcs in the partial network according to PSCA algorithm, which is intended to control the size of partial network
          "MIPTOLERANCE":0.,  # the relative mip tolerance of optimality gap
          "TIMELIMIT":1000,      # the limit of duration in seconds for cplex computation
          }
          
    size=25
    runCPLEX("ACF%d"%size,"ACF%d-SCp"%size,mode)
    
#    for size in range(5,35,5):
#        runCPLEX("ACF%d"%size,"ACF%d-SCm"%size,mode)
#        runCPLEX("ACF%d"%size,"ACF%d-SCp"%size,mode)
#
#    main("ACF25","ACF25-SCp",0.3,mode,0)
#    createScenario("ACF25","ACF25-SCp",0.3,0)
          
         







