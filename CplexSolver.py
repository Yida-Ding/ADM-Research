from Scenarios.ScenarioGenerator import ScenarioGenerator
from ModelExecutor import mainModelExecutor
from ResultAnalyzer import mainResultAnalyzer

def main(dataset,dflight,dtime):
    scenario=dataset+'-'+dflight+'d%dh'%dtime
    
    SC=ScenarioGenerator(dataset,scenario,1)
    SC.setFlightDepartureDelay({dflight:3600*dtime})
    SC.setDelayedReadyTime({})
    
    mainModelExecutor(dataset,scenario)
    mainResultAnalyzer(dataset,scenario)

main("ACF2","F00",3)   
#main("ACF3","F05",5)



  