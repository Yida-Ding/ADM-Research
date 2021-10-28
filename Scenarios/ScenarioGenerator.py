import random
import networkx as nx
from collections import defaultdict
import pandas as pd
import haversine
import json
import os


class Dataset:
    def __init__(self,direname):
        self.direname=direname
        self.dfschedule=pd.read_csv("../Datasets/"+direname+"/Schedule.csv",na_filter=None)
        self.dfitinerary=pd.read_csv("../Datasets/"+direname+"/Itinerary.csv",na_filter=None)
        self.dfduration=pd.read_csv("../Datasets/"+direname+"/Duration.csv",na_filter=None)        
        with open("../Datasets/"+direname+"/Config.json", "r") as outfile:
            self.config=json.load(outfile)
        
        self.flights=set(self.dfschedule["Flight"].tolist())
        self.crews=set(self.dfschedule["Crew"].tolist())
        self.airports=set(self.dfschedule["From"].tolist()+self.dfschedule["To"].tolist())
        
class ScenarioGenerator:        
    def __init__(self,direname,scname):
        self.direname=direname
        self.scname=scname
        self.D=Dataset(direname)
        self.drptype2data=defaultdict(list)
        self.drptypes=["FlightDepartureDelay","FlightCancellation","DelayedReadyTime","AirportClosure"]
    
    def setFlightDepartureDelay(self,flightName,delayTime):
        self.drptype2data[self.drptypes[0]].append((flightName,delayTime))
    
    def setFlightCancellation(self,flightName):
        self.drptype2data[self.drptypes[1]].append(flightName)
    
    def setDelayedReadyTime(self,entityName,delayTime):
        self.drptype2data[self.drptypes[2]].append((entityName,delayTime))
    
    def setAirportClosure(self,airport,startTime,endTime):
        self.drptype2data[self.drptypes[3]].append((airport,startTime,endTime))
        
    def tojsonfile(self):
        if not os.path.exists("%s-%s"%(self.direname,self.scname)):
            os.makedirs("%s-%s"%(self.direname,self.scname))
        with open("%s-%s"%(self.direname,self.scname)+"/DisruptionScenario.json", "w") as outfile:
            json.dump(self.drptype2data, outfile, indent = 4)
    
for i in [2,5,10]:
    direname="ACF%d"%i
    D=Dataset(direname)
    SC0=ScenarioGenerator(direname,"SC0")
    SC0.tojsonfile()


    
