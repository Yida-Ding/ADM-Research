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
        with open("../Datasets/"+direname+"/Config.json", "r") as outfile:
            self.config=json.load(outfile)
        
        self.flights=set(self.dfschedule["Flight"].tolist())
        self.crews=set(self.dfschedule["Crew"].tolist())
        self.airports=set(self.dfschedule["From"].tolist()+self.dfschedule["To"].tolist())
        
class ScenarioGenerator:        
    def __init__(self,direname,scname,seed):
        random.seed(seed)
        self.direname=direname
        self.scname=scname
        self.D=Dataset(direname)
        if not os.path.exists("%s-%s"%(self.direname,self.scname)):
            os.makedirs("%s-%s"%(self.direname,self.scname))
            
    def getTimeString(self,seconds):
        days,remainder=divmod(seconds,24*3600)
        hours,remainder=divmod(remainder,3600)
        minutes,seconds=divmod(remainder,60)
        s='{:02}:{:02}'.format(int(hours),int(minutes))
        return s

    def setFlightDepartureDelay(self,flight2delay={}):
        dfdrpschedule=self.D.dfschedule.copy()
        dfdrpschedule["is_disrupted"]=[0]*len(dfdrpschedule)
        for flight,delayTime in flight2delay.items():
            temp=self.D.dfschedule.loc[self.D.dfschedule['Flight']==flight,["SDT","SAT"]]
            newSDT,newSAT=list(temp["SDT"])[0]+delayTime,list(temp["SAT"])[0]+delayTime
            dfdrpschedule.loc[dfdrpschedule['Flight']==flight,["SDT","SAT","Timestring","is_disrupted"]]=[newSDT,newSAT,self.getTimeString(newSDT)+" -> "+self.getTimeString(newSAT),1]        
        dfdrpschedule.to_csv("%s-%s"%(self.direname,self.scname)+"/DrpSchedule.csv",index=False)
    
    def setDelayedReadyTime(self,entity2delay={}):
        with open("%s-%s"%(self.direname,self.scname)+"/DelayedReadyTime.json", "w") as outfile:
            json.dump(entity2delay,outfile,indent=4)
    
    def getRandomFlightDelay(self,k):
        selflights=random.sample(self.D.flights,k)
        return {flight:random.randint(100,1000) for flight in selflights}

for i in range(50,450,50):
    SC1=ScenarioGenerator("ACF%d"%i,"SC1",1)
    data=SC1.getRandomFlightDelay(10)
    SC1.setFlightDepartureDelay(data)
    SC1.setDelayedReadyTime({})
    



        