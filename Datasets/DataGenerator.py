import random
import networkx as nx
from collections import defaultdict
import pandas as pd
import haversine
import json

class Scenario:
    def __init__(self,config):
        self.config=config
        self.dfacmodels=pd.read_excel("External/FAAAircraftCharacteristicDatabase/AircraftModels.xlsx",na_filter=None).head(config["MAXACT"])
        self.dfairports=pd.read_csv("External/OurAirports/airports.csv",na_filter=None)
        self.dfairports=self.dfairports[(self.dfairports["iso3"]=="USA")&((self.dfairports["lon"]>-130))].head(self.config["MAXAPT"]) # USA without Hawaii and Alaska
        self.airports=self.dfairports["iata_code"].tolist()
        self.ap2loc={row.iata_code:(row.lat,row.lon) for row in self.dfairports.itertuples()}
        self.ap2pax={row.iata_code:row.pax for row in self.dfairports.itertuples()}
        self.appair2distance={(k1,k2):haversine.haversine(self.ap2loc[k1],self.ap2loc[k2]) for k1 in self.ap2loc for k2 in self.ap2loc}
        self.connectableappairs={k for k in self.appair2distance if self.appair2distance[k]>self.config["MINFLIGHTDISTANCE"] and self.appair2distance[k]<self.config["MAXFLIGHTDISTANCE"]} # Eliminate extremely short flights
        self.Gconnectable=nx.Graph(self.connectableappairs)

    def getTimeString(self,seconds):
        days,remainder=divmod(seconds,24*3600)
        hours,remainder=divmod(remainder,3600)
        minutes,seconds=divmod(remainder,60)
        s='{:02}:{:02}'.format(int(hours),int(minutes))
        if days>0:
            s+=" (+%d)"%days
        return s

class CrewHelper:
    def __init__(self,S):
        self.S=S
        self.crewFlights={}

    def getAvailableCrew(self,tail,origin,destination,depTime,arrTime):
        for c in self.crewFlights:
            lastFlight=self.crewFlights[c][-1]
            if lastFlight[1]==origin and lastFlight[3]+self.S.config["CREWCONTIME"]<=depTime:
                # Check whether to really use the crew
                if lastFlight[4]!=tail or sum([cf[3]-cf[2] for cf in self.crewFlights[c]])<self.S.config["CREWMAXREPTIME"]:
                    self.crewFlights[c].append((origin,destination,depTime,arrTime,tail))
                    return c
        # No crew found, create a new one
        crewname="C%02d"%len(self.crewFlights)
        self.crewFlights[crewname]=[(origin,destination,depTime,arrTime,tail)]
        return crewname
    
class ItineraryHelper:
    def __init__(self,S):
        self.S=S
        self.itinFlights={}
        self.itinPax={}
        
    def getAvailableItinerary(self,fltname,origin,destination,depTime,arrTime,pax):
        leaveall=0
        for it in self.itinFlights.copy():
            lastFlight=self.itinFlights[it][-1]
            if lastFlight[2]==origin and lastFlight[4]+self.S.config["PAXCONTIME"]<=depTime:
                if len(self.itinFlights[it])==1 and random.uniform(0,1)>self.S.config["DIRECTITINPROB"]:
                    newItin="I%02d"%len(self.itinFlights)
                    self.itinFlights[newItin]=[lastFlight,(fltname,origin,destination,depTime,arrTime,pax)]
                    leave=int(random.uniform(0.3,0.5)*self.itinPax[it])
                    self.itinPax[it]-=leave
                    self.itinPax[newItin]=leave
                    leaveall+=leave
                    
                elif len(self.itinFlights[it])>=2 and random.uniform(0,1)>self.S.config["DIRECTITINPROB"]+self.S.config["TWOHOPITINPROB"]:
                    newItin="I%02d"%len(self.itinFlights)
                    self.itinFlights[newItin]=self.itinFlights[it]+[(fltname,origin,destination,depTime,arrTime,pax)]
                    leave=int(random.uniform(0.3,0.5)*self.itinPax[it])
                    self.itinPax[it]-=leave
                    self.itinPax[newItin]=leave
                    leaveall+=leave
                    
        itin="I%02d"%len(self.itinFlights)
        self.itinFlights[itin]=[(fltname,origin,destination,depTime,arrTime,pax)]
        self.itinPax[itin]=pax-leaveall
        leaveall=0
        
        
def generateScenario(filename,config,seed=0):
    random.seed(seed)
    S=Scenario(config)
    crewHelper=CrewHelper(S)
    itinHelper=ItineraryHelper(S)
    trajectories=[]
    for i in range(0,S.config["MAXAC"]):
        acname="T%02d"%i
        actyperow=S.dfacmodels.iloc[random.randint(0, len(S.dfacmodels)-1)]
        flights=[]
        flightind=0
        while True:
            if len(flights)==0:
                # Start trajectory anywhere randomly
                origin=S.airports[random.randint(0,len(S.airports)-1)]
                depTime=S.config["STARTTIME"]+random.randint(30,50)*60
            else:
                # Start at previous airport with overhead time
                oldname,oldorigin,olddestination,olddepTime,oldarrTime,oldcruiseTime,oldcrew,olddistance,oldpax=flights[-1]
                origin=olddestination
                depTime=oldarrTime+S.config["MINCONTIME"]+int(random.uniform(0.0,1.0)*(S.config["MAXCONTIME"]-S.config["MINCONTIME"]))
    
            if random.uniform(0.0,1.0)<0.3 and len(flights)>=1: #TODO: this needs to be better parameterized in the future! In general, some constant in the range 0.0 (=hub and spoke) and 1.0 (=point-to-point) would be nice.
                #return to old origin
                destination=flights[-1][1]
            else:
                neighbors=list(S.Gconnectable.neighbors(origin))
                destination=random.choices(population=neighbors,weights=[S.ap2pax[k] for k in neighbors],k=1)[0]
            distance=int(S.appair2distance[(origin,destination)])
            flightTime=int(distance*3600/800)
            cruiseTime=max(flightTime-30*60,0)
            arrTime=depTime+flightTime
            if arrTime>=S.config["ENDTIME"]:
                break
            fltname=acname+"F%02d"%flightind
            flightind+=1
            pax=int(S.config["LOADFACTOR"]*actyperow.PAX)
            crew=crewHelper.getAvailableCrew(acname,origin,destination,depTime,arrTime)
            itin=itinHelper.getAvailableItinerary(fltname,origin,destination,depTime,arrTime,pax)
            flights+=[(fltname,origin,destination,depTime,arrTime,cruiseTime,crew,distance,pax)]
        
        trajectories.append((acname,actyperow.Model,actyperow.PAX,flights))
        
    resd=defaultdict(list)
    for trajectory in trajectories:
        acname,acmodel,accap,flights=trajectory
        for flight in flights:
            resd["Tail"].append(acname)
            resd["Flight"].append(flight[0])
            resd["Crew"].append(flight[6])
            resd["From"].append(flight[1])
            resd["To"].append(flight[2])
            resd["SDT"].append(flight[3])
            resd["SAT"].append(flight[4])
            resd["Flight_time"].append(flight[4]-flight[3])
            resd["Cruise_time"].append(flight[5])
            resd["Distance"].append(flight[7])
            resd["Capacity"].append(accap)
            resd["Pax"].append(flight[8])
            resd["Timestring"].append(S.getTimeString(flight[3])+" -> "+S.getTimeString(flight[4]))
    
    resditin=defaultdict(list)
    for itin,flights in itinHelper.itinFlights.items():
        resditin["Itinerary"].append(itin)
        resditin["Flight_legs"].append("-".join([flight[0] for flight in flights]))
        resditin["Pax"].append(itinHelper.itinPax[itin])
        
    pd.DataFrame(resd).to_csv("Schedule_"+filename+".csv",index=False)
    pd.DataFrame(resditin).to_csv("Itinerary_"+filename+".csv",index=False)
    with open("Schedule_"+filename+".json", "w") as outfile:
        json.dump(config, outfile, indent = 4)
        
config={"MAXAC":5, # Number of aicraft trajectories to generate
        "MAXACT":5, # Number of unique aircraft types
        "MAXAPT":5, # Number of airports
        "LOADFACTOR":0.8, # Load factor for generating passengers from aircraft capacity
        "MINFLIGHTDISTANCE":400, # No flights shorter than this distance
        "MAXFLIGHTDISTANCE":3000, # No flights longer than this distance
        "MINCONTIME":30*60, # Minimum connection time for aircraft
        "MAXCONTIME":50*60, # Maximum connection time for aircraft
        "CREWCONTIME":30*60, # Time for crew to be ready for next flight
        "CREWMAXREPTIME":8*3600, # Time for crew to have a break "from a tail"
        "PAXCONTIME":30*60, # Time for passenger to be ready for next flight
        "STARTTIME":5*3600, #start at 5AM
        "ENDTIME":26*3600, # stop at 2AM next day
        "DIRECTITINPROB":0.85, # probability of direct itinerary
        "TWOHOPITINPROB":0.12, # probability of two-hop itinerary
        }

generateScenario("ACF%d"%config["MAXAC"],config,seed=0)



