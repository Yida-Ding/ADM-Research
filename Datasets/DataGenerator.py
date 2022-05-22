import random
import networkx as nx
from collections import defaultdict
import pandas as pd
import haversine
import json
import os
import sys
import numpy as np

class Dataset:
    def __init__(self,config):
        self.config=config
        self.dfacmodels=pd.read_excel("0_External/FAAAircraftCharacteristicDatabase/AircraftModels.xlsx",na_filter=None).head(config["MAXACT"])
        self.dfairports=pd.read_csv("0_External/OurAirports/airports.csv",na_filter=None)
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
    def __init__(self,D):
        self.D=D
        self.crewFlights={}

    def getAvailableCrew(self,tail,origin,destination,depTime,arrTime):
        for c in self.crewFlights:
            lastFlight=self.crewFlights[c][-1]
            timedelta=depTime-lastFlight[3]
            if lastFlight[1]==origin and self.D.config["CREWMINCONTIME"]<=timedelta and timedelta<=self.D.config["CREWMAXCONTIME"]:
                # Check whether to really use the crew
                if len(self.crewFlights[c])<self.D.config["CREWMAXLAND"] \
                    and sum([cf[3]-cf[2] for cf in self.crewFlights[c]])+(arrTime-depTime)<self.D.config["CREWMAXFLTTIME"] \
                    and arrTime-self.crewFlights[c][0][2]<self.D.config["CREWMAXDUTYTIME"]:
                                    
                    self.crewFlights[c].append((origin,destination,depTime,arrTime,tail))
                    return c
                
        # No crew found, create a new one
        crewname="C%02d"%len(self.crewFlights)
        self.crewFlights[crewname]=[(origin,destination,depTime,arrTime,tail)]
        return crewname
    
class ItineraryHelper:
    def __init__(self,D):
        self.D=D
        self.itinFlights={}
        self.itinPax={}
        
    def getAvailableItinerary(self,fltname,origin,destination,depTime,arrTime,pax):
        leaveall=0
        for it in self.itinFlights.copy():
            lastFlight=self.itinFlights[it][-1]
            timedelta=depTime-lastFlight[4]
            # make sure the new flight added to itin will not return back to the previous origins
            if lastFlight[2]==origin and timedelta>=self.D.config["PAXMINCONTIME"] and timedelta<=self.D.config["PAXMAXCONTIME"] \
                and destination not in [flight[1] for flight in self.itinFlights[it]]:
                    
                flag=False
                newItin="I%02d"%len(self.itinFlights)
                if len(self.itinFlights[it])==1:
                    leave=int(self.D.config["TWOHOPITINPROB"]*self.itinPax[it])
                    if leave>0:
                        self.itinFlights[newItin]=[lastFlight,(fltname,origin,destination,depTime,arrTime,pax)]
                        flag=True
                elif len(self.itinFlights[it])==2:
                    leave=int((1-self.D.config["DIRECTITINPROB"]-self.D.config["TWOHOPITINPROB"])*self.itinPax[it])
                    if leave>0:
                        self.itinFlights[newItin]=self.itinFlights[it]+[(fltname,origin,destination,depTime,arrTime,pax)]
                        flag=True
                
                if flag:
                    self.itinPax[it]-=leave
                    self.itinPax[newItin]=leave
                    leaveall+=leave
                        
        itin="I%02d"%len(self.itinFlights)
        self.itinFlights[itin]=[(fltname,origin,destination,depTime,arrTime,pax)]
        self.itinPax[itin]=pax-leaveall
        
        
def generateDataset(direname,config):
    random.seed(config["SEED"])
    D=Dataset(config)
    crewHelper=CrewHelper(D)
    itinHelper=ItineraryHelper(D)
    trajectories=[]
    flightind=0
    for i in range(0,D.config["MAXAC"]):
        acname="T%02d"%i
        actyperow=D.dfacmodels.iloc[random.randint(0, len(D.dfacmodels)-1)]
        flights=[]
        while True:
            if len(flights)==0:
                # Start trajectory anywhere randomly
                origin=D.airports[random.randint(0,len(D.airports)-1)]
                depTime=D.config["STARTTIME"]+random.randint(30,50)*60
            else:
                # Start at previous airport with overhead time
                oldname,oldorigin,olddestination,olddepTime,oldarrTime,oldcruiseTime,oldcrew,olddistance,oldpax=flights[-1]
                origin=olddestination
                depTime=oldarrTime+D.config["ACMINCONTIME"]+int(random.uniform(0.0,1.0)*(D.config["ACMAXCONTIME"]-D.config["ACMINCONTIME"]))
    
            if random.uniform(0.0,1.0)<0.3 and len(flights)>=1: # In general, some constant in the range 0.0 (=hub and spoke) and 1.0 (=point-to-point) would be nice.
                #return to old origin
                destination=flights[-1][1]
            else:
                neighbors=list(D.Gconnectable.neighbors(origin))
                destination=random.choices(population=neighbors,weights=[D.ap2pax[k] for k in neighbors],k=1)[0]
                
            distance=int(D.appair2distance[(origin,destination)])
            flightTime=int(distance/D.config["ACTAVGSPEED"])
            cruiseTime=max(flightTime-30*60,0)
            arrTime=depTime+flightTime
            if arrTime>=D.config["ENDTIME"]:
                break
            fltname="F%02d"%flightind
            flightind+=1
            pax=int(D.config["LOADFACTOR"]*actyperow.PAX)
            crew=crewHelper.getAvailableCrew(acname,origin,destination,depTime,arrTime)
            itinHelper.getAvailableItinerary(fltname,origin,destination,depTime,arrTime,pax)
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
            resd["Timestring"].append(D.getTimeString(flight[3])+" -> "+D.getTimeString(flight[4]))
    
    resditin=defaultdict(list)
    resdpax=defaultdict(list)
    for itin,flights in itinHelper.itinFlights.items():
        numPax=itinHelper.itinPax[itin]
        flightNames=[flight[0] for flight in flights]
        resditin["Itinerary"].append(itin)
        resditin["Flight_legs"].append("-".join(flightNames))
        resditin["From"].append(flights[0][1])
        resditin["To"].append(flights[-1][2])
        resditin["SDT"].append(flights[0][3])
        resditin["SAT"].append(flights[-1][4])
        resditin["Pax"].append(numPax)
        for i in range(numPax):
            resdpax["Pax"].append(itin+"+P%02d"%i)
            resdpax["Itinerary"].append(itin)
            resdpax["Flights"].append("-".join(flightNames))
    
    resdtime=defaultdict(list)
    for (ap1,ap2),distance in D.appair2distance.items():
        if ap1!=ap2:
            time=int(distance/D.config["ACTAVGSPEED"])
            resdtime["From"].append(ap1)
            resdtime["To"].append(ap2)
            resdtime["Distance"].append(distance)
            resdtime["Duration"].append(time)
            resdtime["Timestring"].append(D.getTimeString(time))
    
    if not os.path.exists(direname):
        os.makedirs(direname)
    pd.DataFrame(resd).to_csv(direname+"/Schedule.csv",index=False)
    pd.DataFrame(resditin).to_csv(direname+"/Itinerary.csv",index=False)
    pd.DataFrame(resdtime).to_csv(direname+"/Duration.csv",index=False)
    pd.DataFrame(resdpax).to_csv(direname+"/Passenger.csv",index=False)
    
    with open(direname+"/Config.json", "w") as outfile:
        json.dump(config, outfile, indent = 4)

if __name__=='__main__':
            
    config={"MAXAC":25, # Number of aicraft trajectories to generate
            "MAXAPT":25+random.randint(-2,2), # Number of airports
            "MAXACT":3, # Number of unique aircraft types
            "LOADFACTOR":0.8, # Load factor for generating passengers from aircraft capacity
            "MINFLIGHTDISTANCE":600, # No flights shorter than this distance
            "MAXFLIGHTDISTANCE":3000, # No flights longer than this distance
            "ACTAVGSPEED":800/3600, # Average speed of aircraft used to estimate flight duration
            "ACMINCONTIME":30*60, # Minimum connection time for aircraft
            "ACMAXCONTIME":600*60, # Maximum connection time for aircraft
            "STARTTIME":5*3600, # start at 5AM
            "ENDTIME":26*3600, # stop at 2AM next day
            "DIRECTITINPROB":0.85, # probability of direct itinerary
            "TWOHOPITINPROB":0.12, # probability of two-hop itinerary
            "CRSTIMECOMPPCT":0.09, # Cruise time compression limit in percentage (Page 6), maximum increase in cruise speed by a factor of 1.1
            "MAXHOLDTIME":2*3600, # Maximum departure/arrival hold time, corresponding to latest departure or arrival time
            "CRUISESTAGEDISTPCT":0.8, # Percentage of cruise stage distance with respect to the flight distance
            "FLIGHTCANCELCOST":20000, # Flight cancellation cost in dollar on page 22
            "FUELCOSTPERKG":0.478/0.454, # Jet fuel price per kg on page 22
            "FUELCONSUMPPARA":[0.01*3600,0.16*60,0.74/3600,2200/(60**3)], # Fuel consumption function parameters in 2014 paper
            "DELAYCOST":1.0242/60, # Delay cost per passenger per second on page 22
            "FOLLOWSCHEDULECOST":-1, # Negative cost to follow schedule arc for aircraft and crew teams on page 15
            "FOLLOWSCHEDULECOSTPAX":-0.1, # Negative cost to follow schedule arc for passenger on page 15
            "SEED":1, # Random seed
            
            "CREWMINCONTIME":30*60, # Minimum connection time for crew to be ready for next flight
            "CREWMAXCONTIME":4*3600, # Maximum connection sit-time between two consecutive flights within a same duty
            "CREWMAXLAND":4, # Maximum number of landings within a crew duty
            "CREWMAXFLTTIME":8*3600, # Maximum flying time within a duty
            "CREWMAXDUTYTIME":12*3600, # Maximum duty duration, including flight times and connection times
            "PAXMINCONTIME":30*60, # Minimum connection time for passenger to be ready for next flight
            "PAXMAXCONTIME":8*3600 # Maximum allowable connection time for passengers
            }

    generateDataset("ACF%d"%config["MAXAC"],config)
        
