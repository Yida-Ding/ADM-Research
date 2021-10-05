import random
import numpy as np
import pandas as pd
import networkx as nx
from haversine import haversine
from datetime import timedelta

class TimeHelper:
    def __init__(self,winStart,winEnd):
        #the start and end of recovery horizon/window
        self.winStart=timedelta(hours=winStart[0],minutes=winStart[1])
        self.winEnd=timedelta(hours=winEnd[0],minutes=winEnd[1])
        
    def time(self,hour,minute=0):
        return timedelta(hours=hour,minutes=minute)
    
    def tostring(self,td):
        hours,remainder=divmod(td.seconds,3600)
        minutes,seconds=divmod(remainder,60)
        return '{:02}:{:02}'.format(int(hours),int(minutes))
            
class AirportHelper:
    def __init__(self,MAXAPT=20,seed=1):
        #MAXAPT: number of airports in the following connected graph
        self.MAXAPT=MAXAPT
        df=pd.read_csv("External/OurAirports/airports.csv")
        df=df[(df["iso3"]=="USA")&(df["type"]=="large_airport")]
        df=df.sample(MAXAPT,random_state=seed)
        self.ap2loc={row.iata_code:(row.lat,row.lon) for row in df.itertuples()}
        self.airports=df["iata_code"].tolist()

        #create complete graph of airports
        self.apGraph=nx.complete_graph(MAXAPT)
        nx.relabel_nodes(self.apGraph,dict(enumerate(self.airports)),copy=False)
        
    def getDistance(self,src,dest):
        return round(haversine(self.ap2loc[src],self.ap2loc[dest]))

class CidHelper:
    def __init__(self):
        self.count=0
        self.cid="C0"
    
    def update(self):
        self.count+=1
        self.cid="C%d"%self.count


class Aircraft:
    def __init__(self,tail):
        self.tail=tail
        #assume three aircraft capacity type, small/medium/large 
        self.cpc=random.choice([180,210,240])
        
        #randomly select src and dest airport from the apGraph for the aircraft
        self.src,self.dest=random.sample(aph.airports,2)
        self.flights=[]
        curTime=tmh.winEnd
        
        #the condition is used to ensure a feasible journey taken by the aircraft
        while curTime>=tmh.winEnd:
            #create random simple path of random length within 7
            #may consider repeated nodes for the path in future
            self.pathleng=random.randint(5,7)
            self.path=next(nx.all_simple_paths(aph.apGraph,self.src,self.dest,self.pathleng))
            #create flight instances with assigned SDT and SAT
            temp=[]
            curTime=tmh.winStart
            for i in range(len(self.path)-1):
                flight=Flight(self.tail+"F%d"%i,self.path[i],self.path[i+1],self)
                #assume connection time ranging from 30-50 min
                flight.fromTime=curTime+tmh.time(0,random.randint(30,50))
                flight.toTime=flight.fromTime+flight.fltTime
                curTime=flight.toTime
                temp.append(flight)
                
            self.flights=temp
            
        #crew assignment: partition the flights of an aircraft, for each crew team
        #may consider a crew team operating two or more aircrafts in future
        self.crews=[]
        total=len(self.flights)
        Nsplit=random.randint(0,total//2)
        inds=sorted(random.sample(range(1,total),Nsplit))
        for i,j in zip([0]+inds,inds+[None]):
            target=self.flights[i:j]
            crew=Crew(cidh.cid,target)
            cidh.update()
            self.crews.append(crew)
            for f in target:
                f.crew=crew
        
    def getAircraftInfo(self):
        print("Tail:",self.tail)
        print("Capacity:",self.cpc)
        print(self.src,"-->",self.dest)
        print("----------")
                
class Flight:
    def __init__(self,fid,fromAp,toAp,aircraft):
        self.fid=fid
        self.fromAp=fromAp
        self.toAp=toAp
        self.dist=aph.getDistance(fromAp,toAp)
        self.aircraft=aircraft
        self.crew=None
        
        #Estimate flight time by distance according to average speed 800km/h
        #cruise time is assumed to be 30 min shorter than flight time
        #may consider variable speed in future
        self.fltTime=tmh.time(round(self.dist/800,1),0)
        self.crsTime=self.fltTime-tmh.time(0,30)
        self.fromTime=None
        self.toTime=None
        
        #Assume the occupancy rate is over 60%
        self.pax=round(self.aircraft.cpc*random.uniform(0.6,1))
        
    def getFlightInfo(self):
        print(self.fid)
        print(self.fromAp,"-->",self.toAp)
        print(self.fromTime)
        print(self.toTime)
        print("Pax:",self.pax)
        print("----------")
        

class Crew:
    def __init__(self,cid,flights):
        self.cid=cid
        self.flights=flights
        #crew source airport is assumed to be the source of first flight
        self.src=flights[0].fromAp
        #crew destination airport is assumed to be the destination of last flight
        self.dest=flights[-1].toAp
        self.fids=[f.fid for f in flights]

    def getCrewInfo(self):
        print(self.cid)
        print(self.src,"-->",self.dest)
        print("Operate flights:",self.fids)
        print("----------")


#init helper
random.seed(0)
aph=AirportHelper(20,1) # MAXAPT;seed 
tmh=TimeHelper((5,0),(24,0)) # time window
cidh=CidHelper()

#init aircrafts
aircrafts=[Aircraft("T%d"%i) for i in range(5)] # operating aircrafts 
flights,crews=[],[]
for air in aircrafts:
    flights+=air.flights
    crews+=air.crews
    
#generate dataframe
labels=["Tail","Flight_ID","Crew_ID","From","To","SDT","SAT","Flight_time","Cruise_time","Distance(km)","Capacity","Pax"]
res=np.array([[f.aircraft.tail,f.fid,f.crew.cid,f.fromAp,f.toAp,tmh.tostring(f.fromTime),tmh.tostring(f.toTime),tmh.tostring(f.fltTime),tmh.tostring(f.crsTime),f.dist,f.aircraft.cpc,f.pax] for f in flights])
pd.DataFrame({labels[i]:res[:,i] for i in range(len(labels))}).to_csv("Schedule_ACF5.csv",index=None)

#visualize entities(aircrafts/flights/crews)
#for c in crews:
#    c.getCrewInfo()


