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

class CrewAssignHelper:
    def __init__(self,acfs):
        self.acfs=acfs
        #use multidigraph to store flights
        self.G=nx.MultiDiGraph()
        #input flights to the graph as edges
        for acf in self.acfs:
            for flt in acf.flights:
                self.G.add_edge(flt.fromAp,flt.toAp,acf=acf.tail,obj=flt,SDT=flt.fromTime,SAT=flt.toTime,fltTime=flt.fltTime,seen=0)
        
        #sort edges by SDT so that crew will always start with an earlier flight
        edges=sorted(self.G.edges(data=True),key=lambda t: t[2].get('SDT'))
        #terminate condition: all the edges are covered by crew
        while len(edges)!=0:
            #pop out the currently earliest flight to start the crew path
            cur=edges.pop(0)
            path=[cur]
            workTime=tmh.time(0,0)
            
            while cur!=None:
                #find a feasible edge that is connected to cur edge
                select=None
                for temp in self.G.out_edges(cur[1],data=True):
                    #conditions: 1)not seen yet; 2)accept 30 min minimum connection time; 3)within 10 hours working time
                    if temp[2]['seen']==0 and temp[2]['SDT']>=cur[2]['SAT']+tmh.time(0,30) and workTime+temp[2]['fltTime']<=tmh.time(10,0):
                        select=temp
                        #if there is a chance to change aircraft, then possibly catch it
                        if temp[2]['acf']!=cur[2]['acf'] and random.uniform(0,1)<0.8:
                            break
                
                #update working time and path by the selected edge
                if select!=None:
                    workTime+=select[2]['fltTime']
                    path.append(select)    
                
                #updation for the inner while loop
                cur=select
            
            #crew assigned to each target flight
            for target in path:
                target[2]['obj'].crew=cidh.cid
                target[2]['seen']=1
                
            #update the global variable crew id
            cidh.update()
            #remove seen edges
            edges=[edge for edge in edges if edge[2]['seen']==0]


class Aircraft:
    def __init__(self,tail):
        self.tail=tail
        #assume three aircraft capacity type, small/medium/large 
        self.cpc=random.choice([180,210,240])
        
        #randomly select src and dest airport from the apGraph for the aircraft
        self.src,self.dest=random.sample(aph.airports,2)
        self.flights=[]
        curTime=tmh.winEnd
        
        #if one of the flight in generated path is invalid regarding duration, pathFlag=False
        pathFlag=True
        #the first condition is used to ensure a feasible journey taken by the aircraft
        while curTime>=tmh.winEnd or pathFlag==False:
            #create random simple path of random length within range 4~6
            #may consider repeated nodes for the path in future
            self.path=[]
            generator=nx.all_simple_paths(aph.apGraph,self.src,self.dest)
            while len(self.path)<4 or len(self.path)>6:
                self.path=next(generator)
                
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
                pathFlag=flight.flag
                
            self.flights=temp
            
                
class Flight:
    def __init__(self,fid,fromAp,toAp,aircraft):
        self.fid=fid
        self.fromAp=fromAp
        self.toAp=toAp
        self.dist=aph.getDistance(fromAp,toAp)
        self.aircraft=aircraft
        self.crew=None
        self.flag=True
        
        #Estimate flight time by distance according to average speed 800km/h
        #cruise time is assumed to be 30 min shorter than flight time
        #may consider variable speed in future
        self.fltTime=tmh.time(round(self.dist/800,1),0)
        #assume flight less than 30 min be an invalid flight
        if self.fltTime<=tmh.time(0,30):
            self.flag=False
            
        self.crsTime=self.fltTime-tmh.time(0,30)
        self.fromTime=None
        self.toTime=None
        
        #Assume the occupancy rate is over 60%
        self.pax=round(self.aircraft.cpc*random.uniform(0.6,1))
        
                
                    
#init helper
random.seed(6)
aph=AirportHelper(6,1) # maximum nb of airports in the connected graph ; random seed 
tmh=TimeHelper((5,0),(24,0)) # time window (hour, minute)
cidh=CidHelper()

#init aircrafts
Napt=3 # nb of operating aircrafts     
aircrafts=[Aircraft("T%d"%i) for i in range(Napt)]

cah=CrewAssignHelper(aircrafts)
flights=[]
for a in aircrafts:
    flights+=a.flights
    
#generate dataframe
labels=["Tail","Flight_ID","Crew_ID","From","To","SDT","SAT","Flight_time","Cruise_time","Distance(km)","Capacity","Pax"]
res=np.array([[f.aircraft.tail,f.fid,f.crew,f.fromAp,f.toAp,tmh.tostring(f.fromTime),tmh.tostring(f.toTime),tmh.tostring(f.fltTime),tmh.tostring(f.crsTime),f.dist,f.aircraft.cpc,f.pax] for f in flights])
df=pd.DataFrame({labels[i]:res[:,i] for i in range(len(labels))})
df.to_csv("Schedule_ACF%d.csv"%Napt,index=None)
print(df)









