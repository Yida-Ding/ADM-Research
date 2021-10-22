import random
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import haversine
import json
import os
import cplex
import itertools

from NetworkVisualizer import Visualizer

class Scenario:
    def __init__(self,direname,scname):
        self.direname=direname
        self.scname=scname
        with open("Datasets/"+direname+"/Config.json", "r") as outfile:
            self.config=json.load(outfile)
        with open("Scenarios/"+scname+"/DisruptionScenario.json", "r") as outfile:
            self.disruption=json.load(outfile)
        self.dfitinerary=pd.read_csv("Datasets/"+direname+"/Itinerary.csv",na_filter=None)
        self.dfschedule=pd.read_csv("Datasets/"+direname+"/Schedule.csv",na_filter=None)
        
        #FlightDepartureDelay
        for flight,delayTime in self.disruption.get("FlightDepartureDelay",[]):
            self.dfschedule.loc[self.dfschedule['Flight']==flight,["SDT"]]+=delayTime
            
        #DelayedReadyTime
        self.entity2delayTime={name:time for (name,time) in self.disruption.get("DelayedReadyTime",[])}             
        
        #Create FNodes
        self.FNodes=[]
        self.flight2dict={dic['Flight']:dic for dic in self.dfschedule.to_dict(orient='record')}
        self.type2mincontime={"ACF":self.config["ACMINCONTIME"],"CRW":self.config["CREWMINCONTIME"],"PAX":self.config["PAXMINCONTIME"]}
        orig2FNode,dest2FNode=defaultdict(list),defaultdict(list)
        for flight in self.dfschedule["Flight"].tolist():
            node=Node(self,ntype="FNode",name=flight)
            self.FNodes.append(node)
            orig2FNode[node.Ori].append(node)
            dest2FNode[node.Des].append(node)
            
        #AirportClosure
        cancelFlights=set()
        for ap,startTime,endTime in self.disruption.get("AirportClosure",[]):
            for node in orig2FNode.get(ap,[]):
                if node.SDT>startTime and node.LDT<endTime:
                    cancelFlights.add(node.name)
                elif node.SDT<startTime and node.LDT>startTime:
                    node.LDT=startTime
                elif node.SDT<endTime and node.LDT>endTime:
                    node.SDT=endTime
            for node in dest2FNode.get(ap,[]):
                if node.EAT>startTime and node.LAT<endTime:
                    cancelFlights.add(node.name)
                elif node.EAT<startTime and node.LAT>startTime:
                    node.LAT=startTime
                elif node.EAT<endTime and node.LAT>endTime:
                    node.EAT=endTime
        
        #FlightCancellation
        cancelFlights=cancelFlights|set(self.disruption.get("FlightCancellation",[]))
        self.FNodes=[node for node in self.FNodes if node.name not in cancelFlights]
        self.dfschedule.drop(self.dfschedule[self.dfschedule.Flight.isin(cancelFlights)].index,inplace=True)
        for index,row in self.dfitinerary.iterrows():
            if set(row['Flight_legs'].split('-'))&cancelFlights!=set():
                self.dfitinerary.drop(index,inplace=True)
        
        self.name2FNode={node.name:node for node in self.FNodes}
        self.tail2flights={tail:df_cur.sort_values(by='SDT')["Flight"].tolist() for tail,df_cur in self.dfschedule.groupby("Tail")}
        self.crew2flights={crew:df_cur.sort_values(by='SDT')["Flight"].tolist() for crew,df_cur in self.dfschedule.groupby("Crew")}
        self.itin2flights={row.Itinerary:row.Flight_legs.split('-') for row in self.dfitinerary.itertuples()}
        self.pax2flights={row.Itinerary+"P%02d"%i:row.Flight_legs.split('-') for row in self.dfitinerary.itertuples() for i in range(row.Pax)}
        
        self.type2flightdict={"ACF":self.tail2flights,"CRW":self.crew2flights,"PAX":self.pax2flights}
        self.itin2pax={row.Itinerary:row.Pax for row in self.dfitinerary.itertuples()}
        self.itin2destination={itin:self.name2FNode[self.itin2flights[itin][-1]].Des for itin in self.itin2flights.keys()}
        self.tail2capacity={tail:df_cur["Capacity"].tolist()[0] for tail,df_cur in self.dfschedule.groupby("Tail")}
        self.checker=ArcFeasibilityChecker(self)
        
        self.problem=cplex.Cplex()
        self.problem.objective.set_sense(self.problem.objective.sense.minimize)
        
        
class Node:
    def __init__(self,S,ntype="FNode",name=None,info=None):
        self.ntype=ntype
        self.name=name
        if ntype=="FNode":
            self.Ori=S.flight2dict[name]["From"]
            self.Des=S.flight2dict[name]["To"]
            self.SDT=S.flight2dict[name]["SDT"]
            self.SAT=S.flight2dict[name]["SAT"]
            self.SFT=S.flight2dict[name]["Flight_time"]
            self.LDT=self.SDT+S.config["MAXHOLDTIME"]
            self.LAT=self.SAT+S.config["MAXHOLDTIME"]
            self.ScheCrsTime=S.flight2dict[name]["Cruise_time"]
            self.CrsTimeComp=int(self.ScheCrsTime*S.config["CRSTIMECOMPPCT"])
            self.EDT=self.SDT
            self.EAT=self.SDT+self.SFT-self.CrsTimeComp
            self.CT=min(S.type2mincontime.values())
            self.CrsStageDistance=S.flight2dict[name]["Distance"]*S.config["CRUISESTAGEDISTPCT"]
            self.ScheCrsSpeed=self.CrsStageDistance/S.flight2dict[name]["Cruise_time"]
            self.MaxCrsSpeed=self.CrsStageDistance/(S.flight2dict[name]["Cruise_time"]-self.CrsTimeComp)
            self.Demand=0
            
        elif ntype=="SNode" or ntype=="TNode":
            self.Ori,self.Des,self.EAT,self.LDT,self.CT,self.Demand=info
            self.SDT=self.SAT=self.LAT=self.EDT=self.CrsTimeComp=self.SFT=None


class Entity:
    def __init__(self,S,name,etype):
        self.S=S
        self.name=name
        self.etype=etype
        self.F=S.type2flightdict[etype][name]
        self.CT=S.type2mincontime[etype]
        self.Ori=S.flight2dict[self.F[0]]["From"]
        self.Des=S.flight2dict[self.F[-1]]["To"]
        self.EDT=S.config["STARTTIME"]+self.S.entity2delayTime.get(name,0)
        self.LAT=S.config["ENDTIME"]
        self.SNode=Node(S,"SNode","S-"+name,(None,self.Ori,self.EDT,None,0,-1))
        self.TNode=Node(S,"TNode","T-"+name,(self.Des,None,None,self.LAT,0,1))
        self.name2Node=self.S.name2FNode.copy()
        self.name2Node.update({"S-"+name:self.SNode,"T-"+name:self.TNode})
        self.string="Type: %s | ID: %s | OD: %s --> %s"%(etype,name,self.Ori,self.Des)
        self.flightPath=[]
        self.PNGA()
        self.graph=nx.DiGraph(self.ArcSet)
        
    def PNGA(self):
        self.NodeSet,self.ArcSet=set(),set()
        tempPath=[self.SNode]
        self.generatePath(tempPath)        
        
    def generatePath(self,tempPath):
        lastNode=tempPath[-1]
        nextNodes=[node for node in self.S.FNodes if self.S.checker.checkConnectionArc(lastNode,node,tempPath)]
        for nextNode in nextNodes:
            tempPathc=tempPath.copy()+[nextNode]
            if nextNode in self.NodeSet:
                self.insert(tempPathc)
            else:
                if nextNode.Des==self.Des:
                    self.insert(tempPathc+[self.TNode])
                self.generatePath(tempPathc)
                    
    def insert(self,tempPath):
        namePath=[node.name for node in tempPath]
        self.NodeSet=self.NodeSet|set(namePath)
        for i in range(len(tempPath)-1):
            self.ArcSet=self.ArcSet|{(tempPath[i].name,tempPath[i+1].name)}
        self.flightPath+=[namePath]
        
    def plotPathOnBaseMap(self,path,ax,title):
        vis=Visualizer(self.S.direname)
        flights=[(S.flight2dict[flight]["From"],S.flight2dict[flight]["To"],flight) for flight in path[1:-1]]
        vis.plotTrajectoryOnBasemap(flights,ax)
        ax.set_title(title)        
        
    
    
class ArcFeasibilityChecker:
    def __init__(self,S):
        self.S=S
    
    def checkConnectionArc(self,node1,node2,tempPath):
        if node2 in tempPath:
            return False
        elif (node1.ntype,node2.ntype) in [("SNode","FNode"),("FNode","TNode")] and node1.Des==node2.Ori and node2.LDT>=node1.EAT:
            return True
        elif (node1.ntype,node2.ntype)==("FNode","FNode") and node1.Des==node2.Ori and node2.LDT>=node1.EAT+node1.CT:
            return True            
        return False
        