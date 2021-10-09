import random
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import haversine
import json
import os

from NetworkVisualizer import Visualizer
from util import ArcFeasibilityChecker,CplexHelper,DisruptionHelper


class Scenario:
    def __init__(self,direname,scname):
        self.direname=direname
        self.scname=scname
        with open("Datasets/"+direname+"/Config.json", "r") as outfile:
            self.config=json.load(outfile)
        with open("Scenarios/"+scname+"/DisruptionScenario.json", "r") as outfile:
            self.disrution=json.load(outfile)
        
        drpHelper=DisruptionHelper(self.disrution)
        dfschedule=pd.read_csv("Datasets/"+direname+"/Schedule.csv",na_filter=None)
        self.dfschedule=drpHelper.processDisruptionType12(dfschedule)
        self.dfitinerary=pd.read_csv("Datasets/"+direname+"/Itinerary.csv",na_filter=None)
        self.dfduration=pd.read_csv("Datasets/"+direname+"/Duration.csv",na_filter=None)
        
        self.entityName2delayTime=drpHelper.processDisruptionType3()
        self.flight2dict={dic['Flight']:dic for dic in self.dfschedule.to_dict(orient='record')}
        self.tail2flights={tail:df_cur.sort_values(by='SDT')["Flight"].tolist() for tail,df_cur in self.dfschedule.groupby("Tail")}
        self.crew2flights={crew:df_cur.sort_values(by='SDT')["Flight"].tolist() for crew,df_cur in self.dfschedule.groupby("Crew")}
        self.itin2flights={row.Itinerary:row.Flight_legs.split('-') for row in self.dfitinerary.itertuples()}
        self.type2flightdict={"ACF":self.tail2flights,"CRW":self.crew2flights,"PAX":self.itin2flights}
        self.itin2pax={row.Itinerary:row.Pax for row in self.dfitinerary.itertuples()}
        self.type2mincontime={"ACF":self.config["ACMINCONTIME"],"CRW":self.config["CREWMINCONTIME"],"PAX":self.config["PAXMINCONTIME"]}
        self.appair2duration={(row.From,row.To):row.Duration for row in self.dfduration.itertuples()}
        
        self.FNodes=[Node(self,ntype="FNode",name=flight) for flight in self.flight2dict.keys()]
        

class Entity:
    def __init__(self,S,name,etype,checker):
        self.S=S
        self.F=S.type2flightdict[etype][name]
        self.CT=S.type2mincontime[etype]
        self.Ori=S.flight2dict[self.F[0]]["From"]
        self.Des=S.flight2dict[self.F[-1]]["To"]
        self.EDT=S.config["STARTTIME"]+self.S.entityName2delayTime.get(name,0)
        self.LAT=S.config["ENDTIME"]
        self.SNode=Node(S,"SNode","S-"+name,(None,self.Ori,self.EDT,None,0,-1))
        self.TNode=Node(S,"TNode","T-"+name,(self.Des,None,None,self.LAT,0,1))
        self.string="Type: %s | ID: %s | OD: %s --> %s"%(etype,name,self.Ori,self.Des)
        self.checker=checker
        self.namePaths=[]
        
    def PNGA(self):
        self.NodeSet,self.ArcSet=set(),set()
        tempPath=[self.SNode]
        self.generatePath(tempPath)
        
    def generatePath(self,tempPath):
        lastNode=tempPath[-1]
        if lastNode.Des==self.Des:
            return
        else:
            nextNodes=[node for node in self.S.FNodes if self.checker.checkConnectionArc(lastNode,node)]
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
        self.namePaths+=[namePath]
        
    def plotPathOnBaseMap(self,path,ax,title):
        vis=Visualizer(self.S.direname)
        visflights=[(S.flight2dict[flight]["From"],S.flight2dict[flight]["To"]) for flight in path[1:-1]]
        vis.plotTrajectoryOnBasemap(visflights,ax)
        ax.set_title(title)
        
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
            self.CrsTimeComp=int(S.flight2dict[name]["Cruise_time"]*S.config["CRSTIMECOMPPCT"]) #TODO: Consider the compression limit by different types of aircraft in the future, and modify EAT
            self.EDT=self.SDT
            self.EAT=self.SDT+self.SFT-self.CrsTimeComp
            self.CT=min(S.type2mincontime.values())
            self.Demand=0
            
        elif ntype=="SNode" or ntype=="TNode":
            self.Ori,self.Des,self.EAT,self.LDT,self.CT,self.Demand=info
            self.SDT=self.SAT=self.LAT=self.EDT=self.CrsTimeComp=self.SFT=None
        #TODO: generate data for schedule maintenance and construct must-nodes 
    
random.seed(0)
S=Scenario("ACF5","ACF5-SC1")
checker=ArcFeasibilityChecker(S)



entity=Entity(S,"T00","ACF",checker)
entity.PNGA()

fig,axes=plt.subplots(3,3,figsize=(15,10),dpi=100)
selpaths=random.sample(entity.namePaths,9)
for i,ax in enumerate(axes.flat):
    entity.plotPathOnBaseMap(selpaths[i],ax,"Path%d"%i)
plt.suptitle(entity.string,fontsize=20)


    




