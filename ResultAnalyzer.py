import pandas as pd
import xml.etree.ElementTree as ET
import networkx as nx
import json
from collections import defaultdict

from NetworkGenerator import Scenario

class Analyzer:
    def __init__(self,S,dataset,scenario):
        with open("Results/"+scenario+"/Variables.json","r") as outfile:
            self.variable2value=json.load(outfile)
        
        #create digraph for tail and crew team
        self.tail2graph={tail:nx.DiGraph() for tail in S.tail2flights}
        self.flight2crew={}
        self.flight2pax=defaultdict(int)
        self.flight2deptime={}
        self.flight2arrtime={}
        self.flight2crstime={}
        
        #process data
        for variable,value in self.variable2value.items():
            terms=variable.split('_')
            if value==1 and terms[0]=="x":
                if terms[1][0]=="T":
                    self.tail2graph[terms[1]].add_edge(terms[2],terms[3])
                elif terms[1][0]=="C":
                    self.flight2crew[terms[2]]=terms[1]
                else:
                    self.flight2pax[terms[2]]+=1
                    
            elif terms[0]=='dt':
                self.flight2deptime[terms[1]]=value
            elif terms[0]=='at':
                self.flight2arrtime[terms[1]]=value
            elif terms[0]=='crt':
                self.flight2crstime[terms[1]]=value                
                            
        #iterate through flight
        resd=defaultdict(list)        
        for tail,tgraph in self.tail2graph.items():
            flights=list(nx.all_simple_paths(tgraph,source='S-%s'%tail,target='T-%s'%tail))[0][1:-1]
            for flight in flights:
                resd["Tail"].append(tail)
                resd["Flight"].append(flight)
                resd["Crew"].append(self.flight2crew[flight])
                resd["From"].append(S.flight2dict[flight]["From"])
                resd["To"].append(S.flight2dict[flight]["To"])
                resd["RDT"].append(self.flight2deptime[flight])
                resd["RAT"].append(self.flight2arrtime[flight])
                resd["Flight_time"].append(self.flight2arrtime[flight]-self.flight2deptime[flight])
                resd["Cruise_time"].append(self.flight2crstime[flight])
                resd["Distance"].append(S.flight2dict[flight]["Distance"])
                resd["Capacity"].append(S.tail2capacity[tail])
                resd["Pax"].append(self.flight2pax[flight])
                resd["Timestring"].append(self.getTimeString(self.flight2deptime[flight])+" -> "+self.getTimeString(self.flight2arrtime[flight]))
                
        df=pd.DataFrame(resd)
        df.to_csv("Results/"+scenario+"/Recovery.csv",index=False)
            
    def getTimeString(self,seconds):
        days,remainder=divmod(seconds,24*3600)
        hours,remainder=divmod(remainder,3600)
        minutes,seconds=divmod(remainder,60)
        s='{:02}:{:02}'.format(int(hours),int(minutes))
        if days>0:
            s+=" (+%d)"%days
        return s
               

dire,scen="ACF2","ACF2-SC0"
S=Scenario(dire,scen)
A=Analyzer(S,dire,scen)

        
