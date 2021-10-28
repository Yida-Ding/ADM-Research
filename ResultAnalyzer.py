import pandas as pd
import xml.etree.ElementTree as ET
import networkx as nx
import json
from collections import defaultdict
from collections import OrderedDict
from NetworkGenerator import Scenario

class Analyzer:
    def __init__(self,S,dataset,scenario):
        self.S=S
        self.scenario=scenario
        self.dataset=dataset
        with open("Results/"+scenario+"/Variables.json","r") as outfile:
            self.variable2value=json.load(outfile)
        self.tail2graph={tail:nx.DiGraph() for tail in S.tail2flights}
        self.flight2crew,self.flight2deptime,self.flight2arrtime,self.flight2crstime={},{},{},{}
        self.flight2pax=defaultdict(int)
        self.pax2delay=OrderedDict()
        
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
                self.flight2crstime[(terms[1],terms[2])]=value
            elif terms[0]=='delay':
                if value>1:
                    self.pax2delay[terms[1]]=int(value)
        
    
    def getTimeString(self,seconds):
        days,remainder=divmod(seconds,24*3600)
        hours,remainder=divmod(remainder,3600)
        minutes,seconds=divmod(remainder,60)
        s='{:02}:{:02}'.format(int(hours),int(minutes))
        return s
    
    def generateRecoveryPlan(self):
        resd=defaultdict(list)        
        for tail,tgraph in self.tail2graph.items():
            flights=list(nx.all_simple_paths(tgraph,source='S-%s'%tail,target='T-%s'%tail))[0][1:-1]
            for flight in flights:
                resd["Tail"].append(tail)
                resd["Flight"].append(flight)
                resd["Crew"].append(self.flight2crew[flight])
                resd["From"].append(self.S.flight2dict[flight]["From"])
                resd["To"].append(self.S.flight2dict[flight]["To"])
                resd["RDT"].append(self.flight2deptime[flight])
                resd["RAT"].append(self.flight2arrtime[flight])
                resd["Flight_time"].append(self.flight2arrtime[flight]-self.flight2deptime[flight])
                resd["Cruise_time"].append(self.flight2crstime[(flight,tail)])
                resd["Distance"].append(self.S.flight2dict[flight]["Distance"])
                resd["Capacity"].append(self.S.tail2capacity[tail])
                resd["Pax"].append(self.flight2pax[flight])
                resd["Timestring"].append(self.getTimeString(self.flight2deptime[flight])+" -> "+self.getTimeString(self.flight2arrtime[flight]))
        
        df=pd.DataFrame(resd).sort_values(by="Flight")
        df.to_csv("Results/"+self.scenario+"/Recovery.csv",index=False)
    
    def display(self):
        dfschedule=pd.read_csv("Datasets/"+self.dataset+"/Schedule.csv")
        dfrecovery=pd.read_csv("Results/"+self.scenario+"/Recovery.csv")
        dfschedule=dfschedule[["Tail","Flight","From","To","Pax","Timestring"]]
        dfrecovery=dfrecovery[["Tail","Flight","From","To","Pax","Timestring"]]
        print(dfschedule)
        print('------------')
        print(dfrecovery)
        

for i in [2,5]:
    dire,scen="ACF%d"%i,"ACF%d-SC%d"%(i,0)
    A=Analyzer(Scenario(dire,scen),dire,scen)
    A.generateRecoveryPlan()
    print(A.pax2delay)



#A.display()
















        
