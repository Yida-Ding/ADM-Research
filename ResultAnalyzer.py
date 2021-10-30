import pandas as pd
import networkx as nx
import json
from collections import defaultdict
from lxml import etree

from NetworkGenerator import Scenario

class Analyzer:
    def __init__(self,S,dataset,scenario):
        self.S=S
        self.scenario=scenario
        self.dataset=dataset
        with open("Results/"+scenario+"/Variables.json","r") as outfile:
            self.variable2value=json.load(outfile)
        
        tree=etree.parse("Results/"+scenario+"/ModelSolution.sol")
        for x in tree.xpath("//header"):
            self.objective=float(x.attrib["objectiveValue"])
            
        self.tail2graph={tail:nx.DiGraph() for tail in S.tail2flights}
        self.flight2crew,self.flight2deptime,self.flight2arrtime,self.flight2crstime,self.flight2crstimecomp={},{},{},{},{}
        self.flight2numpax,self.flight2paxname=defaultdict(int),defaultdict(list)
        self.pax2delay,self.flight2fuel={},{}
        self.cancelFlights=[]
        
        for variable,value in self.variable2value.items():
            terms=variable.split('_')            
            if terms[0]=="x" and abs(value-1)<1e-3:
                if terms[1][0]=="T":
                    self.tail2graph[terms[1]].add_edge(terms[2],terms[3])
                elif terms[1][0]=="C":
                    self.flight2crew[terms[3]]=terms[1]
                elif terms[1][0]=="I":
                    if terms[2][:2]!="S-":      #except start node
                        self.flight2numpax[terms[2]]+=1
                        self.flight2paxname[terms[2]].append(terms[1])        
                    
            elif terms[0]=='z' and abs(value-1)<1e-3:
                self.cancelFlights.append(terms[1])
            elif terms[0]=='dt':    
                self.flight2deptime[terms[1]]=value
            elif terms[0]=='at':
                self.flight2arrtime[terms[1]]=value
            elif terms[0]=='crt':
                self.flight2crstime[(terms[1],terms[2])]=value
            elif terms[0]=='deltat' and value>1:
                self.flight2crstimecomp[terms[1]]=int(value)
            elif terms[0]=='delay' and value>1:
                self.pax2delay[terms[1]]=int(value)
            elif terms[0]=='fc' and value>1:
                self.flight2fuel[terms[1]]=value
        self.generateRecoveryPlan()
        
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
                resd["Pax"].append(self.flight2numpax[flight])
                resd["Timestring"].append(self.getTimeString(self.flight2deptime[flight])+" -> "+self.getTimeString(self.flight2arrtime[flight]))
        
        self.dfrecovery=pd.DataFrame(resd).sort_values(by="Flight").reset_index()
        self.dfrecovery.to_csv("Results/"+self.scenario+"/Recovery.csv",index=False)
        self.flight2recdict={dic['Flight']:dic for dic in self.dfrecovery.to_dict(orient='record')}
        
    def getTimeString(self,seconds):
        days,remainder=divmod(seconds,24*3600)
        hours,remainder=divmod(remainder,3600)
        minutes,seconds=divmod(remainder,60)
        s='{:02}:{:02}'.format(int(hours),int(minutes))
        if days>0:
            s+=" (+%d)"%days
        return s
    
    def displayScheduleAndRecovery(self):
        attributes=["Tail","Flight","Crew","From","To","Pax","Timestring"]
        print("*Original Schedule*")
        print(self.S.dfschedule[attributes])
        print('-'*60)
#        print("*Disrupted Schedule*")
#        print(self.S.dfdrpschedule[attributes])
#        print('-'*60)
        print("*Recovery Plan*")
        print(self.dfrecovery[attributes])
        
    def getReroutingActions(self):
        crewMessages,tailMessages,paxMessages=[],[],[]
        for flight,recdict in self.flight2recdict.items():
            scheCrew,recCrew=self.S.flight2dict[flight]["Crew"],recdict["Crew"]
            scheTail,recTail=self.S.flight2dict[flight]["Tail"],recdict["Tail"]
            if scheCrew!=recCrew:
                crewMessages.append("\t%s: %s -> %s"%(flight,scheCrew,recCrew))
            if scheTail!=recTail:
                tailMessages.append("\t%s: %s -> %s"%(flight,scheTail,recTail))
        
        paxRerouteCount=0
        for flight,recPax in self.flight2paxname.items():
            schePax,recPax=set(self.S.flight2paxnames[flight]),set(recPax)            
            leave,come=schePax-(schePax&recPax),recPax-(schePax&recPax)
            paxRerouteCount+=len(leave)+len(come)
            leavedict,comedict=defaultdict(int),defaultdict(int)
            for pax in leave:
                leavedict[self.S.paxname2itin[pax]]+=1
            for pax in come:
                comedict[self.S.paxname2itin[pax]]+=1
            if len(come)!=0:
                paxMessages.append("\t%s: (+) "%flight+json.dumps(comedict))
            if len(leave)!=0:
                paxMessages.append("\t%s: (-) "%flight+json.dumps(leavedict))
            
        print('-'*60)
        print("Tail Swap (count=%d):\n"%(len(tailMessages))+'\n'.join(tailMessages))
        print('-'*60)
        print("Crew Swap (count=%d):\n"%(len(crewMessages))+'\n'.join(crewMessages))
        print('-'*60)
        print("Passenger Rerouting (count=%d):\n"%paxRerouteCount+'\n'.join(paxMessages))
        print('-'*60)

    
    def getCostTerms(self):
        extraFuelCost=(sum(self.flight2fuel.values())-sum([node.ScheFuelConsump for node in self.S.FNodes]))*self.S.config["FUELCOSTPERKG"]
        delayCost=sum(self.pax2delay.values())*self.S.config["DELAYCOST"]
        cancelCost=len(self.cancelFlights)*self.S.config["FLIGHTCANCELCOST"]
        followGain=self.objective-(delayCost+extraFuelCost+cancelCost)
        
        resd=defaultdict(list)       
        resd["Cost_term"]=["extra_fuel_cost","delay_cost","cancel_cost","follow_cost","objective"]
        resd["Value"]=[extraFuelCost,delayCost,cancelCost,followGain,self.objective]
        resd["Percentage"]=["{0:.0%}".format(v/self.objective) for v in resd["Value"]]
        dfcost=pd.DataFrame(resd)
        dfcost.to_csv("Results/"+self.scenario+"/Cost.csv",index=False)
        print(dfcost)
        
        

dire,scen="ACF2","ACF2-SC1"
analyzer=Analyzer(Scenario(dire,scen),dire,scen)
analyzer.displayScheduleAndRecovery()
analyzer.getReroutingActions()
analyzer.getCostTerms()









        
