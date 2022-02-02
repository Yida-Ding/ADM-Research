import pandas as pd
import networkx as nx
import json
import cplex
from collections import defaultdict
from lxml import etree
from NetworkGenerator import Scenario
import matplotlib.pyplot as plt

class Analyzer:
    def __init__(self,dataset,scenario,modeid):
        self.scenario=scenario
        self.dataset=dataset
        self.modeid=modeid
        with open("Results/%s/Mode.json"%(self.scenario),"r") as outfile:
            self.mode=json.load(outfile)                        
        self.S=Scenario(dataset,scenario,self.mode["PAXTYPE"])
    
    def parseOutputData(self):
        with open("Results/%s/Variables.json"%(self.scenario),"r") as outfile:
            self.variable2value=json.load(outfile)            
        with open("Results/%s/Coefficients.json"%(self.scenario),"r") as outfile:
            self.variable2coeff=json.load(outfile)            
        tree=etree.parse("Results/%s/ModelSolution.sol"%(self.scenario))
        for x in tree.xpath("//header"):
            self.objective=float(x.attrib["objectiveValue"])
        self.tail2graph={tail:nx.DiGraph() for tail in self.S.tail2flights}
        self.flight2crew,self.flight2deptime,self.flight2arrtime,self.flight2crstime,self.flight2crstimecomp={},{},{},{},{}
        self.flight2numpax,self.flight2paxname,self.flight2itinNum=defaultdict(int),defaultdict(list),defaultdict(list)
        self.flight2delay,self.variable2fuel,self.scheflow2value,self.cancel2value={},{},{},{}
        
        for variable,value in self.variable2value.items():
            terms=variable.split('_')
            if terms[0]=="x" and round(value)!=0:
                if terms[1][0]=="T":
                    self.tail2graph[terms[1]].add_edge(terms[2],terms[3])
                elif terms[1][0]=="C":
                    self.flight2crew[terms[3]]=terms[1]
                elif terms[1][0]=="I":
                    if terms[2][:2]!="S-":      #except start node
                        self.flight2numpax[terms[2]]+=round(value)
                        self.flight2paxname[terms[2]].append(terms[1])
                        self.flight2itinNum[terms[2]].append((terms[1],round(value)))
                        
            if terms[0]=="x" and self.variable2coeff[variable]!=0.0:
                self.scheflow2value[variable]=value              
                print(variable)
            elif terms[0]=='z' and round(value)!=0:
                self.cancel2value[variable]=value
            elif terms[0]=='dt':    
                self.flight2deptime[terms[1]]=value
            elif terms[0]=='at':
                self.flight2arrtime[terms[1]]=value
            elif terms[0]=='crt':
                self.flight2crstime[(terms[1],terms[2])]=value
            elif terms[0]=='deltat' and round(value)!=0:
                self.flight2crstimecomp[terms[1]]=round(value)
            elif terms[0]=='delay'and round(value)!=0:
                self.flight2delay[terms[1]]=round(value)
            elif terms[0]=='fc' and round(value)!=0:
                self.variable2fuel[variable]=value
               
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
                arrdelay=round(resd["RAT"][-1]-self.S.flight2scheduleAT[flight])
                resd["Delay"].append(self.getTimeString(arrdelay) if arrdelay>0 else '')
                if flight in self.S.disruptedFlights:
                    resd["DelayType"].append(1)
                elif arrdelay>0:
                    resd["DelayType"].append(2)
                else:
                    resd["DelayType"].append(0)
        
        self.dfrecovery=pd.DataFrame(resd).reset_index()
        self.dfrecovery.to_csv("Results/%s/Recovery.csv"%(self.scenario),index=False)
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
        print("Original Schedule:")
        print(self.S.dfschedule[attributes])
        print('-'*60)
        print("Recovery Plan:")
        print(self.dfrecovery[attributes+["Delay"]])
        
    def getReroutingActions(self):
        crewMessages,tailMessages,paxMessages=[],[],[]
        for flight,recdict in self.flight2recdict.items():
            scheCrew,recCrew=self.S.flight2dict[flight]["Crew"],recdict["Crew"]
            scheTail,recTail=self.S.flight2dict[flight]["Tail"],recdict["Tail"]
            if scheCrew!=recCrew:
                crewMessages.append("\t%s: %s -> %s"%(flight,scheCrew,recCrew))
            if scheTail!=recTail:
                tailMessages.append("\t%s: %s -> %s"%(flight,scheTail,recTail))
        
        if self.mode["PAXTYPE"]=="PAX":
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
                    
        elif self.mode["PAXTYPE"]=="ITIN":
            paxRerouteCount=0
            for flight in self.flight2itinNum:
                leavedict=defaultdict(int)
                schedict,recdict=dict(self.S.flight2itinNum[flight]),dict(self.flight2itinNum[flight])
                for itin in set(schedict.keys())|set(recdict.keys()):
                    leavedict[itin]+=schedict.get(itin,0)-recdict.get(itin,0)
                leave,come={k:v for k,v in leavedict.items() if v>0},{k:-v for k,v in leavedict.items() if v<0}
                paxRerouteCount+=len(leave)+len(come)
                if len(come)!=0:
                    paxMessages.append("\t%s: (+) "%flight+json.dumps(come))
                if len(leave)!=0:
                    paxMessages.append("\t%s: (-) "%flight+json.dumps(leave))
                
        print('-'*60)
        print("Tail Swap (count=%d):\n"%(len(tailMessages))+'\n'.join(tailMessages))
        print('-'*60)
        print("Crew Swap (count=%d):\n"%(len(crewMessages))+'\n'.join(crewMessages))
        print('-'*60)
        print("Passenger Rerouting (count=%d):\n"%paxRerouteCount+'\n'.join(paxMessages))
    
    def getCostTerms(self):
        extraFuelCost=sum([self.variable2coeff[variable]*fuel for variable,fuel in self.variable2fuel.items()])+self.variable2value["offset"]
        cancelCost=sum([self.variable2coeff[variable]*value for variable,value in self.cancel2value.items()])
        followGain=sum([self.variable2coeff[variable]*value for variable,value in self.scheflow2value.items()])
        if self.mode["DELAYTYPE"]=="approx":
            delayCost=sum([self.flight2delay.get(node1.name,0)*self.variable2coeff["delay_"+node1.name] for node1 in self.S.FNodes])
        else: # actual
            delayCost=sum(self.flight2delay.values())*self.S.config["DELAYCOST"]
        
        resd=defaultdict(list)
        values=[extraFuelCost,delayCost,cancelCost,followGain,self.objective]
        resd["Cost"]=["extra_fuel_cost","delay_cost","cancel_cost","follow_gain","objective"]
        resd["Value"]=["%d"%value for value in values]
        resd["Pct"]=["{0:.0%}".format(v/self.objective) for v in values[:4]]+["\\"]*1
        dfcost=pd.DataFrame(resd)
        dfcost.to_csv("Results/%s/Cost.csv"%(self.scenario),index=False)
        print('-'*60)
        print(dfcost)
        
    def getRunTimeAndGap(self):
        runtime=self.variable2value["runtime"]
        gap=self.variable2value["optimalityGap"]
        return runtime,gap
    
def mainResultAnalyzer(dataset,scenario,mode="Mode1"):
    analyzer=Analyzer(dataset,scenario,mode)
    analyzer.parseOutputData()
    analyzer.generateRecoveryPlan()
    analyzer.displayScheduleAndRecovery()
    analyzer.getReroutingActions()
    analyzer.getCostTerms()

    
analyzer=Analyzer("ACF4","ACF4-SC1","Mode1")
analyzer.parseOutputData() 
    
