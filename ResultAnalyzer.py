import pandas as pd
import networkx as nx
import json
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
        
    def getTimeString(self,seconds):
        days,remainder=divmod(seconds,24*3600)
        hours,remainder=divmod(remainder,3600)
        minutes,seconds=divmod(remainder,60)
        s='{:02}:{:02}'.format(int(hours),int(minutes))
        if days>0:
            s+=" (+%d)"%days
        return s

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
        self.pax2delay,self.variable2fuel,self.scheflow2value,self.cancel2value={},{},{},{}
        self.flight2pax,self.pname2flts=defaultdict(int),defaultdict(list)
        
        for variable,value in self.variable2value.items():
            terms=variable.split('_')
            if terms[0]=="x" and round(value)!=0:
                if terms[1][0]=="T":
                    self.tail2graph[terms[1]].add_edge(terms[2],terms[3])
                elif terms[1][0]=="C":
                    self.flight2crew[terms[3]]=terms[1]
                elif terms[1][0]=="I":
                    if terms[2][:2]!="S-":      # except start node
                        self.flight2pax[terms[2]]+=round(value)
                    self.pname2flts[terms[1]].append((terms[2],terms[3]))
                        
            if terms[0]=="x" and self.variable2coeff[variable]!=0.0:
                self.scheflow2value[variable]=value   
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
                self.pax2delay[terms[1]]=round(value)
            elif terms[0]=='fc' and round(value)!=0:
                self.variable2fuel[variable]=value
        
        
        self.fltleg2itin=self.S.fltlegs2itin.copy()
        self.recItin2flts={}; self.bothItin2pax=defaultdict(int)
        for pname,pairs in self.pname2flts.items():
            skdItin=pname.split('+')[0]
            if len(pairs)==2:
                fltleg=pairs[0][0] if pairs[0][1][:2]=='T-' else pairs[0][1]
            elif len(pairs)==3:
                for pair in pairs:
                    if pair[0][:2]!='S-' and pair[1][:2]!='T-':
                        fltleg=pair[0]+'-'+pair[1]
            elif len(pairs)==4:
                L=[pair for pair in pairs if pair[0][:2]!='S-' and pair[1][:2]!='T-']
                fltleg=L[0][0]+'-'+L[0][1]+'-'+L[1][1] if L[0][1]==L[1][0] else L[1][0]+'-'+L[0][0]+'-'+L[0][1]
        
            if fltleg in self.fltleg2itin.keys():
                recItin=self.fltleg2itin[fltleg]
            else:
                recItin="I%02d"%len(self.fltleg2itin)
                self.fltleg2itin[fltleg]=recItin
            self.recItin2flts[recItin]=fltleg
            self.bothItin2pax[(recItin,skdItin)]+=1
                
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
                arrdelay=round(resd["RAT"][-1]-self.S.flight2scheduleAT[flight])
                resd["Delay"].append(self.getTimeString(arrdelay) if arrdelay>0 else '')
                if flight in self.S.disruptedFlights:
                    resd["DelayType"].append(1)
                elif arrdelay>0:
                    resd["DelayType"].append(2)
                else:
                    resd["DelayType"].append(0)
        
        resdItin=defaultdict(list)
        for bothItin,pax in self.bothItin2pax.items():
            recItin,skdItin=bothItin
            recFlts=self.recItin2flts[recItin].split('-')
            resdItin["Rec_itin"].append(recItin)
            resdItin["Skd_itin"].append(skdItin)
            resdItin["Rec_flights"].append(self.recItin2flts[recItin])
            resdItin["Skd_flights"].append('-'.join(self.S.itin2flights[skdItin]))
            resdItin["Pax"].append(pax)
            resdItin["From"].append(self.S.itin2origin[skdItin])
            resdItin["To"].append(self.S.itin2destination[skdItin])
            resdItin["RDT"].append(self.flight2deptime[recFlts[0]])
            resdItin["RAT"].append(self.flight2arrtime[recFlts[-1]])
            delay=self.flight2arrtime[recFlts[-1]]-self.S.itin2skdtime[skdItin][1]
            if delay<0:
                delay=0
            resdItin["Arr_delay"].append(delay)
            resdItin["Cost"].append("%.1f"%(self.S.config["DELAYCOST"]*pax*delay))
            resdItin["Rec_string"].append(self.S.getTimeString(self.flight2deptime[recFlts[0]])+' -> '+self.S.getTimeString(self.flight2arrtime[recFlts[-1]]))
            
        self.dfrecovery=pd.DataFrame(resd).reset_index()
        self.dfrecovery.to_csv("Results/%s/RecoveryCPLEX.csv"%(self.scenario),index=False)
        self.dfitinerary=pd.DataFrame(resdItin).sort_values(by=["Rec_itin","Skd_itin"])
        self.dfitinerary.to_csv("Results/%s/ItineraryCPLEX.csv"%(self.scenario),index=False)
        self.flight2recdict={dic['Flight']:dic for dic in self.dfrecovery.to_dict(orient='records')}
    
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
                        
        print('-'*60)
        print("Tail Swap (count=%d):\n"%(len(tailMessages))+'\n'.join(tailMessages))
        print('-'*60)
        print("Crew Swap (count=%d):\n"%(len(crewMessages))+'\n'.join(crewMessages))
    
    def getCostTerms(self):
        extraFuelCost=sum([self.variable2coeff[variable]*fuel for variable,fuel in self.variable2fuel.items()])+self.variable2value["offset"]
        cancelCost=sum([self.variable2coeff[variable]*value for variable,value in self.cancel2value.items()])
        followGain=sum([self.variable2coeff[variable]*value for variable,value in self.scheflow2value.items()])
        if self.mode["DELAYTYPE"]=="approx":
            delayCost=sum([self.pax2delay.get(node1.name,0)*self.variable2coeff["delay_"+node1.name] for node1 in self.S.FNodes])
        else: # actual
            delayCost=sum(self.pax2delay.values())*self.S.config["DELAYCOST"]
        costDict={"DelayCost":delayCost,"ExtraFuelCost":extraFuelCost,"CancelCost":cancelCost,"FollowGain":followGain,"Objective":self.objective}
        with open("Results/%s/CostCPLEX.json"%(self.scenario), "w") as outfile:
            json.dump(costDict, outfile, indent = 4)
            
        print('-'*60)
        print(pd.DataFrame({term:["%.1f"%cost] for term,cost in costDict.items()}))
        
    def getRunTimeAndGap(self):
        runtime=self.variable2value["cplexTime"]
        gap=self.variable2value["optimalityGap"]
        return runtime,gap
    
    
class DatasetScenarioSummarizer:
    def __init__(self,dataset,scenario):
        self.dataset=dataset
        self.scenario=scenario
        #load from Dataset
        with open("Datasets/"+dataset+"/Config.json", "r") as outfile:
            self.config=json.load(outfile)
        self.dfschedule=pd.read_csv("Datasets/"+dataset+"/Schedule.csv",na_filter=None)
        self.dfitinerary=pd.read_csv("Datasets/"+dataset+"/Itinerary.csv",na_filter=None)
        self.flight2scheduleAT={row.Flight:row.SAT for row in self.dfschedule.itertuples()}
        self.flight2scheduleDT={row.Flight:row.SDT for row in self.dfschedule.itertuples()}
        self.airports=set(self.dfschedule["From"])|set(self.dfschedule["To"])
        
        #load from Scenario
        self.dfdrpschedule=pd.read_csv("Scenarios/"+scenario+"/DrpSchedule.csv",na_filter=None)
        self.disruptedFlights=self.dfdrpschedule[self.dfdrpschedule["is_disrupted"]==1]["Flight"].tolist()
        self.flight2drpAT={row.Flight:row.SAT for row in self.dfdrpschedule.itertuples()}
        self.flight2drpDT={row.Flight:row.SDT for row in self.dfdrpschedule.itertuples()}
    

    
def mainResultAnalyzer(dataset,scenario,mode="Mode1"):
    analyzer=Analyzer(dataset,scenario,mode)
    analyzer.parseOutputData()
    analyzer.generateRecoveryPlan()
    analyzer.displayScheduleAndRecovery()
    analyzer.getReroutingActions()
    analyzer.getRunTimeAndGap()
    analyzer.getCostTerms()
    
    
def getDatasetSummary(dataset):
    dfschedule=pd.read_csv("Datasets/"+dataset+"/Schedule.csv",na_filter=None)
    flights=dfschedule["Flight"].to_list()
    tails=set(list(dfschedule["Tail"]))
    crews=set(list(dfschedule["Crew"]))
    dfitinerary=pd.read_csv("Datasets/"+dataset+"/Itinerary.csv",na_filter=None)
    paxs=dfitinerary["Pax"].to_list()
    flight2scheduleAT={row.Flight:row.SAT for row in dfschedule.itertuples()}
    flight2scheduleDT={row.Flight:row.SDT for row in dfschedule.itertuples()}
    airports=set(dfschedule["From"])|set(dfschedule["To"])
    return len(flights),len(airports),len(tails),len(crews),len(paxs),sum(paxs)
    
def getScenarioSummary(dataset,scenario):
    dfschedule=pd.read_csv("Datasets/"+dataset+"/Schedule.csv",na_filter=None)
    flight2scheduleDT={row.Flight:row.SDT for row in dfschedule.itertuples()}
    dfdrpschedule=pd.read_csv("Scenarios/"+scenario+"/DrpSchedule.csv",na_filter=None)
    disruptedFlights=dfdrpschedule[dfdrpschedule["is_disrupted"]==1]["Flight"].tolist()
    flight2drpDT={row.Flight:row.SDT for row in dfdrpschedule.itertuples()}
    count1,count2=0,0
    for flight in disruptedFlights:
        delay=flight2drpDT[flight]-flight2scheduleDT[flight]
        if delay<=2*3600:
            count1+=1
        else:
            count2+=1
    return count1,count2,len(disruptedFlights)

if __name__=="__main__":
    
    for size in range(5,35,5):
        for typ in ["m","p"]:
            print(size,typ)
            ana=Analyzer("ACF%d"%size,"ACF%d-SC%s"%(size,typ),"Mode1")
            ana.parseOutputData()
            res=ana.getRunTimeAndGap()
            print(res)






