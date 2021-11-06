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
import time
import numpy as np
import heapq as hq

from NetworkVisualizer import Visualizer

class Scenario:
    def __init__(self,direname,scname):
        self.direname=direname
        self.scname=scname
        #load from Dataset
        with open("Datasets/"+direname+"/Config.json", "r") as outfile:
            self.config=json.load(outfile)
        self.dfitinerary=pd.read_csv("Datasets/"+direname+"/Itinerary.csv",na_filter=None)
        self.dfschedule=pd.read_csv("Datasets/"+direname+"/Schedule.csv",na_filter=None)
        self.dfpassenger=pd.read_csv("Datasets/"+direname+"/Passenger.csv",na_filter=None)
        self.flight2scheduleAT={row.Flight:row.SAT for row in self.dfschedule.itertuples()}
        self.flight2scheduleDT={row.Flight:row.SDT for row in self.dfschedule.itertuples()}
        
        #load from Scenario
        self.dfdrpschedule=pd.read_csv("Scenarios/"+scname+"/DrpSchedule.csv",na_filter=None)
        self.disruptedFlights=self.dfdrpschedule[self.dfdrpschedule["is_disrupted"]==1]["Flight"].tolist()
        with open("Scenarios/"+scname+"/DelayedReadyTime.json", "r") as outfile:
            self.entity2delayTime=json.load(outfile)
                
        #Create FNodes
        self.flight2dict={dic['Flight']:dic for dic in self.dfdrpschedule.to_dict(orient='record')}
        self.type2mincontime={"ACF":self.config["ACMINCONTIME"],"CRW":self.config["CREWMINCONTIME"],"PAX":self.config["PAXMINCONTIME"]}
        self.FNodes=[Node(self,ntype="FNode",name=flight) for flight in self.dfdrpschedule["Flight"].tolist()]
            
        self.name2FNode={node.name:node for node in self.FNodes}
        self.FNode2name={node:node.name for node in self.FNodes}
        self.drpFNodes=[self.name2FNode[flight] for flight in self.disruptedFlights]
        self.tail2flights={tail:df_cur.sort_values(by='SDT')["Flight"].tolist() for tail,df_cur in self.dfdrpschedule.groupby("Tail")}
        self.crew2flights={crew:df_cur.sort_values(by='SDT')["Flight"].tolist() for crew,df_cur in self.dfdrpschedule.groupby("Crew")}
        self.itin2flights={row.Itinerary:row.Flight_legs.split('-') for row in self.dfitinerary.itertuples()}
        
        self.paxname2flights,self.paxname2itin,self.flight2paxnames={},{},defaultdict(list)
        for row in self.dfpassenger.itertuples():
            flights=row.Flights.split('-')
            self.paxname2flights[row.Pax]=flights
            self.paxname2itin[row.Pax]=row.Itinerary
            for flight in flights:
                self.flight2paxnames[flight].append(row.Pax)
        
        self.itin2pax={row.Itinerary:row.Pax for row in self.dfitinerary.itertuples()}
        self.itin2destination={itin:self.name2FNode[self.itin2flights[itin][-1]].Des for itin in self.itin2flights.keys()}
        self.tail2capacity={tail:df_cur["Capacity"].tolist()[0] for tail,df_cur in self.dfdrpschedule.groupby("Tail")}
        self.type2flightdict={"ACF":self.tail2flights,"CRW":self.crew2flights,"PAX":self.paxname2flights}

        #Generate connectable digraph for flight nodes
        self.connectableGraph=nx.DiGraph()
        for node1 in self.FNodes:
            for node2 in self.FNodes:
                if node1.Des==node2.Ori and node2.LDT>=node1.EAT+node1.CT:
                    self.connectableGraph.add_edge(node1,node2)
                    
    def plotEntireFlightNetwork(self,ax):
        colormap=['orange' if node.name in self.disruptedFlights else 'lightblue' for node in self.connectableGraph.nodes]
        nx.draw_circular(self.connectableGraph,labels=self.FNode2name,node_color=colormap,ax=ax)
        ax.set_title("Entire Flight Network")
        
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
            self.ScheduleAT=S.flight2scheduleAT[name]
            self.ScheCrsTime=S.flight2dict[name]["Cruise_time"]
            self.CrsTimeComp=int(self.ScheCrsTime*S.config["CRSTIMECOMPPCT"])
            self.EDT=self.SDT
            self.EAT=self.SDT+self.SFT-self.CrsTimeComp
            self.CT=min(S.type2mincontime.values())
            self.CrsStageDistance=S.flight2dict[name]["Distance"]*S.config["CRUISESTAGEDISTPCT"]
            self.ScheCrsSpeed=self.CrsStageDistance/S.flight2dict[name]["Cruise_time"]
            self.MaxCrsSpeed=self.CrsStageDistance/(S.flight2dict[name]["Cruise_time"]-self.CrsTimeComp)
            self.ScheFuelConsump=self.CrsStageDistance*(sum([S.config["FUELCONSUMPPARA"][i]*[(self.ScheCrsSpeed)**2,self.ScheCrsSpeed,(self.ScheCrsSpeed)**(-2),(self.ScheCrsSpeed)**(-3)][i] for i in range(4)]))
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
        self.LAT=S.config["ENDTIME"]
        if etype!="PAX":
            self.EDT=S.config["STARTTIME"]+self.S.entity2delayTime.get(name,0) #ready time of crew or aircraft
        else:
            self.EDT=S.flight2scheduleDT[self.F[0]] #ready time of passenger
            self.scheduleAT=S.flight2scheduleAT[self.F[-1]]
        self.SNode=Node(S,"SNode","S-"+name,(None,self.Ori,self.EDT,None,0,-1))
        self.TNode=Node(S,"TNode","T-"+name,(self.Des,None,None,self.LAT,0,1))
        self.FNodes=[S.name2FNode[name] for name in self.F]
        self.name2Node=S.name2FNode.copy()
        self.name2Node.update({"S-"+name:self.SNode,"T-"+name:self.TNode})
        self.Node2name={node:name for name,node in self.name2Node.items()}
        
        self.connectableGraph=S.connectableGraph.copy()
        for FNode in S.connectableGraph:
            if self.SNode.Des==FNode.Ori and FNode.LDT>=self.SNode.EAT:
                self.connectableGraph.add_edge(self.SNode,FNode)
            if FNode.Des==self.TNode.Ori and self.TNode.LDT>=FNode.EAT:
                self.connectableGraph.add_edge(FNode,self.TNode)
        
        self.partialGraph=nx.DiGraph()
        self.generatePath([self.SNode])     #PNGA algorithm
        self.schedNodes=[self.SNode]+self.FNodes+[self.TNode]
        self.schedArcs={(self.schedNodes[i],self.schedNodes[i+1]) for i in range(len(self.schedNodes)-1)}
        self.schedArcs2=self.schedArcs|{(y,x) for (x,y) in self.schedArcs}
        
    def generatePath(self,tempPath):
        lastNode=tempPath[-1]
        if self.etype=="PAX" and lastNode.Des==self.Des:
            return
        for nextNode in self.connectableGraph.successors(lastNode):
            if nextNode not in tempPath:
                tempPathc=tempPath.copy()+[nextNode]
                if nextNode in self.partialGraph.nodes:
                    nx.add_path(self.partialGraph,tempPathc)
                else:
                    if nextNode.Des==self.Des:
                        nx.add_path(self.partialGraph,tempPathc+[self.TNode])
                    self.generatePath(tempPathc)
    
    def plotPartialNetwork(self,ax):
        edgecolor=['blue' if edge in self.schedArcs else 'lightgrey' for edge in self.partialGraph.edges]
        nodecolor=['lightgreen' if node.name[:2] in ["S-","T-"] else 'orange' if node.name in self.S.disruptedFlights else 'lightblue' for node in self.partialGraph.nodes]
        nodelabel={node:self.Node2name[node] for node in self.partialGraph.nodes}
        nx.draw_circular(self.partialGraph,labels=nodelabel,with_labels=True,node_color=nodecolor,edge_color=edgecolor,ax=ax)
        ax.set_title("Partial Network of %s"%self.name)
    
    def eliminateArc(self,arc):
        if arc in self.partialGraph.edges:
            self.partialGraph.remove_edge(*arc)
        if arc[0] in self.partialGraph.nodes and self.partialGraph.out_degree(arc[0])==0:
            self.eliminateNode(arc[0])
        if arc[1] in self.partialGraph.nodes and self.partialGraph.in_degree(arc[1])==0:
            self.eliminateNode(arc[1])
    
    def eliminateNode(self,node):
        arcset=set(self.partialGraph.in_edges(node))|set(self.partialGraph.out_edges(node))
        if node in self.partialGraph.nodes:
            self.partialGraph.remove_node(node)
        for arc in arcset:
            self.eliminateArc(arc)
          
            
class PSCAHelper:
    def __init__(self,S,etype,sizebound=float("inf")):
        self.S=S
        self.etype=etype
        if etype=="ACF":
            self.entities=[Entity(S,tname,"ACF") for tname in S.tail2flights]
        elif etype=="CRW":
            self.entities=[Entity(S,cname,"CRW") for cname in S.crew2flights]
        
        #merge the partial graphs of the same entity type
        self.etypeGraph=nx.Graph()
        self.schedArcs=set()
        self.Node2name={}
        for entity in self.entities:
            self.etypeGraph.add_edges_from(entity.partialGraph.edges,weight=1)
            self.schedArcs=self.schedArcs|entity.schedArcs2
            self.Node2name.update(entity.Node2name)
        self.etypeGraph.add_edges_from(self.schedArcs,weight=0)
        
        #calculate the shortest path distance from each flight node to nearest disrupted nodes
        self.node2distDrp={}
        for drpNode in S.drpFNodes:
            node2distance=nx.single_source_dijkstra_path_length(self.etypeGraph,drpNode,cutoff=None,weight='weight')
            self.node2distDrp.update({node:dis for node,dis in node2distance.items() if node not in self.node2distDrp or self.node2distDrp[node]>dis})
        
        #iteratively remove arcs and nodes based on arc value
        resd=defaultdict(list)
        for entity in self.entities:
            selnodes=set(S.FNodes)-set(entity.FNodes)
            valueWithArc=[(-np.mean([self.node2distDrp[arc[0]],self.node2distDrp[arc[1]]]) if arc[0] in selnodes or arc[1] in selnodes else 0, ind, arc) for ind,arc in enumerate(entity.partialGraph.edges)] #Eqn40
            hq.heapify(valueWithArc)
            resd["entity"].append(entity.name)
            resd["orig_nodes"].append(entity.partialGraph.number_of_nodes())
            resd["orig_arcs"].append(entity.partialGraph.number_of_edges())

            while entity.partialGraph.number_of_edges()>sizebound:
                value,ind,arc=hq.heappop(valueWithArc)
                entity.eliminateArc(arc)
            
            resd["final_nodes"].append(entity.partialGraph.number_of_nodes())
            resd["final_arcs"].append(entity.partialGraph.number_of_edges())
            resd["node_reduce"].append("{0:.0%}".format((resd["orig_nodes"][-1]-resd["final_nodes"][-1])/resd["orig_nodes"][-1]))
            resd["arc_reduce"].append("{0:.0%}".format((resd["orig_arcs"][-1]-resd["final_arcs"][-1])/resd["orig_arcs"][-1]))
        
        print("PSCA (Bound=%f):"%sizebound)
        print(pd.DataFrame(resd))
            
    def plotEntityTypeNetwork(self,ax):
        edgecolor=['blue' if edge in self.schedArcs else 'lightgrey' for edge in self.etypeGraph.edges]
        nodecolor=['lightgreen' if node.name[:2] in ["S-","T-"] else 'orange' if node.name in self.S.disruptedFlights else 'lightblue' for node in self.etypeGraph.nodes]
        nodelabel={node:self.Node2name[node] for node in self.etypeGraph.nodes}
        nx.draw_circular(self.etypeGraph,labels=nodelabel,with_labels=True,node_color=nodecolor,edge_color=edgecolor,ax=ax)
        ax.set_title("Network of %s"%self.etype)
        

#dataset="ACF2"
#S=Scenario(dataset,dataset+"-SC1")
#ACFs=PSCAHelper(S,"ACF",25)
#fig,axes=plt.subplots(1,2,figsize=(15,6))
#A1=ACFs.entities[0]
#A1.plotPartialNetwork(axes[0])
#A2=ACFs.entities[1]
#A2.plotPartialNetwork(axes[1])
























