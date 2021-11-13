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
        self.type2mincontime={"ACF":self.config["ACMINCONTIME"],"CRW":self.config["CREWMINCONTIME"],"PAX":self.config["PAXMINCONTIME"],"ITIN":self.config["PAXMINCONTIME"]}
        self.FNodes=[Node(self,ntype="FNode",name=flight) for flight in self.dfdrpschedule["Flight"].tolist()]
            
        self.name2FNode={node.name:node for node in self.FNodes}
        self.FNode2name={node:node.name for node in self.FNodes}
        self.drpFNodes=[self.name2FNode[flight] for flight in self.disruptedFlights]
        self.tail2flights={tail:df_cur.sort_values(by='SDT')["Flight"].tolist() for tail,df_cur in self.dfdrpschedule.groupby("Tail")}
        self.crew2flights={crew:df_cur.sort_values(by='SDT')["Flight"].tolist() for crew,df_cur in self.dfdrpschedule.groupby("Crew")}
        
        self.itin2flights,self.itin2pax,self.flight2itinNum={},{},defaultdict(list)
        for row in self.dfitinerary.itertuples():
            flights=row.Flight_legs.split('-')
            self.itin2flights[row.Itinerary]=flights
            self.itin2pax[row.Itinerary]=row.Pax
            for flight in flights:
                self.flight2itinNum[flight].append((row.Itinerary,row.Pax))
        
        self.paxname2flights,self.paxname2itin,self.flight2paxnames={},{},defaultdict(list)
        for row in self.dfpassenger.itertuples():
            flights=row.Flights.split('-')
            self.paxname2flights[row.Pax]=flights
            self.paxname2itin[row.Pax]=row.Itinerary
            for flight in flights:
                self.flight2paxnames[flight].append(row.Pax)        
        
        self.itin2destination={itin:self.name2FNode[self.itin2flights[itin][-1]].Des for itin in self.itin2flights.keys()}
        self.tail2capacity={tail:df_cur["Capacity"].tolist()[0] for tail,df_cur in self.dfdrpschedule.groupby("Tail")}
        self.type2flightdict={"ACF":self.tail2flights,"CRW":self.crew2flights,"PAX":self.paxname2flights,"ITIN":self.itin2flights}

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
        
        if etype=="ACF" or etype=="CRW":
            self.EDT=S.config["STARTTIME"]+self.S.entity2delayTime.get(name,0) #ready time of crew or aircraft
            self.Nb=1
        elif etype=="PAX":
            self.EDT=S.flight2scheduleDT[self.F[0]] #ready time of passenger
            self.scheduleAT=S.flight2scheduleAT[self.F[-1]]
            self.Nb=1
        elif etype=="ITIN":
            self.EDT=S.flight2scheduleDT[self.F[0]] #ready time of itinerary
            self.scheduleAT=S.flight2scheduleAT[self.F[-1]]
            self.Nb=S.itin2pax[name]
            
        self.SNode=Node(S,"SNode","S-"+name,(None,self.Ori,self.EDT,None,0,-self.Nb))
        self.TNode=Node(S,"TNode","T-"+name,(self.Des,None,None,self.LAT,0,self.Nb))
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
        self.orig_nodes,self.orig_arcs=self.partialGraph.number_of_nodes(),self.partialGraph.number_of_edges()
        
    def generatePath(self,tempPath):
        lastNode=tempPath[-1]
        if (self.etype=="PAX" or self.etype=="ITIN") and lastNode.Des==self.Des:
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
          
    def plotPartialNetwork(self,ax):
        edgecolor=['blue' if edge in self.schedArcs else 'lightgrey' for edge in self.partialGraph.edges]
        nodecolor=['lightgreen' if node.name[:2] in ["S-","T-"] else 'orange' if node.name in self.S.disruptedFlights else 'lightblue' for node in self.partialGraph.nodes]
        nodelabel={node:self.Node2name[node] for node in self.partialGraph.nodes}
        nx.draw_circular(self.partialGraph,labels=nodelabel,with_labels=True,node_color=nodecolor,edge_color=edgecolor,ax=ax)
        ax.set_title("Partial Network of %s"%self.name)
            

class PSCAHelper:
    def __init__(self,S,etype,isbound,sizebound=None):
        self.S=S
        self.etype=etype
        self.isbound=isbound
        self.sizebound=sizebound
        print("Initiate %s Entities"%etype)
        if etype=="ACF":
            self.entities=[Entity(S,tname,"ACF") for tname in S.tail2flights]
        elif etype=="CRW":
            self.entities=[Entity(S,cname,"CRW") for cname in S.crew2flights]
        elif etype=="PAX":
            self.entities=[Entity(S,pname,"PAX") for pname in S.paxname2flights]
        elif etype=="ITIN":
            self.entities=[Entity(S,iname,"ITIN") for iname in S.itin2flights]
        if self.isbound:
            print("Initiate %s Reduced Partial Graphs"%self.etype)
            self.controlPartialGraphSize()
            
    def controlPartialGraphSize(self):        
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
        for drpNode in self.S.drpFNodes:
            node2distance=nx.single_source_dijkstra_path_length(self.etypeGraph,drpNode,cutoff=None,weight='weight')
            self.node2distDrp.update({node:dis for node,dis in node2distance.items() if node not in self.node2distDrp or self.node2distDrp[node]>dis})
        
        #iteratively remove arcs and nodes based on arc value
        for entity in self.entities:
            selnodes=set(self.S.FNodes)-set(entity.FNodes)
            valueWithArc=[]
            for ind,arc in enumerate(entity.partialGraph.edges):
                if arc[0].ntype=="SNode" or arc[1].ntype=="TNode":
                    valueWithArc.append((0,ind,arc))
                elif arc[0] in selnodes or arc[1] in selnodes:
                    valueWithArc.append((-np.mean([self.node2distDrp.get(arc[0],1),self.node2distDrp.get(arc[1],1)]),ind,arc))
                else:   # the schedule arcs fall here
                    valueWithArc.append((0,ind,arc))
                        
            hq.heapify(valueWithArc)
            while entity.partialGraph.number_of_edges()>self.sizebound:
                value,ind,arc=hq.heappop(valueWithArc)
                entity.eliminateArc(arc)
                
    def getGraphReductionStat(self,modeid):
        resd=defaultdict(list)
        for entity in self.entities:
            resd["entity"].append(entity.name)
            resd["orig_arc"].append(entity.orig_arcs)
            resd["final_arc"].append(entity.partialGraph.number_of_edges())
            resd["arc_reduce"].append("{0:.0%}".format((resd["orig_arc"][-1]-resd["final_arc"][-1])/resd["orig_arc"][-1]))
            resd["orig_sdArc"].append(len(entity.schedArcs))
            resd["final_sdArc"].append(len(set(entity.partialGraph.edges)&set(entity.schedArcs)))
            resd["sdArc_reduce"].append("{0:.0%}".format((resd["orig_sdArc"][-1]-resd["final_sdArc"][-1])/resd["orig_sdArc"][-1]))

        pd.DataFrame(resd).to_csv("Results/%s/%s/GraphStat-%s.csv"%(self.S.scname,modeid,self.etype))
        
    def plotEntityTypeNetwork(self,ax):
        edgecolor=['blue' if edge in self.schedArcs else 'lightgrey' for edge in self.etypeGraph.edges]
        nodecolor=['lightgreen' if node.name[:2] in ["S-","T-"] else 'orange' if node.name in self.S.disruptedFlights else 'lightblue' for node in self.etypeGraph.nodes]
        nodelabel={node:self.Node2name[node] for node in self.etypeGraph.nodes}
        nx.draw_circular(self.etypeGraph,labels=nodelabel,with_labels=True,node_color=nodecolor,edge_color=edgecolor,ax=ax)
        ax.set_title("Network of %s"%self.etype)





