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

class Scenario:
    def __init__(self,direname,scname,paxtype):
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
        self.tail2flights={tail:df_cur["Flight"].tolist() for tail,df_cur in self.dfdrpschedule.groupby("Tail")}
        self.crew2flights={crew:df_cur["Flight"].tolist() for crew,df_cur in self.dfdrpschedule.groupby("Crew")}
        
        self.itin2flights,self.itin2pax,self.flt2pax,self.flight2itinNum={},{},{},defaultdict(list)
        for row in self.dfitinerary.itertuples():
            flights=row.Flight_legs.split('-')
            self.itin2flights[row.Itinerary]=flights
            self.itin2pax[row.Itinerary]=row.Pax
            for flight in flights:
                self.flight2itinNum[flight].append((row.Itinerary,row.Pax))
                self.flt2pax[flight]=row.Pax
                    
        if paxtype=="PAX":
            self.paxname2flights,self.paxname2itin,self.flight2paxnames={},{},defaultdict(list)
            for row in self.dfpassenger.itertuples():
                flights=row.Flights.split('-')
                self.paxname2flights[row.Pax]=flights
                self.paxname2itin[row.Pax]=row.Itinerary
                for flight in flights:
                    self.flight2paxnames[flight].append(row.Pax)        
        
        self.itin2destination={itin:self.name2FNode[self.itin2flights[itin][-1]].Des for itin in self.itin2flights.keys()}
        self.tail2capacity={tail:df_cur["Capacity"].tolist()[0] for tail,df_cur in self.dfdrpschedule.groupby("Tail")}
        self.type2flightdict={"ACF":self.tail2flights,"CRW":self.crew2flights,paxtype:self.itin2flights if paxtype=="ITIN" else self.paxname2flights}

        #Generate connectable digraph for flight nodes
        self.connectableGraph,self.connectableGraphByName=nx.DiGraph(),nx.DiGraph()
        connectableEdges=[(node1,node2) for node1 in self.FNodes for node2 in self.FNodes if node1.Des==node2.Ori and node2.LDT>=node1.EAT+node1.CT]
        connectableEdgesByNames=[(node1.name,node2.name) for (node1,node2) in connectableEdges]
        self.connectableGraph.add_edges_from(connectableEdges)
        self.connectableGraphByName.add_edges_from(connectableEdgesByNames)
                    
    def plotEntireFlightNetwork(self,ax):
        colormap=['orange' if node.name in self.disruptedFlights else 'lightblue' for node in self.connectableGraph.nodes]
        nx.draw_circular(self.connectableGraph,labels=self.FNode2name,node_color=colormap,ax=ax)
        ax.set_title("Entire Flight Network")\
        
    def getTimeString(self,seconds):
        days,remainder=divmod(seconds,24*3600)
        hours,remainder=divmod(remainder,3600)
        minutes,seconds=divmod(remainder,60)
        s='{:02}:{:02}'.format(int(hours),int(minutes))
        if days>0:
            s+=" (+%d)"%days
        return s
        
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
            self.ScheduleDT=S.flight2scheduleDT[name]
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
            self.fathers=[]  # VNS
            self.children=[] # VNS
            
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
                        
        descendants,ancestors,newedges=set(),set(),set()
        for FNode in S.connectableGraph:
            if self.SNode.Des==FNode.Ori and FNode.LDT>=self.SNode.EAT:
                descendants.add(FNode)
                newedges.add((self.SNode,FNode))
                descendants=descendants|nx.descendants(S.connectableGraph,FNode)
            if FNode.Des==self.TNode.Ori and self.TNode.LDT>=FNode.EAT:
                ancestors.add(FNode)
                newedges.add((FNode,self.TNode))
                ancestors=ancestors|nx.ancestors(S.connectableGraph,FNode)
        
        partialNodes=(descendants&ancestors)|set([self.SNode,self.TNode])
        self.partialGraph=nx.DiGraph(S.connectableGraph.subgraph(partialNodes))
        self.partialGraph.add_edges_from(newedges)
        
        self.schedNodes=[self.SNode]+self.FNodes+[self.TNode]
        self.schedArcs={(self.schedNodes[i],self.schedNodes[i+1]) for i in range(len(self.schedNodes)-1)}
        self.schedArcs2=self.schedArcs|{(y,x) for (x,y) in self.schedArcs}
        self.orig_nodes,self.orig_arcs=self.partialGraph.number_of_nodes(),self.partialGraph.number_of_edges()
          
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
        
        for entity in self.entities:
            reduceGraph=nx.DiGraph()
            reduceGraph.add_edges_from(entity.schedArcs)
            selnodes=set(self.S.FNodes)-set(entity.FNodes)
            valueArcs=[(np.mean([self.node2distDrp.get(arc[0],1),self.node2distDrp.get(arc[1],1)]),ind,arc) if arc[0] in selnodes or arc[1] in selnodes else (0,ind,arc) for ind,arc in enumerate(entity.partialGraph.edges)]
            
            #revised size control algorithm: rebuild the path which contains arcs of small value until the size of reduceGraph is larger than the bound
            waitingArcs=[]
            hq.heapify(valueArcs)
            hq.heapify(waitingArcs)
            bound=min(entity.partialGraph.number_of_edges(),self.sizebound)
            while reduceGraph.number_of_edges()<bound:
                if len(valueArcs)!=0:
                    val,ind,arc=hq.heappop(valueArcs)
                elif len(waitingArcs)!=0:
                    val,ind,arc=hq.heappop(waitingArcs)
                    reduceGraph.add_edge(*arc)
                    continue
                    
                if nx.has_path(entity.partialGraph,entity.SNode,arc[0]) and nx.has_path(entity.partialGraph,arc[1],entity.TNode):
                    sourcepath=nx.shortest_path(entity.partialGraph,source=entity.SNode,target=arc[0])
                    targetpath=nx.shortest_path(entity.partialGraph,source=arc[1],target=entity.TNode)
                    reduceGraph.add_edge(*arc)
                    nx.add_path(reduceGraph,sourcepath)
                    nx.add_path(reduceGraph,targetpath)                    
                else:
                    hq.heappush(waitingArcs,(val,ind,arc))
            
            entity.partialGraph=reduceGraph
            print(entity.name)
                
    def getGraphReductionStat(self,modeid):
        resd=defaultdict(list)
        for entity in self.entities:
            resd["entity"].append(entity.name)
            resd["orig_node"].append(entity.orig_nodes)
            resd["final_node"].append(entity.partialGraph.number_of_nodes())
            resd["node_reduce"].append("{0:.0%}".format((resd["orig_node"][-1]-resd["final_node"][-1])/resd["orig_node"][-1]))
            resd["orig_arc"].append(entity.orig_arcs)
            resd["final_arc"].append(entity.partialGraph.number_of_edges())
            resd["arc_reduce"].append("{0:.0%}".format((resd["orig_arc"][-1]-resd["final_arc"][-1])/resd["orig_arc"][-1]))
            resd["orig_sdArc"].append(len(entity.schedArcs))
            resd["final_sdArc"].append(len(set(entity.partialGraph.edges)&set(entity.schedArcs)))
            resd["sdArc_reduce"].append("{0:.0%}".format((resd["orig_sdArc"][-1]-resd["final_sdArc"][-1])/resd["orig_sdArc"][-1]))

#        pd.DataFrame(resd).to_csv("Results/%s/%s/GraphStat-%s.csv"%(self.S.scname,modeid,self.etype),index=None)
        
    def plotEntityTypeNetwork(self,ax):
        edgecolor=['blue' if edge in self.schedArcs else 'lightgrey' for edge in self.etypeGraph.edges]
        nodecolor=['lightgreen' if node.name[:2] in ["S-","T-"] else 'orange' if node.name in self.S.disruptedFlights else 'lightblue' for node in self.etypeGraph.nodes]
        nodelabel={node:self.Node2name[node] for node in self.etypeGraph.nodes}
        nx.draw_circular(self.etypeGraph,labels=nodelabel,with_labels=True,node_color=nodecolor,edge_color=edgecolor,ax=ax)
        ax.set_title("Network of %s"%self.etype)

