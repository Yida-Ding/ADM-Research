import json
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import networkx as nx
import haversine
import random
import itertools


class Visualizer:
    def __init__(self,direname):
        with open("Datasets/"+direname+"/Config.json", "r") as outfile:
            self.config=json.load(outfile)
        self.dfschedule=pd.read_csv("Datasets/"+direname+"/Schedule.csv",na_filter=None)
        self.dfairports=pd.read_csv("Datasets/0_External/OurAirports/airports.csv",na_filter=None)
        usedaps=set(self.dfschedule["From"].tolist()+self.dfschedule["To"].tolist())
        self.dfairports=self.dfairports[self.dfairports["iata_code"].isin(usedaps)]
        self.airports=self.dfairports["iata_code"].tolist()
        self.ap2loc={row.iata_code:(row.lat,row.lon) for row in self.dfairports.itertuples()}
        self.ap2pax={row.iata_code:row.pax for row in self.dfairports.itertuples()}
        self.appair2distance={(k1,k2):haversine.haversine(self.ap2loc[k1],self.ap2loc[k2]) for k1 in self.ap2loc for k2 in self.ap2loc}
        self.connectableappairs={k for k in self.appair2distance if self.appair2distance[k]>self.config["MINFLIGHTDISTANCE"] and self.appair2distance[k]<self.config["MAXFLIGHTDISTANCE"]} # Eliminate extremely short flights
        self.Gconnectable=nx.Graph(self.connectableappairs)

    def plotBasemap(self,highlightaps=None,conalpha=0.01,ax=None):
        if ax is None:
            fig,ax=plt.subplots(1,1,figsize=(12,8),dpi=100)
        ax.plot(self.dfairports.lon,self.dfairports.lat,"ko",markersize=0.1)
        for row in self.dfairports.itertuples():
            if highlightaps is None:
                ax.text(row.lon, row.lat, row.iata_code, fontsize=8,ha="center",va="center",bbox=dict(facecolor='lightblue', alpha=0.99,pad=1.5))
            else:
                if row.iata_code in highlightaps:
                    ax.text(row.lon, row.lat, row.iata_code, fontsize=10,ha="center",va="center",bbox=dict(facecolor='lightblue', alpha=0.9,pad=1.5))
                else:
                    ax.text(row.lon, row.lat, row.iata_code, fontsize=5,ha="center",va="center",bbox=dict(facecolor='lightblue', alpha=0.9,pad=1.5))
    
        for ap1,ap2 in self.connectableappairs:
            ax.plot([self.ap2loc[ap1][1],self.ap2loc[ap2][1]],[self.ap2loc[ap1][0],self.ap2loc[ap2][0]],"g-",alpha=conalpha)
        return ax

    def plotTrajectoryOnBasemap(self,flights,ax=None):
        usedaps=set([f[0] for f in flights]+[flights[-1][1]])
        ax=self.plotBasemap(highlightaps=usedaps,conalpha=0.06,ax=ax)
        for j in range(len(flights)):
            ap1,ap2,name=flights[j]
            ax.plot([self.ap2loc[ap1][1],self.ap2loc[ap2][1]],[self.ap2loc[ap1][0],self.ap2loc[ap2][0]],"k--",linewidth=4.0,alpha=0.5,zorder=-10)
            ax.text(sum([self.ap2loc[ap1][1],self.ap2loc[ap2][1]])/2+random.uniform(0,1),sum([self.ap2loc[ap1][0],self.ap2loc[ap2][0]])/2,name,fontsize=10,ha="center",va="center",bbox=dict(facecolor='orange',alpha=0.9,pad=1.5))
        return ax
    


