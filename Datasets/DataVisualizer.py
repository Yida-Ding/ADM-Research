import math
import json
import random
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import networkx as nx
import haversine

class Dataset:
    def __init__(self,config,usedaps):
        self.config=config
        self.dfairports=pd.read_csv("0_External/OurAirports/airports.csv",na_filter=None)
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
            ap1,ap2,depTime,arrTime=flights[j]
            ax.plot([self.ap2loc[ap1][1],self.ap2loc[ap2][1]],[self.ap2loc[ap1][0],self.ap2loc[ap2][0]],"k--",linewidth=4.0,alpha=0.5,zorder=-10)
        return ax            
            
    def plotEntityTrajectoriesOnMap(self,dfschedule,entity):
        groups=dfschedule.groupby([entity])
        trajCount=len(groups)
        xpl=int(math.sqrt(trajCount))+1
        ypl=(trajCount//xpl)+1
        if xpl*(ypl-1)>=trajCount:
            ypl-=1
        fig,axs=plt.subplots(ypl,xpl,figsize=(12,12*ypl/xpl),dpi=100)
        for y in range(ypl):
            for x in range(xpl):
                axs[y][x].set_xticks([])
                axs[y][x].set_yticks([])
        for i,(groupname,groupdf) in enumerate(groups):
            ax=axs[i//xpl][i%xpl]
            flights=[(row.From,row.To,row.SDT,row.SAT) for row in groupdf.itertuples()]
            self.plotTrajectoryOnBasemap(flights,ax)
            ax.set_title("%s: "%entity+groupname)
            
    def plotEntityTrajectoriesAsTimeSpaceNetwork(self,dfschedule,entity):
        fig,ax=plt.subplots(1,1,figsize=(12,12),dpi=100)
        ax.set_yticks(range(len(self.airports)))
        ax.set_yticklabels(self.airports)
        ax.set_xlim(min(dfschedule.SDT),max(dfschedule.SAT))
        ax.grid(alpha=0.5)
        cmap = cm.Paired
        names=list(set(dfschedule[entity]))
        lastind,lastap2x=None,None
        for row in dfschedule.itertuples():
            ap1y=self.airports.index(row.From)
            ap2y=self.airports.index(row.To)
            ap1x=row.SDT
            ap2x=row.SAT
            name=getattr(row,entity)
            ind=names.index(name)
            ls=["-","--"][ind%2]
            ax.plot([ap1x,ap2x],[ap1y,ap2y],ls,color=cmap(ind/len(names)),label=name,linewidth=3,alpha=0.8)
            if lastind is not None and lastind==ind:
                ax.plot([lastap2x,ap1x],[ap1y,ap1y],ls,color=cmap(ind/len(names)),label=name,linewidth=3,alpha=0.8)
            lastind,lastap2x=ind,ap2x
            
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))
        ax.set_title("%sTrajectoriesAsTimeSpaceNetwork"%entity)


direname="ACF5"
with open(direname+"/Config.json", "r") as outfile:
    config=json.load(outfile)

dfschedule=pd.read_csv(direname+"/Schedule.csv",na_filter=None)
usedaps=set(dfschedule["From"].tolist()+dfschedule["To"].tolist())
D=Dataset(config,usedaps)

#D.plotBasemap(conalpha=0.9).set_title("Airports and possible flight connections") # Visualize airports and connectable airport pairs
D.plotEntityTrajectoriesOnMap(dfschedule,"Tail")
#D.plotEntityTrajectoriesAsTimeSpaceNetwork(dfschedule,"Crew")
    



