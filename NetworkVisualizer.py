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
import numpy as np


class Visualizer:
    def __init__(self,direname,scenario=None):
        self.direname=direname
        self.scenario=scenario
        
    def getTimeString(self,seconds):
        days,remainder=divmod(seconds,24*3600)
        hours,remainder=divmod(remainder,3600)
        minutes,seconds=divmod(remainder,60)
        s='{:02}:{:02}'.format(int(hours),int(minutes))
        return s
    
    def getScheduleData(self):
        dfschedule=pd.read_csv("Datasets/"+self.direname+'/Schedule.csv',na_filter=None)
        flightlists=[dfcur["Flight"].tolist() for tail,dfcur in dfschedule.groupby("Tail")]
        flight2info=defaultdict(list)
        for row in dfschedule.itertuples():
            flight2info[row.Flight].append(row.Crew)
            flight2info[row.Flight].append(row.Pax)
            flight2info[row.Flight].append(row.From)
            flight2info[row.Flight].append(row.To)
            flight2info[row.Flight].append(self.getTimeString(row.SDT))
            flight2info[row.Flight].append(self.getTimeString(row.SAT))
            flight2info[row.Flight].append(row.Tail)
            flight2info[row.Flight].append(0) #DelayType
        return dfschedule,flightlists,flight2info,"Schedule"
            
    def getRecoveryData(self,title="CPLEX"):
        dfrecovery=pd.read_csv("Results/"+self.scenario+'/Recovery%s.csv'%title,na_filter=None)
        with open("Results/"+self.scenario+"/Cost%s.json"%title, "r") as outfile:
            costDict=json.load(outfile)
            cost=costDict["Objective"]
                
        title=title+"(obj=%d)"%cost
        flightlists=[dfcur["Flight"].tolist() for tail,dfcur in dfrecovery.groupby("Tail")]
        flight2info=defaultdict(list)
        for row in dfrecovery.itertuples():
            flight2info[row.Flight].append(row.Crew)
            flight2info[row.Flight].append(row.Pax)
            flight2info[row.Flight].append(row.From)
            flight2info[row.Flight].append(row.To)
            flight2info[row.Flight].append(self.getTimeString(row.RDT))
            flight2info[row.Flight].append(self.getTimeString(row.RAT))
            flight2info[row.Flight].append(row.Tail)
            flight2info[row.Flight].append(row.DelayType) #DelayType
        return dfrecovery,flightlists,flight2info,title
        
    def plotFlightNetwork(self,df,flightlists,flight2info,title,ax=None): # depends on flightlists and flight2info
        xmax=max([len(fltlst) for fltlst in flightlists])
        ymax=len(flightlists)
        if ax==None:
            fig,ax=plt.subplots(1,1,figsize=(xmax*5,ymax*3))
        
        locs=[(fltlst.index(flt),-flightlists.index(fltlst)) for fltlst in flightlists for flt in fltlst]   
        crew2color={crew:plt.get_cmap("gist_rainbow")(i/len(set(df["Crew"]))) for i,crew in enumerate(set(df["Crew"]))}
        tails=df.Tail.unique()
        r1,r2=0.1,0.1;lambdas=[120,60,180,0,240,300];thetas=np.linspace(0,2*np.pi,100)
        fltCircleColor={0:"lightblue",1:"red",2:"orange"}
        tail2capacity={row.Tail:row.Capacity for row in df.itertuples()}
        
        for (x,y) in locs:
            flight=flightlists[-y][x]
            ax.plot(x+(r1+2*r2)*np.cos(thetas),y+(r1+2*r2)*np.sin(thetas),c='lightblue')
            ax.plot(x+r1*np.cos(thetas),y+r1*np.sin(thetas),c=fltCircleColor[flight2info[flight][7]],linewidth=2)
            ax.text(x,y,flight,c='black',fontsize=14,weight="bold",horizontalalignment='center',verticalalignment='center')
            for i,lamda in enumerate(lambdas):
                newx,newy=x+(r1+r2)*np.cos(np.deg2rad(lamda)),y+(r1+r2)*np.sin(np.deg2rad(lamda))
                ax.plot(newx+r2*np.cos(thetas),newy+r2*np.sin(thetas),c='lightblue',alpha=0.8)
                if i==0:
                    crew=flight2info[flight][i]
                    ax.text(newx,newy,crew,c=crew2color[crew],weight="bold",fontsize=14,horizontalalignment='center',verticalalignment='center')
                elif i==1:
                    ax.text(newx,newy,str(flight2info[flight][i]),c='black',fontsize=14,horizontalalignment='center',verticalalignment='center')
                else:
                    ax.text(newx,newy,flight2info[flight][i],c='black',fontsize=11,horizontalalignment='center',verticalalignment='center')
            if (x+1,y) in locs:
                ax.plot([x+r1+2*r2,x+1-r1-2*r2],[y,y],'-->',c='grey',linewidth=2)
        
        for y in range(ymax):
            ax.text(-0.6,-y,tails[y]+"\n(%d)"%(tail2capacity[tails[y]]),c='black',fontsize=14,weight="bold",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='none', edgecolor='black'))
            
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.axis('equal')
        plt.savefig("Results/%s/%s.png"%(V.scenario,title))
        plt.close()
        
if __name__ == '__main__':
    
    for i in range(5,20,5):
        for typ in ['m','p']:

            V=Visualizer("ACF%d"%i,"ACF%d-SC%s"%(i,typ))
            V.plotFlightNetwork(*V.getScheduleData())
            V.plotFlightNetwork(*V.getRecoveryData("CPLEX"))
            V.plotFlightNetwork(*V.getRecoveryData("VNS"))
    
    V=Visualizer("ACF5","ACF5-SCm")
    V.plotFlightNetwork(*V.getScheduleData())
    V.plotFlightNetwork(*V.getRecoveryData("CPLEX"))
    V.plotFlightNetwork(*V.getRecoveryData("VNS"))

    

