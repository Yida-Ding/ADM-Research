import numpy as np
import matplotlib.pyplot as plt
import json


class ResultVisualizer:
    def __init__(self,scenario,methods,length):
        self.scenario=scenario
        self.methods=methods
        with open("Results/%s/CostCPLEX.json"%scenario,"r") as outfile:
            self.cplexObj=json.load(outfile)["Objective"]
        self.Y=[]
        for method in self.methods:
            ys=np.load('Results/%s/res_%s.npz'%(scenario,method))['res']
            self.Y.append(ys[:length])
        self.Y=np.array(self.Y)
    
    def setMovingAverages(self,period):
        self.period=period
        self.xs=np.array([t for t in range(period,len(self.Y[0]))])
        self.movAvgY=np.array([[np.mean(row[t-period:t]) for t in range(period,len(row))] for row in self.Y])
    
    def plotMovingAverage(self,ax,save):
        for i,row in enumerate(self.movAvgY):
            ax.plot(self.xs,row,lw=1,label="VNS_"+self.methods[i])
        ax.plot(self.xs,[self.cplexObj]*len(self.xs),lw=1,label="CPLEX_optimal")
        ax.legend()
        ax.set_title(self.scenario)
        ax.set_xlabel("episode")
        ax.set_ylabel("moving average of objective in %d episodes"%self.period)
        if save:
            plt.savefig("Results/%s/Performance.png"%(self.scenario))
            
        
RV=ResultVisualizer("ACF7-SC1",["uniform","degree","distance","RFC","PPO"],2500)
RV.setMovingAverages(50)
fig,ax=plt.subplots(1,1,figsize=(8,6))
RV.plotMovingAverage(ax,save=False)



