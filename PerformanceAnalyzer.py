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
        labels=["U-VNS","DT-VNS","DG-VNS","PPO-VNS"]
        for i,row in enumerate(self.movAvgY):
            ax.plot(self.xs,row,lw=1,label=labels[i])
        ax.plot(self.xs,[self.cplexObj]*len(self.xs),lw=1,label="CPLEX")
        ax.set_xlabel("Episode",fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=13)


        
scen=[["ACF5-SCm","ACF10-SCm","ACF15-SCm"],["ACF5-SCp","ACF10-SCp","ACF15-SCp"]]
fig,axes=plt.subplots(2,3,figsize=(20,10))
for i in range(2):
    for j in range(3):
        RV=ResultVisualizer(scen[i][j],["uniform","distance","degree","PPO"],2000)
        RV.setMovingAverages(100)
        RV.plotMovingAverage(axes[i][j],save=False)
        
axes[0][0].set_ylabel("Mov. avg. of obj. in 100 episodes",fontsize=16)
axes[1][0].set_ylabel("Mov. avg. of obj. in 100 episodes",fontsize=16)
axes[0][0].set_title("AC05_DS$^-$",fontsize=18)
axes[0][1].set_title("AC10_DS$^-$",fontsize=18)
axes[0][2].set_title("AC15_DS$^-$",fontsize=18)
axes[1][0].set_title("AC05_DS$^+$",fontsize=18)
axes[1][1].set_title("AC10_DS$^+$",fontsize=18)
axes[1][2].set_title("AC15_DS$^+$",fontsize=18)

axes[0][2].legend(fontsize=14,loc=5)
axes[1][2].legend(fontsize=14,loc=5)


plt.tight_layout()
#plt.savefig("Results/PerformanceSummary.pdf")
