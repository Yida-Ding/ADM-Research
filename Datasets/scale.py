import pandas as pd
import numpy as np
from collections import defaultdict

for size in range(25,30,5):
    df=pd.read_csv("ACF%d/Schedule.csv"%size)
    df["Capacity"]=np.array(df["Capacity"])//20
    df["Pax"]=np.array(df["Pax"])//20
    df.to_csv("ACF%d/Schedule.csv"%size,index=None)
    
    df=pd.read_csv("ACF%d/Itinerary.csv"%size)
    df["Pax"]=np.array(df["Pax"])//20
    df.to_csv("ACF%d/Itinerary.csv"%size,index=None)
    
for size in range(25,30,5):
    df=pd.read_csv("ACF%d/Itinerary.csv"%size)
    resdpax=defaultdict(list)
    for row in df.itertuples():
         for i in range(row.Pax):
            resdpax["Pax"].append(row.Itinerary+"+P%02d"%i)
            resdpax["Itinerary"].append(row.Itinerary)
            resdpax["Flights"].append(row.Flight_legs)
    ndf=pd.DataFrame(resdpax)
    ndf.to_csv("ACF%d/Passenger.csv"%size,index=None)
    print(ndf)


