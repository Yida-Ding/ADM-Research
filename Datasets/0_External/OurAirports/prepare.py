import pandas as pd

df=pd.read_csv("OurAirports.csv",na_filter=None)
df=df.rename(columns={"latitude_deg":"lat","longitude_deg":"lon"})
df=df[df["type"].isin(["small_airport","medium_airport","large_airport"])]
df=df[(df["lat"]>=-89)&(df["lat"]<=89)]
df=df[df["scheduled_service"]=="yes"]
df=df[(df["iata_code"]!="")&(df["iata_code"]!="0")]
print(df)

iso22iso3={row.ISO2:row.ISO3 for row in pd.read_csv("../Countries/countries.csv",na_filter=None).itertuples()}
df["iso3"]=[iso22iso3.get(row.iso_country,"") for row in df.itertuples()]
df=df[["iata_code","lat","lon","name","type","iso3","iso_region","municipality"]]
df=df[df["iso3"]!=""]

df2=pd.read_csv("airportsWithYearlyPax.csv",na_filter=None)
d={row.iata_code:row.passengers2015 for row in df2.itertuples()}

df["pax"]=[d.get(row.iata_code,0) for row in df.itertuples()]
df=df[df["pax"]>=100]

df=df.sort_values(["pax"],ascending=False)

df.to_csv("airports.csv",index=False)
