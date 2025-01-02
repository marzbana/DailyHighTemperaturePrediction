# add the columns windspeed, radiation and evaporation from openmateo-9_17.csv to miami_weather_data_total.csv

import pandas as pd

# import miami_weather_data_total.csv as a df
df = pd.read_csv('../MiamiData/miami_weather_data_total.csv')

#import openmateo-9_17.csv as a df
df2 = pd.read_csv('openmateo-9_17.csv')

#windspeed_10m_max (mp/h),shortwave_radiation_sum (MJ/m²),et0_fao_evapotranspiration (mm)
#add the columns windspeed, radiation and evaporation from openmateo-9_17.csv to miami_weather_data_total.csv
df['windspeed_10m_max_(mp/h)'] = df2['windspeed_10m_max (mp/h)']

df['shortwave_radiation_sum_(MJ/m²)'] = df2['shortwave_radiation_sum (MJ/m²)']

df['et0_fao_evapotranspiration_(mm)'] = df2['et0_fao_evapotranspiration (mm)']

#save the new df as miami_weather_data_total.csv
df.to_csv('../DataSets/miami_weather_data_san_mateo.csv', index=False)