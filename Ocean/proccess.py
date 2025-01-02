# combine the data from the three years into one DataFrame, and then merge it with the weather data.
# the data is hourly so find the average of the ocean temperature for each day

import pandas as pd
import numpy as np

#read Ocean data
df_2021 = pd.read_csv('Jan1_Dec31_2021.csv')
df_2022 = pd.read_csv('Jan1_Dec31_2022.csv')
df_2023 = pd.read_csv('Jan1_Sep25_2023.csv')

df_2021['Water Temp (°F)'] = pd.to_numeric(df_2021['Water Temp (°F)'], errors='coerce')
df_2022['Water Temp (°F)'] = pd.to_numeric(df_2022['Water Temp (°F)'], errors='coerce')
df_2023['Water Temp (°F)'] = pd.to_numeric(df_2023['Water Temp (°F)'], errors='coerce')

#new dataframe with only the columns we need


#iterate through the data and find the average of the ocean temperature for each day
#add the average to a new column in the dataframe

# new df avg that only has the average ocean temperature column
df1 = pd.DataFrame(columns=['average_ocean_temp'])
df2 = pd.DataFrame(columns=['average_ocean_temp'])
df3 = pd.DataFrame(columns=['average_ocean_temp'])

for i in range(0, len(df_2021), 24):
    avg_temp_2021 = df_2021[i:i+24]['Water Temp (°F)'].mean()
    avg_temp_2022 = df_2022[i:i+24]['Water Temp (°F)'].mean()
    
    df1 = df1._append({'average_ocean_temp': avg_temp_2021}, ignore_index=True)
    df2 = df2._append({'average_ocean_temp': avg_temp_2022}, ignore_index=True)
    
for i in range(0, len(df_2023),24):
    #add the average of the ocean temperature over each hour to the new dataframe
     avg_temp_2023 = df_2023[i:i+24]['Water Temp (°F)'].mean()
     df3 = df3._append({'average_ocean_temp': avg_temp_2023}, ignore_index=True)
df = pd.concat([df1, df2, df3], ignore_index=True)
print(df)
#merge the average ocean temperature with the weather data
weather = pd.read_csv('../DataSets/weather_mateo_aqi.csv')
weather['average_ocean_temp'] = df['average_ocean_temp']
weather.to_csv('../DataSets/weather_mateo_aqi_ocean.csv', index=False)

