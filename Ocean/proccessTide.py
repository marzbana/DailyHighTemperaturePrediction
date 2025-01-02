# combine the data from the three years into one DataFrame, and then merge it with the weather data.
# the data is hourly so find the average of the ocean temperature for each day

import pandas as pd
import numpy as np

#read Ocean data
df_2021 = pd.read_csv('Tide_Jan1_Dec31_2021.csv')
df_2022 = pd.read_csv('Tide_Jan1_Dec31_2022.csv')
df_2023 = pd.read_csv('Tide_Jan1_Aug31.csv')
df2_2023 = pd.read_csv('Tide_Sep1_Sep24.csv')

df_2021['Verified (ft)'] = pd.to_numeric(df_2021['Verified (ft)'], errors='coerce')
df_2022['Verified (ft)'] = pd.to_numeric(df_2022['Verified (ft)'], errors='coerce')
df_2023['Verified (ft)'] = pd.to_numeric(df_2023['Verified (ft)'], errors='coerce')
df2_2023['Predicted (ft)'] = pd.to_numeric(df2_2023['Predicted (ft)'], errors='coerce')

#new dataframe with only the columns we need


#iterate through the data and find the average of the ocean temperature for each day
#add the average to a new column in the dataframe

# new df avg that only has the average ocean temperature column
df1 = pd.DataFrame(columns=['tide_height(ft)'])
df2 = pd.DataFrame(columns=['tide_height(ft)'])
df3 = pd.DataFrame(columns=['tide_height(ft)'])


for i in range(0, len(df_2021), 24):
    avg_tide_2021 = df_2021[i:i+24]['Verified (ft)'].mean()
    avg_tide_2022 = df_2022[i:i+24]['Verified (ft)'].mean()
    
    df1 = df1._append({'tide_height(ft)': avg_tide_2021}, ignore_index=True)
    df2 = df2._append({'tide_height(ft)': avg_tide_2022}, ignore_index=True)
    
for i in range(0, len(df_2023),24):
    #add the average of the ocean temperature over each hour to the new dataframe
     avg_tide_2023 = df_2023[i:i+24]['Verified (ft)'].mean()
     df3 = df3._append({'tide_height(ft)': avg_tide_2023}, ignore_index=True)
for i in range(0, len(df2_2023),240):
    #add the average of the ocean temperature over each hour to the new dataframe
     avg_tide_2023 = df2_2023[i:i+240]['Predicted (ft)'].mean()
     df3 = df3._append({'tide_height(ft)': avg_tide_2023}, ignore_index=True)
df = pd.concat([df1, df2, df3], ignore_index=True)
#merge the average ocean temperature with the weather data
weather = pd.read_csv('../DataSets/weather_mateo_aqi_ocean.csv')
weather['tide_height(ft)'] = df['tide_height(ft)']
weather.to_csv('../DataSets/weather_mateo_aqi_ocean_tide.csv', index=False)

