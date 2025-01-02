import pandas as pd

# Load data
df = pd.read_csv('../DataSets/weather_mateo_aqi_ocean_tide.csv')
#remove the last 12 rows
df = df[:-13]

# for the column percipitation replace T with .001
df['precipitation'] = df['precipitation'].replace('T', .001)

#use interpolation to fill in values that have M in columns: high_temp, avg_temp, HDD, CDD, windspeed_10m_max_(mp/h),shortwave_radiation_sum_(MJ/m²),et0_fao_evapotranspiration_(mm),AQI,average_ocean_temp,tide_height(ft)
df['high_temp'] = df['high_temp'].replace('M', None)
df['high_temp'] = df['high_temp'].interpolate(method='linear', limit_direction='forward', axis=0)
df['avg_temp'] = df['avg_temp'].replace('M', None)
df['avg_temp'] = df['avg_temp'].interpolate(method='linear', limit_direction='forward', axis=0)
df['HDD'] = df['HDD'].replace('M', None)
df['HDD'] = df['HDD'].interpolate(method='linear', limit_direction='forward', axis=0)
df['CDD'] = df['CDD'].replace('M', None)
df['CDD'] = df['CDD'].interpolate(method='linear', limit_direction='forward', axis=0)
df['windspeed_10m_max_(mp/h)'] = df['windspeed_10m_max_(mp/h)'].replace('M', None)
df['windspeed_10m_max_(mp/h)'] = df['windspeed_10m_max_(mp/h)'].interpolate(method='linear', limit_direction='forward', axis=0)
df['shortwave_radiation_sum_(MJ/m²)'] = df['shortwave_radiation_sum_(MJ/m²)'].replace('M', None)
df['shortwave_radiation_sum_(MJ/m²)'] = df['shortwave_radiation_sum_(MJ/m²)'].interpolate(method='linear', limit_direction='forward', axis=0)
df['et0_fao_evapotranspiration_(mm)'] = df['et0_fao_evapotranspiration_(mm)'].replace('M', None)
df['et0_fao_evapotranspiration_(mm)'] = df['et0_fao_evapotranspiration_(mm)'].interpolate(method='linear', limit_direction='forward', axis=0)
df['AQI'] = df['AQI'].replace('M', None)
df['AQI'] = df['AQI'].interpolate(method='linear', limit_direction='forward', axis=0)
df['average_ocean_temp'] = df['average_ocean_temp'].replace('M', None)
df['average_ocean_temp'] = df['average_ocean_temp'].interpolate(method='linear', limit_direction='forward', axis=0)
df['tide_height(ft)'] = df['tide_height(ft)'].replace('M', None)
df['tide_height(ft)'] = df['tide_height(ft)'].interpolate(method='linear', limit_direction='forward', axis=0)



#save file
df.to_csv('../DataSets/weather_mateo_aqi_ocean_tide_train.csv', index=False)