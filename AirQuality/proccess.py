import pandas as pd

# Read AQI data
aqi_2021 = pd.read_csv('aqidaily2021.csv')
aqi_2022 = pd.read_csv('aqidaily2022.csv')
aqi_2023 = pd.read_csv('aqidaily2023.csv')

# Combine AQI data
all_aqi = pd.concat([aqi_2021, aqi_2022, aqi_2023], ignore_index=True)

# Read weather data
weather_data = pd.read_csv('../DataSets/miami_weather_data_san_mateo.csv')

# Assuming both have a 'Date' column to merge on
weather_data['AQI'] = all_aqi['Overall AQI Value']

weather_data.to_csv('../DataSets/weather_mateo_aqi.csv', index=False)
