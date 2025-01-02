# Importing required libraries
import pandas as pd
import json
import matplotlib.pyplot as plt

def load_and_visualize_data(file_path):
    # Load data from file
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Convert timestamps to readable datetime format
    df['dt'] = pd.to_datetime(df['dt'], unit='s')
    
    # Extract the main parameters like temp, pressure, humidity into separate columns
    df = pd.concat([df.drop(['main'], axis=1), df['main'].apply(pd.Series)], axis=1)
    
    # Plotting Temperature over Time
    plt.figure(figsize=(14, 6))
    plt.plot(df['dt'], df['temp'], label='Temperature (K)', color='r')
    plt.xlabel('Time')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature over Time')
    plt.legend()
    plt.show()
    
    # Plotting Humidity over Time
    plt.figure(figsize=(14, 6))
    plt.plot(df['dt'], df['humidity'], label='Humidity (%)', color='b')
    plt.xlabel('Time')
    plt.ylabel('Humidity (%)')
    plt.title('Humidity over Time')
    plt.legend()
    plt.show()
    
    # Plotting Pressure over Time
    plt.figure(figsize=(14, 6))
    plt.plot(df['dt'], df['pressure'], label='Pressure (hPa)', color='g')
    plt.xlabel('Time')
    plt.ylabel('Pressure (hPa)')
    plt.title('Pressure over Time')
    plt.legend()
    plt.show()
    
    # Extract the wind parameters like speed into separate columns
    df_wind = pd.concat([df.drop(['wind'], axis=1), df['wind'].apply(pd.Series)], axis=1)
    
    # Plotting Wind Speed over Time
    plt.figure(figsize=(14, 6))
    plt.plot(df_wind['dt'], df_wind['speed'], label='Wind Speed (m/s)', color='y')
    plt.xlabel('Time')
    plt.ylabel('Wind Speed (m/s)')
    plt.title('Wind Speed over Time')
    plt.legend()
    plt.show()
    
    # Extract the cloud parameters like cloudiness into separate columns
    df_clouds = pd.concat([df.drop(['clouds'], axis=1), df['clouds'].apply(pd.Series)], axis=1)
    
    # Plotting Cloud Coverage over Time
    plt.figure(figsize=(14, 6))
    plt.plot(df_clouds['dt'], df_clouds['all'], label='Cloud Coverage (%)', color='c')
    plt.xlabel('Time')
    plt.ylabel('Cloud Coverage (%)')
    plt.title('Cloud Coverage over Time')
    plt.legend()
    plt.show()

# Replace 'your_file_path_here.json' with the path to your JSON file containing weather data
load_and_visualize_data('miami_weather_data.json')
