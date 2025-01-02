import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_json('miami_weather_data.json')

# Convert Unix timestamps to datetime objects
df['dt'] = pd.to_datetime(df['dt'], unit='s')

# Extract features from the 'main' and 'wind' dictionaries
df_main = df['main'].apply(pd.Series)
df_wind = df['wind'].apply(pd.Series)

# Merge the extracted features into the main dataframe
df = pd.concat([df, df_main, df_wind], axis=1)

# Drop the original 'main' and 'wind' columns
df.drop(['main', 'wind'], axis=1, inplace=True)

# Set the datetime as the index
df.set_index('dt', inplace=True)

# Sorting the dataframe by datetime index
df.sort_index(inplace=True)

# Plotting and saving graphs
def plot_and_save(column_name):
    plt.figure(figsize=(15, 5))
    plt.plot(df.index, df[column_name])
    plt.title(f'{column_name} over Time')
    plt.xlabel('Time')
    plt.ylabel(column_name)
    plt.savefig(f'weather_plots/{column_name}_over_Time.png')
    plt.close()

# Create a directory for saving plots if it doesn't exist
import os
if not os.path.exists('weather_plots'):
    os.makedirs('weather_plots')

# Columns to plot
columns_to_plot = ['temp', 'pressure', 'humidity', 'speed']

# Generate and save plots
for col in columns_to_plot:
    plot_and_save(col)
