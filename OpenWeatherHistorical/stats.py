import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Recreate the statistics summary DataFrame
statistics_df = df.describe()

# Save the statistics summary plot to a file
statistics_summary_plot = statistics_df.plot.bar(rot=0, figsize=(16, 8), grid=True)
plt.title('Statistics Summary of Weather Data')
plt.xlabel('Statistics')
plt.ylabel('Value')
statistics_summary_plot.get_figure().savefig('weather_data_statistics_summary.png')
