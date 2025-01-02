# Let's first run the cleaning and processing code to get the processed_data_json
import pandas as pd
import json
import re

def remove_alphabetical_columns_from_json(df):
    cols_to_remove = [col for col in df.columns if df[col].apply(lambda x: str(x).isalpha()).all()]
    df_cleaned = df.drop(columns=cols_to_remove)
    return df_cleaned

def remove_non_numerical_from_json_automatic(df):
    columns_to_clean = [col for col in df.columns if df[col].apply(lambda x: any(char.isdigit() for char in str(x)) and any(char.isalpha() for char in str(x))).any()]
    for col in columns_to_clean:
        df[col] = df[col].apply(lambda x: float(re.sub('[^0-9.]', '', str(x))))
    return df

def process_weather_data(raw_data):
    processed_data = []
    for period in raw_data['properties']['periods']:
        data_point = {
            "timestamp": period.get("startTime", None),
            "temperature": period.get("temperature", None),
            "windSpeed": period.get("windSpeed", None),
            "dewpoint": period.get("dewpoint", None),
            "probabilityOfPrecipitation": period.get("probabilityOfPrecipitation", None),
            "relativeHumidity": period.get("relativeHumidity", None),
        }
        processed_data.append(data_point)
    
    df = pd.DataFrame(processed_data)
    df.columns = [col.rstrip('/') for col in df.columns]  # Remove trailing slashes from column names
    df_cleaned = remove_alphabetical_columns_from_json(df)
    df_final = remove_non_numerical_from_json_automatic(df_cleaned)
    return df_final.to_json(orient='records')

# Load the JSON data from the uploaded file
with open('data.json', 'r') as infile:
    raw_data = json.load(infile)

# Process the data
processed_data_json = process_weather_data(raw_data)

# Now let's visualize the processed data
import matplotlib.pyplot as plt
import seaborn as sns

# Reload the cleaned data into a DataFrame
df_cleaned = pd.read_json(processed_data_json)

# Generate pairplot for all numerical columns to visualize the relationships between them
sns.pairplot(df_cleaned)
plt.suptitle('Pairplot of All Numerical Variables', y=1.02)
plt.show()

# Generate correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Generate individual plots for each variable against 'temperature'
target = 'temperature'
features = [col for col in df_cleaned.columns if col != target]

for feature in features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=feature, y=target, data=df_cleaned)
    plt.title(f'{target} vs {feature}')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.show()
