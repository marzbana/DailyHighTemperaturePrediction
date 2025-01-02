import requests
import json
import re
import pandas as pd

response = requests.get("https://api.weather.gov/points/25.7617,-80.1918", headers={"User-Agent": "Bjmalexunited@yahoo.com"})
if response.status_code == 200:
    data = response.json()
    office = data['properties']['cwa']
    gridX = data['properties']['gridX']
    gridY = data['properties']['gridY']
    print(data)

# Fetch the hourly forecast using the grid information
hourly_forecast_url = f"https://api.weather.gov/gridpoints/{office}/{gridX},{gridY}/forecast/hourly"
response = requests.get(hourly_forecast_url, headers={"User-Agent": "Bjmalexunited@yahoo.com"})
if response.status_code == 200:
    hourly_data = response.json()
    # Process the hourly_data
    print(hourly_data)
    #store to json file
    with open('data.json', 'w') as outfile:
        json.dump(hourly_data, outfile)
def remove_alphabetical_columns_from_json(json_data):
    """
    Remove columns from a JSON object that contain only alphabetical characters.
    """
    # Load the JSON data into a Python object
    python_data = json.loads(json_data)
    
    # Convert the Python object to a DataFrame
    df = pd.DataFrame(python_data)
    
    # Identify columns to remove
    cols_to_remove = []
    for col in df.columns:
        if df[col].apply(lambda x: str(x).isalpha()).all():
            cols_to_remove.append(col)
            
    # Drop identified columns
    df_cleaned = df.drop(columns=cols_to_remove)
    
    # Convert the cleaned DataFrame back to JSON
    cleaned_json = df_cleaned.to_json(orient='records')
    
    return cleaned_json

def remove_non_numerical_from_json_automatic(json_data):
    """
    Remove non-numerical characters from columns that contain mixed data types in a JSON object.
        
    Returns:
        str: The cleaned JSON-formatted data.
    """
    # Load the JSON data into a Python object
    python_data = json.loads(json_data)
    
    # Convert the Python object to a DataFrame
    df = pd.DataFrame(python_data)
    
    # Automatically identify columns to clean
    columns_to_clean = []
    for col in df.columns:
        if df[col].apply(lambda x: any(char.isdigit() for char in str(x)) and any(char.isalpha() for char in str(x))).any():
            columns_to_clean.append(col)
    
    # Remove non-numerical characters from identified columns
    for col in columns_to_clean:
        df[col] = df[col].apply(lambda x: float(re.sub('[^0-9.]', '', str(x))))
    
    # Convert the cleaned DataFrame back to JSON
    cleaned_json = df.to_json(orient='records')
    
    return cleaned_json

def process_weather_data(json_data):
    processed_data = []
    for period in json_data['properties']['periods']:
        data_point = {
            "timestamp": period.get("startTime", None),
            "temperature": period.get("temperature", None),
            "temperatureUnit": period.get("temperatureUnit", None),
            "windSpeed": period.get("windSpeed", None),
            "windDirection": period.get("windDirection", None),
            "dewpoint": period.get("dewpoint", None),
            "probabilityOfPrecipitation": period.get("probabilityOfPrecipitation", None),
            "relativeHumidity": period.get("relativeHumidity", None),
        }
        processed_data.append(data_point)
    return remove_alphabetical_columns_from_json(json.dumps(processed_data))


# Load the JSON data from the file (assuming the file name is 'data.json')
with open('data.json', 'r') as infile:
    raw_data = json.load(infile)

# Process the data
processed_data = remove_non_numerical_from_json_automatic(process_weather_data(raw_data))

# Optionally, save the processed data to another JSON file
with open('processed_data.json', 'w') as outfile:
    json.dump(processed_data, outfile)