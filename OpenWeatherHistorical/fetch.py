import requests
import time
import json

# Your API key
api_key = "cc4832bb8665ca60ff635f32c6052b07"

# Miami latitude and longitude
lat, lon = 25.7617, -80.1918

# Unix timestamp for one year ago and now
start_time = int(time.time()) - (2*365 * 24 * 60 * 60)  # 4 years ago
end_time = int(time.time())  # Current time

# Initialize an empty list to store all the data
all_data = []

# Loop through each week in the past year
for week_start in range(start_time, end_time, 7 * 24 * 60 * 60):
    week_end = week_start + (7 * 24 * 60 * 60)  # One week later
    
    # Construct the API URL
    url = f"https://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={week_start}&end={week_end}&appid={api_key}"
    
    # Make the API call
    response = requests.get(url)
    
    if response.status_code == 200:
        # Append the data for this week to all_data
        all_data.extend(response.json()['list'])
    else:
        print(f"Failed to get data for week starting {week_start}")
    
    # Wait for 1 second to respect the rate limit
    time.sleep(1)

# Save all the data to a JSON file
with open('miami_weather_data_4years.json', 'w') as f:
    json.dump(all_data, f)
