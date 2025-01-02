from tensorflow import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Load the saved model
loaded_model = keras.models.load_model('../Refining/best_model.keras')

 
# Create a new sample data point (Make sure to preprocess this the same way as your training data)
# This is just a demonstration; replace these values with the actual feature values you want to predict
sd= np.array([[9, 24, 2023, 81.5, 72, 0, 17, .4, 10.4, 21.54, 4.37, 34, 85, 1.0590783410138251]])

#encode the month to sin(2pi*month/12) and cos(2pi*month/12)
sd['Month_sin'] = np.sin(2 * np.pi * sd['month']/12)
sd['Month_cos'] = np.cos(2 * np.pi * sd['month']/12)
sd.drop('month', axis=1, inplace=True)

#encode the day to sin(2pi*day/31) and cos(2pi*day/31)
sd['Day_sin'] = np.sin(2 * np.pi * sd['day']/31)
sd['Day_cos'] = np.cos(2 * np.pi * sd['day']/31)
sd.drop('day', axis=1, inplace=True)

# Standardize the features
cols_to_standardize = ['avg_temp', 'low_temp', 'HDD', 'CDD', 
            'precipitation', 'windspeed_10m_max_(mp/h)', 'shortwave_radiation_sum_(MJ/m²)',
            'et0_fao_evapotranspiration_(mm)', 'AQI', 'average_ocean_temp', 'tide_height(ft)']
cols_to_leave = ['Month_sin', 'year', 'Month_cos', 'Day_sin', 'Day_cos']
scaler =  ColumnTransformer(
    transformers=[
        ('standardize', StandardScaler(), cols_to_standardize),
        ('leave', 'passthrough', cols_to_leave)
    ])
# Don't forget to scale the new sample using the same scaler object that you used for the training data
sample_data_scaled = scaler.transform(sd)

# Make the prediction
predicted_temperature = loaded_model.predict(sample_data_scaled)

print(f"The predicted high temperature for the next day is: {predicted_temperature[0][0]}")
