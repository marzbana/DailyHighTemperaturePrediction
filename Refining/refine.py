from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np

# Load model
data = pd.read_csv("../DataSets/weather_mateo_aqi_ocean_tide_train.csv")

# Select features
features = ['month','day','year', 'avg_temp', 'low_temp', 'HDD', 'CDD', 
            'precipitation', 'windspeed_10m_max_(mp/h)', 'shortwave_radiation_sum_(MJ/m²)',
            'et0_fao_evapotranspiration_(mm)', 'AQI', 'average_ocean_temp', 'tide_height(ft)']

target = 'high_temp'

X = data[features][:-1]  # Drop the last row of features
y = data[target].shift(-1)[:-1]  # Shift the target up by one row, then drop the last row
#make sure all the values are floats
X = X.astype(float)
y = y.astype(float)

X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

#encode the month to sin(2pi*month/12) and cos(2pi*month/12)
X['Month_sin'] = np.sin(2 * np.pi * X['month']/12)
X['Month_cos'] = np.cos(2 * np.pi * X['month']/12)
X.drop('month', axis=1, inplace=True)

#encode the day to sin(2pi*day/31) and cos(2pi*day/31)
X['Day_sin'] = np.sin(2 * np.pi * X['day']/31)
X['Day_cos'] = np.cos(2 * np.pi * X['day']/31)
X.drop('day', axis=1, inplace=True)

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
X = scaler.fit_transform(X)

print("Mean of scaled features:", np.mean(X, axis=0))
print("Std of scaled features:", np.std(X, axis=0))
print("Check NaN in features:", np.isnan(X).sum())
print("Check NaN in target:", y.isna().sum())

# Create training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Neural Network architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # Linear activation function for regression

# Compile the model with additional metrics
model.compile(optimizer='adam', loss='mean_squared_error',
              metrics=[RootMeanSquaredError(name='rmse'), MeanAbsoluteError(name='mae')])

# Train the model (assuming you've already defined X_train, y_train, etc.)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
loss, rmse, mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test RMSE: {rmse}')
print(f'Test MAE: {mae}')

