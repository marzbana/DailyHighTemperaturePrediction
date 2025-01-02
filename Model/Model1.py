import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam

# Load data
data = pd.read_csv("../DataSets/weather_mateo_aqi_ocean_tide_train.csv")

# Select features
features = ['days', 'avg_temp', 'low_temp', 'HDD', 'CDD', 
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


# Standardize the features
scaler = StandardScaler()
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

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Sample Predictions:", model.predict(X_test[:10]))
print(f'Test Loss: {loss}')
model.save('Model1.keras')

# To make a prediction for the next day, you can do:
# next_day_features = np.array([feature1, feature2, ...])  # Replace with the actual feature values
# next_day_features = scaler.transform([next_day_features])  # Note the extra brackets
# prediction = model.predict(next_day_features)
# print(f'Predicted High Temp for Next Day: {prediction[0][0]}')
