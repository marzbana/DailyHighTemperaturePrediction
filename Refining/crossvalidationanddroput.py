# Import libraries
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import numpy as np

# Load model
data = pd.read_csv("../DataSets/weather_mateo_aqi_ocean_tide_train.csv")

# Select features
features = ['month','day','year', 'high_temp', 'avg_temp', 'low_temp', 'HDD', 'CDD', 
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
cols_to_standardize = ['avg_temp', 'high_temp','low_temp', 'HDD', 'CDD', 
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


# Function to create a model
def create_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.001))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=[RootMeanSquaredError(name='rmse'), MeanAbsoluteError(name='mae')])
    return model

# Create training, validation, and test sets (your code for this remains unchanged)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize variables for 5-fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_scores = []

# Loop through the k-folds
for train_index, val_index in kf.split(X_train):
    train_X, val_X = X_train[train_index], X_train[val_index]
    train_y, val_y = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model = create_model()
    model.fit(train_X, train_y, epochs=100, batch_size=32, verbose=0)
    
    # Evaluate the model on the validation data
    loss, rmse, mae = model.evaluate(val_X, val_y, verbose=0)
    
    cv_scores.append(loss)

# Output cross-validation results
print(f"Cross-Validation Loss: {np.mean(cv_scores):.4f} ({np.std(cv_scores):.4f})")

# Train the final model on the entire training set
final_model = create_model()
final_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Evaluate the model on the test set
loss, rmse, mae = final_model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss}')
print(f'Test RMSE: {rmse}')
print(f'Test MAE: {mae}')

# Save the model
final_model.save('best_model.keras')

#'month','day','year', 'high_temp', 'avg_temp', 'low_temp', 'HDD', 'CDD', 'precipitation', 'windspeed_10m_max_(mp/h)', 'shortwave_radiation_sum_(MJ/m²)','et0_fao_evapotranspiration_(mm)', 'AQI', 'average_ocean_temp', 'tide_height(ft)'
sd = pd.DataFrame([[9, 26, 2023, 91, 84.5, 78, 0, 20, .12, 14.3, 17.35, 3.6, 28, 84.92, 1.0624583333333335]], columns=features)

# Encode the month and day
sd['Month_sin'] = np.sin(2 * np.pi * sd['month'] / 12)
sd['Month_cos'] = np.cos(2 * np.pi * sd['month'] / 12)
sd.drop('month', axis=1, inplace=True)

sd['Day_sin'] = np.sin(2 * np.pi * sd['day'] / 31)
sd['Day_cos'] = np.cos(2 * np.pi * sd['day'] / 31)
sd.drop('day', axis=1, inplace=True)

# Standardize the features (Note: Use the same scaler you used for training)
# Here, I'm assuming you've already loaded or instantiated a Scaler object named 'scaler'
sd_scaled = scaler.transform(sd)

# Make the prediction (Use the loaded model)
predicted_temperature = model.predict(sd_scaled)

# Uncomment the following line after you've handled the scaling and model loading
print(f"The predicted high temperature for the next day is: {predicted_temperature[0][0]}")