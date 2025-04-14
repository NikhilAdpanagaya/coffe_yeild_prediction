import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("dataset/coffee_plantation_dataset.csv")

# Define features and target
features = [
    'District', 'Season', 'Rainfall_mm', 'Temperature_C',
    'Soil_Moisture_%', 'Pest_Infestation_%', 'Fertilizer_Usage_kg_per_hectare'
]
target = 'Coffee_Yield_kg_per_hectare'

# Drop missing values
df = df[features + [target]].dropna()

label_encoders = {}
for col in ['District', 'Season']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df[features]
y = np.log1p(df[target])  

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# from math import sqrt

y_pred = rf_model.predict(X_test)
# rmse = sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)
# print(f"RMSE: {rmse:.3f}, R²: {r2:.3f}")

with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("Model, scaler, and label encoders saved!")
