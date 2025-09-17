import pandas as pd
import numpy as np
import json
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Load dataset
df = pd.read_csv('simulated_tower_power_data.csv')

# Features and target
X_gen = df[['power_source', 'load_kw', 'phcn_uptime', 'solar_output', 'temperature']]
y_gen = df['gen_run_hours']

# Define categorical and numerical features
categorical_features = ['power_source']
numerical_features = ['load_kw', 'phcn_uptime', 'solar_output', 'temperature']

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', 'passthrough', numerical_features)
])

# Final pipeline with RandomForestRegressor
gen_runtime_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_estimators=80))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_gen, y_gen, test_size=0.2, random_state=42)

# Grid search for hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [50, 80, 100],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(gen_runtime_model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model and params
best_model = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)

# Save best model
joblib.dump(best_model, 'gen_runtime_model_best.pkl')

# Predict and evaluate with the best model
train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)


# Save metrics
metrics = {
    "train_mse": round(train_mse, 2),
    "test_mse": round(test_mse, 2),
    "train_rmse": round(train_rmse, 2),
    "test_rmse": round(test_rmse, 2)
}

with open('model_metrics.json_1', 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"Train MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print("Best tuned model saved as 'gen_runtime_model_best.pkl'")
print("Metrics saved to 'model_metrics.json'")
