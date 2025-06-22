# SECTION 10: Iterative Design and Optimization (with Graphs)

import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ✅ Sample dataset (21 columns: 5 base + 16 sensors)
data = """
1 1 -0.0007 0.0003 100.0 518.67 641.82 1587.54 1400.6 14.62 21.61 553.67 2388.06 9066.54 1.3 47.47 521.66 2388.74 8138.17 8.4195 0.03
1 2 0.0019 -0.0003 100.0 518.67 642.15 1588.45 1400.6 14.62 21.61 546.45 2388.06 9066.54 1.3 47.49 522.28 2388.74 8138.17 8.4318 0.03
1 3 -0.0043 -0.0004 100.0 518.67 642.35 1589.07 1400.6 14.62 21.61 544.96 2388.06 9066.54 1.3 47.49 522.42 2388.74 8138.17 8.4171 0.03
2 1 -0.0016 0.0006 100.0 518.67 642.15 1588.68 1400.6 14.62 21.61 553.13 2388.06 9066.54 1.3 47.55 523.44 2388.74 8138.17 8.4224 0.03
2 2 -0.0032 -0.0005 100.0 518.67 642.35 1589.3 1400.6 14.62 21.61 550.77 2388.06 9066.54 1.3 47.53 523.4 2388.74 8138.17 8.4355 0.03
"""

# Load dataset
df = pd.read_csv(StringIO(data), sep='\s+', header=None)
df.columns = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor_{i}' for i in range(1, 17)]

# Calculate RUL
rul_df = df.groupby('engine_id')['cycle'].max().reset_index()
rul_df.columns = ['engine_id', 'max_cycle']
df = df.merge(rul_df, on='engine_id')
df['RUL'] = df['max_cycle'] - df['cycle']
df.drop('max_cycle', axis=1, inplace=True)

# Feature Engineering
for sensor in ['sensor_2', 'sensor_3', 'sensor_4']:
    df[f'{sensor}_roll_mean'] = df.groupby('engine_id')[sensor].rolling(window=2, min_periods=1).mean().reset_index(level=0, drop=True)
    df[f'{sensor}_diff'] = df.groupby('engine_id')[sensor].diff().fillna(0)

# Select features and target
features = ['cycle', 'op1', 'op2', 'op3'] + [f'sensor_{i}' for i in range(1, 17)]
features += [col for col in df.columns if 'roll_mean' in col or 'diff' in col]
X = df[features]
y = df['RUL']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaleiid, y, test_size=0.4, random_state=42)

# Random Forest with GridSearch
param_grid = {
    'n_estimators': [10],
    'max_depth': [5, None],
    'min_samples_split': [2]
}
model = RandomForestRegressor(random_state=42)
grid = GridSearchCV(model, param_grid, cv=2, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

# Best model & predictions
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print("✅ Model Training Complete")
print("Best Parameters:", grid.best_params_)
print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAE: {mae:.2f}")

# 📊 Plot Actual vs Predicted RUL
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted RUL")
plt.grid(True)
plt.tight_layout()
plt.show()

# 📈 Feature Importance Plot
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), np.array(features)[indices], rotation=90)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()
