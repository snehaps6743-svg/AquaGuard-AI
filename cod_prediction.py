import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data/lake_biwa_water_quality_combined.csv")

data = data.sort_values("Year").reset_index(drop=True)

# Select COD column
df = data[["Year", "COD_North_Basin_mg_L"]].copy()

# Create previous-year features
df["COD_Lag1"] = df["COD_North_Basin_mg_L"].shift(1)
df["COD_Lag2"] = df["COD_North_Basin_mg_L"].shift(2)
df["COD_Lag3"] = df["COD_North_Basin_mg_L"].shift(3)

df = df.dropna().reset_index(drop=True)

X = df[["COD_Lag1", "COD_Lag2", "COD_Lag3"]]
y = df["COD_North_Basin_mg_L"]

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=3, random_state=42
    ),
    "Dummy Regressor": DummyRegressor(strategy="mean")
}

results = {name: [] for name in models}
actual_values = []
persistence_values = []
prediction_years = []

# Walk-forward validation
minimum_training_size = 15

for i in range(minimum_training_size, len(X)):

    X_train = X.iloc[:i]
    y_train = y.iloc[:i]

    X_test = X.iloc[[i]]
    actual = y.iloc[i]

    actual_values.append(actual)
    prediction_years.append(df["Year"].iloc[i])

    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)[0]
        results[name].append(prediction)

    # Persistence baseline: predict using previous year's COD
    persistence_values.append(X_test["COD_Lag1"].iloc[0])

# Calculate performance
print("\nAQUAGUARD AI - WALK-FORWARD VALIDATION")
print("=" * 65)

actual_values = np.array(actual_values)

for name, predictions in results.items():

    predictions = np.array(predictions)

    mae = mean_absolute_error(actual_values, predictions)
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))
    r2 = r2_score(actual_values, predictions)

    print(f"\n{name}")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

# Persistence baseline metrics
persistence_values = np.array(persistence_values)

print("\nPERSISTENCE BASELINE")
print("MAE:", round(mean_absolute_error(actual_values, persistence_values), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(actual_values, persistence_values)), 4))
print("R2 Score:", round(r2_score(actual_values, persistence_values), 4))

print("\nVALIDATION YEARS:", prediction_years)
print("\nTotal predictions:", len(actual_values))
# Save walk-forward validation results

import os

validation_output = pd.DataFrame({
    "Year": prediction_years,
    "Actual_COD": actual_values,
    "Linear_Regression": results["Linear Regression"],
    "Random_Forest": results["Random Forest"],
    "Dummy_Regressor": results["Dummy Regressor"],
    "Persistence_Baseline": persistence_values
})

os.makedirs("outputs", exist_ok=True)

validation_output.to_csv(
    "outputs/walk_forward_results.csv",
    index=False
)

print("\nValidation results saved successfully!")
print("File: outputs/walk_forward_results.csv")
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.plot(
    prediction_years,
    actual_values,
    marker="o",
    label="Actual COD"
)

plt.plot(
    prediction_years,
    results["Random Forest"],
    marker="s",
    label="Random Forest Predicted COD"
)

plt.xlabel("Year")
plt.ylabel("COD (mg/L)")
plt.title("AquaGuard AI - Actual vs Predicted COD")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/cod_prediction_validation.png", dpi=300)

plt.show()

print("Prediction graph saved successfully!")
