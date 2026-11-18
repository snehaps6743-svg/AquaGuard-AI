import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load water quality dataset
df = pd.read_csv("data/lake_biwa_water_quality_combined.csv")

# Select numerical water quality columns
features = df.drop(columns=["Year", "Japanese_Era"], errors="ignore")
features = features.select_dtypes(include="number")

# Handle missing values
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(features)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest model
model = IsolationForest(
    contamination=0.1,
    random_state=42
)

model.fit(X_scaled)

# Detect anomalies
df["Anomaly"] = model.predict(X_scaled)

# Convert predictions into readable labels
df["Status"] = df["Anomaly"].map({
    1: "Normal",
    -1: "Unusual"
})

# Display results
print("\nAQUAGUARD AI - ANOMALY DETECTION")
print("=" * 45)

print(df[["Year", "Anomaly", "Status"]].to_string(index=False))

print("\nAnomaly Summary:")
print(df["Status"].value_counts())

# Save results
df.to_csv("outputs/aquaguard_anomaly_results.csv", index=False)

print("\nResults saved successfully!")