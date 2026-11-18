import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

print("AQUAGUARD AI - IMPROVED ANOMALY DETECTION")
print("=" * 55)

# Load dataset
df = pd.read_csv("aquaguard_ai_training_v2.csv")

df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month

features = [
    "Water_Temperature",
    "pH",
    "DO",
    "COD",
    "Total_Nitrogen",
    "Total_Phosphorus"
]

# Adjust for station and monthly differences
adjusted = pd.DataFrame(index=df.index)

print("\nAdjusting station and monthly differences...")

for feature in features:

    values = df[feature].astype(float)
    grand_mean = values.median()

    station_effect = pd.Series(
        0.0,
        index=df["Station_Code"].unique()
    )

    month_effect = pd.Series(
        0.0,
        index=range(1, 13)
    )

    for _ in range(15):

        station_effect = (
            values
            - grand_mean
            - df["Month"].map(month_effect)
        ).groupby(df["Station_Code"]).median()

        month_effect = (
            values
            - grand_mean
            - df["Station_Code"].map(station_effect)
        ).groupby(df["Month"]).median()

    expected_values = (
        grand_mean
        + df["Station_Code"].map(station_effect)
        + df["Month"].map(month_effect)
    )

    adjusted[feature] = values - expected_values

# Scale adjusted data
scaler = RobustScaler()
X_scaled = scaler.fit_transform(adjusted)

# Train Isolation Forest
model = IsolationForest(
    n_estimators=300,
    contamination=0.05,
    random_state=42
)

df["Anomaly_Label"] = model.fit_predict(X_scaled)

df["Anomaly_Score"] = model.decision_function(X_scaled)

df["Status"] = df["Anomaly_Label"].map({
    1: "Normal",
    -1: "Potential Anomaly"
})

# Save improved results separately
df.to_csv(
    "aquaguard_anomaly_results_v2.csv",
    index=False,
    encoding="utf-8-sig"
)

# Display results
print("\nMODEL TRAINING COMPLETED!")
print("-" * 40)

print("Total records:", len(df))

print("\nDetection Results:")
print(df["Status"].value_counts())

print("\nPotential Anomalies by Station:")

anomalies = df[df["Status"] == "Potential Anomaly"]

print(
    anomalies.groupby("Station_Code")
    .size()
    .sort_values(ascending=False)
)

print("\nPotential Anomaly Records:")

print(
    anomalies[
        ["Date", "Station_Code", "Anomaly_Score"]
    ].to_string(index=False)
)

print("\nNew results saved successfully!")
print("File: aquaguard_anomaly_results_v2.csv")