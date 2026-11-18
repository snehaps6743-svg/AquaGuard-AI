import pandas as pd

df = pd.read_csv("aquaguard_anomaly_results.csv")

print("AQUAGUARD AI - ANOMALY ANALYSIS")
print("=" * 55)

# Count records and anomalies per station
report = df.groupby("Station_Code").agg(
    Total_Records=("Anomaly_Label", "count"),
    Anomalies=("Anomaly_Label", lambda x: (x == -1).sum())
)

report["Anomaly_Percentage"] = (
    report["Anomalies"] / report["Total_Records"] * 100
).round(2)

print("\n1. STATION-WISE ANOMALY REPORT")
print(
    report.sort_values(
        "Anomaly_Percentage",
        ascending=False
    ).to_string()
)

# Overall feature averages
features = [
    "Water_Temperature",
    "pH",
    "DO",
    "COD",
    "Total_Nitrogen",
    "Total_Phosphorus"
]

print("\n2. NORMAL RECORD AVERAGES")
print(
    df[df["Anomaly_Label"] == 1][features].mean()
)

print("\n3. ANOMALOUS RECORD AVERAGES")
print(
    df[df["Anomaly_Label"] == -1][features].mean()
)

print("\nANALYSIS COMPLETED!")