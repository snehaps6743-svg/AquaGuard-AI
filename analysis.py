import pandas as pd
from pathlib import Path

print("=" * 55)
print("AQUAGUARD AI - DATASET INTEGRATION")
print("=" * 55)

# Load water quality dataset
water_df = pd.read_csv(
    "data/lake_biwa_water_quality_combined.csv"
)

# Load yearly earthquake counts
earthquake_df = pd.read_csv(
    "outputs/yearly_earthquake_counts.csv"
)

# Ensure year columns have the same data type
water_df["Year"] = pd.to_numeric(
    water_df["Year"], errors="coerce"
)

earthquake_df["Year"] = pd.to_numeric(
    earthquake_df["Year"], errors="coerce"
)

# Merge datasets using Year
combined_df = pd.merge(
    water_df,
    earthquake_df,
    on="Year",
    how="left"
)

# Missing earthquake counts mean no matching
# earthquake record was found in the supplied dataset.
combined_df["Earthquake_Count"] = (
    combined_df["Earthquake_Count"].fillna(0)
)

# Save combined dataset
combined_df.to_csv(
    "outputs/aquaguard_combined_dataset.csv",
    index=False
)

print("\nCombined dataset shape:", combined_df.shape)

print("\nFirst five records:")
print(combined_df.head())

print("\nEarthquake count by year:")
print(
    combined_df[
        ["Year", "Earthquake_Count"]
    ].to_string(index=False)
)

print("\nCombined dataset saved successfully!")
import pandas as pd
import matplotlib.pyplot as plt

# Load combined dataset
df = pd.read_csv(
    "outputs/aquaguard_combined_dataset.csv"
)

# Plot earthquake activity
plt.figure(figsize=(12, 5))

plt.bar(
    df["Year"],
    df["Earthquake_Count"],
    label="Earthquake Count"
)

plt.title("Lake Biwa - Annual Nearby Earthquake Activity")
plt.xlabel("Year")
plt.ylabel("Number of Earthquakes")
plt.legend()
plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    "outputs/annual_earthquake_activity.png"
)

plt.show()

print("Earthquake activity graph generated successfully!")

# Plot COD water quality trend
plt.figure(figsize=(12, 5))

plt.plot(
    df["Year"],
    df["COD_North_Basin_mg_L"],
    label="North Basin"
)

plt.plot(
    df["Year"],
    df["COD_South_Basin_mg_L"],
    label="South Basin"
)

plt.title("Lake Biwa - COD Water Quality Trend")
plt.xlabel("Year")
plt.ylabel("COD (mg/L)")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/combined_cod_trend.png"
)

plt.show()

print("COD comparison graph generated successfully!")