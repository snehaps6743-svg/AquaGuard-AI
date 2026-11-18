import pandas as pd
from pathlib import Path

print("=" * 55)
print("AQUAGUARD AI - CORRELATION ANALYSIS")
print("=" * 55)

# Load datasets
water_df = pd.read_csv(
    "data/lake_biwa_water_quality_combined.csv"
)

earthquake_df = pd.read_csv(
    "outputs/earthquakes_within_50km.csv"
)

# Extract year from earthquake dates
earthquake_df["time"] = pd.to_datetime(
    earthquake_df["time"],
    errors="coerce",
    utc=True
)

earthquake_df["Year"] = earthquake_df["time"].dt.year

# Count nearby earthquakes per year
yearly_earthquakes = (
    earthquake_df.groupby("Year")
    .size()
    .reset_index(name="Earthquake_Count")
)

# Merge only years represented in both datasets
combined = pd.merge(
    water_df,
    yearly_earthquakes,
    on="Year",
    how="inner"
)

print("\nEarthquake records:", len(earthquake_df))
print("Years with earthquake records:", len(yearly_earthquakes))
print("Overlapping years:", len(combined))

if len(combined) >= 3:

    # Calculate correlations
    north_corr = combined[
        ["Earthquake_Count", "COD_North_Basin_mg_L"]
    ].corr().iloc[0, 1]

    south_corr = combined[
        ["Earthquake_Count", "COD_South_Basin_mg_L"]
    ].corr().iloc[0, 1]

    print("\nCORRELATION RESULTS")
    print("-" * 40)

    print("Earthquake vs North Basin COD:", round(north_corr, 3))
    print("Earthquake vs South Basin COD:", round(south_corr, 3))

    combined.to_csv(
        "outputs/correlation_dataset.csv",
        index=False
    )

    print("\nCorrelation dataset saved!")

else:
    print("\nNot enough overlapping years for correlation analysis.")
    print("Please check the earthquake dataset date coverage.")

print("\nAnalysis completed!")
import matplotlib.pyplot as plt

# North Basin correlation graph
plt.figure(figsize=(8, 5))

plt.scatter(
    combined["Earthquake_Count"],
    combined["COD_North_Basin_mg_L"]
)

plt.title("Earthquake Activity vs North Basin COD")
plt.xlabel("Nearby Earthquake Count")
plt.ylabel("North Basin COD (mg/L)")
plt.grid(True)

plt.tight_layout()
plt.savefig("outputs/north_basin_correlation.png")
plt.show()


# South Basin correlation graph
plt.figure(figsize=(8, 5))

plt.scatter(
    combined["Earthquake_Count"],
    combined["COD_South_Basin_mg_L"]
)

plt.title("Earthquake Activity vs South Basin COD")
plt.xlabel("Nearby Earthquake Count")
plt.ylabel("South Basin COD (mg/L)")
plt.grid(True)

plt.tight_layout()
plt.savefig("outputs/south_basin_correlation.png")
plt.show()

print("Both correlation graphs generated successfully!")