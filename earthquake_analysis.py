import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

# -----------------------------------------
# FILE PATHS
# -----------------------------------------

INPUT_PATH = Path("data/lake_biwa_earthquakes.csv")
OUTPUT_PATH = Path("outputs")

OUTPUT_PATH.mkdir(exist_ok=True)

# -----------------------------------------
# LOAD EARTHQUAKE DATA
# -----------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 55)
print("AQUAGUARD AI - EARTHQUAKE DATA ANALYSIS")
print("=" * 55)

print("\nOriginal records:", len(df))

# -----------------------------------------
# DATA CLEANING
# -----------------------------------------

df["time"] = pd.to_datetime(
    df["time"],
    errors="coerce",
    utc=True
)

numeric_columns = [
    "latitude",
    "longitude",
    "depth",
    "mag"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Remove missing essential values

clean_df = df.dropna(
    subset=[
        "time",
        "latitude",
        "longitude",
        "depth",
        "mag"
    ]
).copy()

# Remove duplicate earthquake IDs

clean_df = clean_df.drop_duplicates(
    subset=["id"]
)

# Sort by date

clean_df = clean_df.sort_values(
    by="time"
)

# -----------------------------------------
# SAVE CLEANED DATASET
# -----------------------------------------

clean_file = OUTPUT_PATH / "lake_biwa_earthquakes_cleaned.csv"

clean_df.to_csv(
    clean_file,
    index=False
)

print("\nCleaned records:", len(clean_df))

print(
    "Removed records:",
    len(df) - len(clean_df)
)

print(
    "\nEarthquake date range:",
    clean_df["time"].min(),
    "to",
    clean_df["time"].max()
)

# -----------------------------------------
# EARTHQUAKE STATISTICS
# -----------------------------------------

print("\nMagnitude Statistics:")

print(clean_df["mag"].describe())

print(
    "\nAverage Magnitude:",
    round(clean_df["mag"].mean(), 2)
)

print(
    "Maximum Magnitude:",
    clean_df["mag"].max()
)

print(
    "Minimum Magnitude:",
    clean_df["mag"].min()
)

# -----------------------------------------
# GRAPH 1: MAGNITUDE DISTRIBUTION
# -----------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(
    clean_df["mag"],
    bins=15,
    edgecolor="black"
)

plt.title("Earthquake Magnitude Distribution")
plt.xlabel("Magnitude")
plt.ylabel("Number of Earthquakes")

plt.grid(axis="y")
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "earthquake_magnitude_distribution.png"
)

plt.show()

print("\nMagnitude graph generated successfully!")

# -----------------------------------------
# GRAPH 2: EARTHQUAKE LOCATION MAP
# -----------------------------------------

fig = px.scatter_geo(
    clean_df,
    lat="latitude",
    lon="longitude",
    color="mag",
    hover_name="place",
    hover_data=[
        "time",
        "depth",
        "mag"
    ],
    scope="asia",
    title="Earthquake Locations Around the Lake Biwa Region",
    color_continuous_scale="Viridis"
)

fig.update_geos(
    center={
        "lat": 35.5,
        "lon": 136.5
    },
    projection_scale=15
)

fig.write_html(
    OUTPUT_PATH / "earthquake_locations.html"
)

print("Earthquake location map generated successfully!")

# -----------------------------------------
# FINAL MESSAGE
# -----------------------------------------

print("\n" + "=" * 55)
print("AQUAGUARD AI - EARTHQUAKE ANALYSIS COMPLETED")
print("=" * 55)

print("\nGenerated Files:")

print("1. lake_biwa_earthquakes_cleaned.csv")
print("2. earthquake_magnitude_distribution.png")
print("3. earthquake_locations.html")

print("\nAll earthquake analysis files saved successfully!")
# -----------------------------------------
# DISTANCE FROM LAKE BIWA
# -----------------------------------------

from math import radians, sin, cos, sqrt, atan2

# Approximate Lake Biwa reference coordinates
lake_lat = 35.3
lake_lon = 136.1

def calculate_distance(lat1, lon1, lat2, lon2):

    R = 6371

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


clean_df["Distance_From_Lake_Biwa_km"] = clean_df.apply(
    lambda row: calculate_distance(
        lake_lat,
        lake_lon,
        row["latitude"],
        row["longitude"]
    ),
    axis=1
)

clean_df.to_csv(
    "outputs/lake_biwa_earthquakes_cleaned.csv",
    index=False
)

print("\nEarthquake Distance Analysis")
print(clean_df[
    ["time", "mag", "depth", "Distance_From_Lake_Biwa_km"]
].head(10))

print("\nClosest earthquake distance:")
print(
    round(
        clean_df["Distance_From_Lake_Biwa_km"].min(),
        2
    ),
    "km"
)

print("\nDistance calculation completed!")
# -----------------------------------------
# FILTER EARTHQUAKES WITHIN 50 KM
# -----------------------------------------

nearby_earthquakes = clean_df[
    clean_df["Distance_From_Lake_Biwa_km"] <= 50
].copy()

nearby_earthquakes.to_csv(
    "outputs/earthquakes_within_50km.csv",
    index=False
)

print("\nEARTHQUAKES WITHIN 50 KM OF LAKE BIWA")
print("=" * 45)

print("Total nearby earthquakes:", len(nearby_earthquakes))

print("\nFirst 10 nearby earthquakes:")

print(
    nearby_earthquakes[
        ["time", "place", "mag", "depth",
         "Distance_From_Lake_Biwa_km"]
    ].head(10).to_string(index=False)
)

print("\nNearby earthquake dataset saved successfully!")
# -----------------------------------------
# CHECK EARTHQUAKE DATA COVERAGE
# -----------------------------------------

print("\nEARTHQUAKE DATA COVERAGE")
print("=" * 45)

print("Total nearby earthquakes:", len(nearby_earthquakes))

print(
    "First recorded date:",
    nearby_earthquakes["time"].min()
)

print(
    "Last recorded date:",
    nearby_earthquakes["time"].max()
)

nearby_earthquakes["Year"] = (
    nearby_earthquakes["time"].dt.year
)

yearly_counts = (
    nearby_earthquakes
    .groupby("Year")
    .size()
    .reset_index(name="Earthquake_Count")
)

print("\nEarthquakes per year:")
print(yearly_counts.to_string(index=False))

yearly_counts.to_csv(
    "outputs/yearly_earthquake_counts.csv",
    index=False
)

print("\nYearly earthquake summary saved successfully!")
# -----------------------------------------
# NEARBY EARTHQUAKE STATISTICS
# -----------------------------------------

print("\nNEARBY EARTHQUAKE ANALYSIS")
print("=" * 45)

print("Total nearby earthquakes:", len(nearby_earthquakes))

print("\nMagnitude statistics:")
print(nearby_earthquakes["mag"].describe())

print("\nDate coverage:")
print("First date:", nearby_earthquakes["time"].min())
print("Last date:", nearby_earthquakes["time"].max())

# Count earthquakes per year
nearby_earthquakes["Year"] = (
    nearby_earthquakes["time"].dt.year
)

yearly_counts = (
    nearby_earthquakes.groupby("Year")
    .size()
    .reset_index(name="Earthquake_Count")
)

yearly_counts.to_csv(
    "outputs/yearly_earthquake_counts.csv",
    index=False
)

print("\nYearly earthquake counts:")
print(yearly_counts.to_string(index=False))

print("\nYearly earthquake dataset saved!")