import pandas as pd

df = pd.read_csv("lake_biwa_preprocessed.csv")

print("AQUAGUARD AI - DATASET QUALITY CHECK")
print("=" * 50)

# 1. Dataset size
print("\n1. DATASET SIZE")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# 2. Date range
df["Date"] = pd.to_datetime(df["Date"])

print("\n2. DATE RANGE")
print("Start:", df["Date"].min())
print("End:", df["Date"].max())

# 3. Unique locations
print("\n3. MONITORING LOCATIONS")
print("Unique locations:", df["Location"].nunique())

# 4. Records per station
print("\n4. RECORDS PER STATION")
print(df.groupby("Station_Code").size().to_string())

# 5. Duplicate records
print("\n5. DUPLICATE CHECK")
print(
    "Duplicate Date + Station:",
    df.duplicated(subset=["Date", "Station_Code"]).sum()
)

# 6. Missing values
print("\n6. MISSING VALUES")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nQUALITY CHECK COMPLETED!")