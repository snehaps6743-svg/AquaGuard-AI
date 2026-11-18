import pandas as pd

file = "lake_biwa_all_locations.csv"

df = pd.read_csv(file)

print("AQUAGUARD AI - DATASET VALIDATION")
print("=" * 50)

print("\n1. DATASET SIZE")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n2. UNIQUE MONITORING LOCATIONS")
print(df["Location"].nunique())

print("\n3. DATE RANGE")
print("Start:", df["Date"].min())
print("End:", df["Date"].max())

print("\n4. MISSING VALUES")
print(df.isnull().sum())

print("\n5. DUPLICATE RECORDS")
print(df.duplicated(
    subset=["Date", "Station_Code"]
).sum())

print("\n6. BELOW-DETECTION READINGS")

for column in df.columns:
    if df[column].dtype == "object":
        count = df[column].astype(str).str.startswith("<").sum()

        if count > 0:
            print(column, ":", count)

print("\n7. DATA TYPES")
print(df.dtypes)

print("\nVALIDATION COMPLETED!")
print("\n8. INSPECTING NON-NUMERIC VALUES")

columns = [
    "Transparency",
    "DO",
    "BOD",
    "SS",
    "Coliform_Count"
]

for column in columns:
    print(f"\n{column}:")
    values = df[column].dropna().astype(str).unique()
    print(values[:20])

    below_limit = df[column].astype(str).str.contains(
        "<", regex=False
    ).sum()

    print("Below-detection readings:", below_limit)