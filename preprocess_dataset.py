import pandas as pd
import re

df = pd.read_csv("lake_biwa_all_locations.csv")

print("AQUAGUARD AI - IMPROVED PREPROCESSING")
print("=" * 50)

df["Date"] = pd.to_datetime(df["Date"])

df["Location"] = (
    df["Location"].astype(str).str.replace(r"\s+", "", regex=True)
)

measurement_columns = [
    "Air_Temperature",
    "Water_Temperature",
    "Transparency",
    "pH",
    "DO",
    "BOD",
    "COD",
    "SS",
    "Coliform_Count",
    "Total_Nitrogen",
    "Total_Phosphorus"
]

for column in measurement_columns:

    raw = df[column].astype("string").str.strip()

    # Preserve original readings
    df[column + "_Raw"] = raw

    # Identify measurement qualifiers
    df[column + "_Below_Detection"] = (
        raw.str.startswith("<", na=False)
    )

    df[column + "_Above_Bound"] = (
        raw.str.startswith(">", na=False)
    )

    # Extract numeric threshold/value
    extracted = raw.str.extract(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
        expand=False
    )

    df[column] = pd.to_numeric(extracted, errors="coerce")

# Save improved dataset
df.to_csv(
    "lake_biwa_preprocessed.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nPREPROCESSING COMPLETED!")
print("Total records:", len(df))
print("Unique locations:", df["Location"].nunique())

print("\nMissing numeric values:")
print(df[measurement_columns].isnull().sum())

print("\nBelow-detection readings:")
for column in measurement_columns:
    count = df[column + "_Below_Detection"].sum()
    if count > 0:
        print(column, ":", count)

print("\nAbove-bound readings:")
for column in measurement_columns:
    count = df[column + "_Above_Bound"].sum()
    if count > 0:
        print(column, ":", count)

print("\nSaved: lake_biwa_preprocessed.csv")