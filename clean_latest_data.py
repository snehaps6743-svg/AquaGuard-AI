import pandas as pd
from pathlib import Path

input_file = Path("data/lake_biwa_latest_water_quality.csv")
output_file = Path("data/lake_biwa_latest_cleaned.csv")

df = pd.read_csv(input_file)

# Convert dates
df["Date"] = pd.to_datetime(df["Date"])

# Preserve original readings and create numeric columns
columns = [
    "Transparency",
    "DO",
    "BOD",
    "COD",
    "SS",
    "Coliform_Count"
]

for col in columns:

    df[col + "_Original"] = df[col].astype("string")

    # Convert only exact numeric readings.
    # Special readings such as <0.5 and >2.5 become NaN.
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Save cleaned dataset
df.to_csv(output_file, index=False)

print("SCIENTIFIC DATA PREPARATION COMPLETED!")
print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df[columns].isnull().sum())

print("\nSaved:", output_file)