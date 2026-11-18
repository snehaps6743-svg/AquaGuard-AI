import pandas as pd

original = pd.read_csv("lake_biwa_all_locations.csv")
cleaned = pd.read_csv("lake_biwa_preprocessed.csv")

columns = [
    "Transparency",
    "DO",
    "BOD",
    "SS",
    "Coliform_Count"
]

print("INVESTIGATING CONVERSION RESULTS")
print("=" * 50)

for column in columns:

    print(f"\n{column}")

    raw = original[column].astype("string").str.strip()
    converted = pd.to_numeric(raw, errors="coerce")

    unusual = raw[raw.notna() & converted.isna()]

    print("Values that could not be converted:")
    print(unusual.value_counts().head(20))

    print("Total:", len(unusual))

print("\nINVESTIGATION COMPLETED!")