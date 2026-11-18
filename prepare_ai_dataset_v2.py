import pandas as pd

print("AQUAGUARD AI - IMPROVED AI DATASET")
print("=" * 55)

# Load original preprocessed dataset
df = pd.read_csv("lake_biwa_preprocessed.csv")

# Select reliable features
features = [
    "Water_Temperature",
    "pH",
    "DO",
    "COD",
    "Total_Nitrogen",
    "Total_Phosphorus"
]

# Keep date and station information
ai_df = df[
    ["Date", "Location", "Station_Code"] + features
].copy()

# Remove only records missing selected measurements
ai_df = ai_df.dropna(subset=features).copy()

# Save improved dataset
ai_df.to_csv(
    "aquaguard_ai_training_v2.csv",
    index=False
)

print("\nPREPARATION COMPLETED!")

print("Original records:", len(df))
print("AI training records:", len(ai_df))
print("Removed records:", len(df) - len(ai_df))

print("\nSelected features:")
for feature in features:
    print("-", feature)

print("\nUnique monitoring stations:",
      ai_df["Station_Code"].nunique())

print("\nRecords per station:")
print(ai_df.groupby("Station_Code").size().to_string())

print("\nMissing values:")
print(ai_df[features].isnull().sum())

print("\nSaved: aquaguard_ai_training_v2.csv")