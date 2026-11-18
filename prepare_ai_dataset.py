import pandas as pd

print("AQUAGUARD AI - AI DATASET PREPARATION")
print("=" * 50)

# Load processed dataset
df = pd.read_csv("lake_biwa_preprocessed.csv")

# Select initial AI features
features = [
    "Water_Temperature",
    "pH",
    "DO",
    "Transparency",
    "COD",
    "Total_Nitrogen",
    "Total_Phosphorus"
]

# Preserve station and date information
ai_df = df[
    ["Date", "Location", "Station_Code"] + features
].copy()

# Exclude transparency readings recorded as > values
ai_df = ai_df[
    ~df["Transparency_Above_Bound"].fillna(False)
]

# Remove rows with missing selected measurements
ai_df = ai_df.dropna(subset=features)

# Save AI training dataset
ai_df.to_csv(
    "aquaguard_ai_training.csv",
    index=False
)

print("\nAI DATASET PREPARATION COMPLETED!")

print("Original records:", len(df))
print("AI training records:", len(ai_df))
print("Removed records:", len(df) - len(ai_df))

print("\nSelected features:")
print(features)

print("\nFinal dataset shape:", ai_df.shape)

print("\nMissing values:")
print(ai_df[features].isnull().sum())

print("\nSaved: aquaguard_ai_training.csv")