import pandas as pd

df = pd.read_csv("lake_biwa_preprocessed.csv")

stations = ["001-51", "501-76", "501-77"]

features = [
    "Water_Temperature",
    "pH",
    "DO",
    "Transparency",
    "COD",
    "Total_Nitrogen",
    "Total_Phosphorus"
]

for station in stations:

    print("\n" + "=" * 60)
    print("STATION:", station)
    print("=" * 60)

    data = df[df["Station_Code"] == station].copy()

    print("Total records:", len(data))

    print("\nTransparency original readings:")
    print(data["Transparency_Raw"].to_string(index=False))

    print("\nTransparency above-bound count:")
    print(data["Transparency_Above_Bound"].sum())

    print("\nMissing measurements:")
    print(data[features].isnull().sum())

    print("\nComplete measurement details:")
    print(data[
        ["Date", "Transparency_Raw"] + features
    ].to_string(index=False))

print("\nINSPECTION COMPLETED!")