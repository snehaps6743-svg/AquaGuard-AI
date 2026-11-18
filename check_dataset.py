import pandas as pd
import zipfile

zip_file = "AquaGuard_AI_English_Datasets.zip"

with zipfile.ZipFile(zip_file, "r") as z:

    for filename in z.namelist():

        if filename.endswith(".csv"):

            print("\n" + "=" * 50)
            print("DATASET:", filename)
            print("=" * 50)

            with z.open(filename) as file:
                df = pd.read_csv(file)

            print("Rows:", len(df))
            print("Columns:", len(df.columns))

            print("\nColumn Names:")
            print(df.columns.tolist())

            print("\nFirst 5 Records:")
            print(df.head())