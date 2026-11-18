import pandas as pd
import unicodedata
import re

file = "5584973.xlsx"

excel = pd.ExcelFile(file)

def clean_label(value):
    value = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", "", value).upper()

parameter_map = {
    "気温": "Air_Temperature",
    "水温": "Water_Temperature",
    "透明度": "Transparency",
    "PH": "pH",
    "DO": "DO",
    "BOD": "BOD",
    "COD": "COD",
    "SS": "SS",
    "大腸菌数": "Coliform_Count",
    "全窒素": "Total_Nitrogen",
    "全りん": "Total_Phosphorus"
}

records = []

for sheet in excel.sheet_names:

    df = pd.read_excel(file, sheet_name=sheet, header=None)

    location = str(df.iloc[3, 6]).strip()
    station_code = str(df.iloc[4, 2]).strip()

    for col in range(5, df.shape[1]):

        date = pd.to_datetime(df.iloc[5, col], errors="coerce")

        if pd.isna(date):
            continue

        record = {
            "Date": date,
            "Location": location,
            "Station_Code": station_code
        }

        for row in range(8, df.shape[0]):

            label = clean_label(df.iloc[row, 2])

            if label in parameter_map:

                english_name = parameter_map[label]
                value = df.iloc[row, col]

                if pd.notna(value):
                    record[english_name] = value

        records.append(record)

result = pd.DataFrame(records)

result.to_csv(
    "lake_biwa_all_locations.csv",
    index=False,
    encoding="utf-8-sig"
)

print("EXTRACTION COMPLETED!")
print("=" * 50)
print("Total records:", len(result))
print("Total locations:", result["Location"].nunique())
print("\nExtracted parameters:")
print(result.columns.tolist())

print("\nMissing values:")
print(result.isnull().sum())

print("\nFirst 10 records:")
print(result.head(10))