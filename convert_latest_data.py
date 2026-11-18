import pandas as pd
import unicodedata
from pathlib import Path

# File paths
input_file = Path("data/official_lake_biwa_latest.xlsx")
output_file = Path("data/lake_biwa_latest_water_quality.csv")

# Parameter mapping
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


# Clean Japanese and English labels
def clean_text(value):
    if pd.isna(value):
        return ""

    return "".join(
        unicodedata.normalize("NFKC", str(value)).split()
    ).upper()


# Load Excel workbook
print("Loading Lake Biwa official dataset...")

excel = pd.ExcelFile(input_file)

all_records = []

# Process every sheet
for sheet_name in excel.sheet_names:

    df = pd.read_excel(
        input_file,
        sheet_name=sheet_name,
        header=None
    )

    # Extract dates from row 5
    date_row = pd.to_datetime(df.iloc[5], errors="coerce")

    date_columns = [
        col for col, date in date_row.items()
        if pd.notna(date)
    ]

    if not date_columns:
        continue

    # Extract monitoring location
    location = df.iloc[3, 6]

    if pd.isna(location):
        location = sheet_name

    location = "".join(
        unicodedata.normalize("NFKC", str(location)).split()
    )

    # Find parameter rows
    parameter_rows = {}

    for row in range(8, min(30, len(df))):

        label = clean_text(df.iloc[row, 2])

        if label in parameter_map:
            parameter_rows[parameter_map[label]] = row

    # Extract observations
    for col in date_columns:

        record = {
            "Date": date_row[col],
            "Sheet": sheet_name,
            "Location": location
        }

        for parameter, row in parameter_rows.items():

            value = df.iloc[row, col]

            if pd.isna(value):
                value = None

            record[parameter] = value

        all_records.append(record)


# Create final dataset
latest_df = pd.DataFrame(all_records)

# Sort by date and location
if not latest_df.empty:
    latest_df["Date"] = pd.to_datetime(
        latest_df["Date"],
        errors="coerce"
    )

    latest_df = latest_df.sort_values(
        by=["Date", "Location"]
    )

# Save CSV
output_file.parent.mkdir(parents=True, exist_ok=True)

latest_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

# Display results
print("\n======================================")
print("DATA EXTRACTION COMPLETED!")
print("======================================")

print("Total records:", len(latest_df))
print("Total locations:", latest_df["Location"].nunique())

print("\nAvailable columns:")
print(latest_df.columns.tolist())

print("\nFirst 5 records:")
print(latest_df.head())

print("\nMissing values:")
print(latest_df.isnull().sum())

print("\nSaved file:", output_file)