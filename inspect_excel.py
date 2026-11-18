import pandas as pd

df = pd.read_excel("5584973.xlsx", sheet_name="18A", header=None)

print("PARAMETER LABEL INSPECTION")
print("=" * 50)

for row in range(8, 25):
    print(f"Row {row}:")

    for col in range(5):
        value = df.iloc[row, col]

        if pd.notna(value):
            print(f"  Column {col}: {value}")