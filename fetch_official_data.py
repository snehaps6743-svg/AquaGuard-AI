import requests
import pandas as pd
import re
from io import StringIO
from pathlib import Path

URL = "https://www.pref.shiga.lg.jp/dk00/2922.html"

OUTPUT_FILE = Path("data/lake_biwa_official_online.csv")


def fetch_official_data():

    print("Connecting to Shiga Prefecture website...")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        URL,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()
    response.encoding = "utf-8"

    # Extract survey date
    match = re.search(
        r"調査日\s*[:：]\s*令和(\d+)年(\d+)月(\d+)日",
        response.text
    )

    if not match:
        raise ValueError("Survey date not found.")

    year = 2018 + int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))

    observation_date = (
        f"{year}-{month:02d}-{day:02d}"
    )

    # Extract official tables
    tables = pd.read_html(
        StringIO(response.text)
    )

    df = tables[0].copy()

    # Fill missing area and location names
    df["エリア大区分"] = df["エリア大区分"].ffill()
    df["エリア中区分"] = df["エリア中区分"].ffill()

    # Rename columns into English
    df = df.rename(columns={
        "エリア大区分": "Lake_Area",
        "エリア中区分": "Monitoring_Location",
        "エリア小区分": "Depth",
        "水温（℃）": "Water_Temperature_C",
        "透明度（m）": "Transparency_m",
        "溶存酸素（mg/L）": "Dissolved_Oxygen_mg_L"
    })

    df.insert(
        0,
        "Observation_Date",
        observation_date
    )

    # Save separately without changing historical data
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDATA RETRIEVAL SUCCESSFUL!")
    print("Survey date:", observation_date)
    print("Records retrieved:", len(df))
    print("Saved to:", OUTPUT_FILE)

    print("\nRetrieved observations:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    fetch_official_data()