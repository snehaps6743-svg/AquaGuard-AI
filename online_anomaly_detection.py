import pandas as pd
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


# File paths
HISTORICAL_FILE = Path("aquaguard_ai_training.csv")

OFFICIAL_FILE = Path("data/lake_biwa_official_online.csv")

OUTPUT_FILE = Path("outputs/official_online_anomaly_results.csv")


# Matching water-quality features
HISTORICAL_FEATURES = [
    "Water_Temperature",
    "Transparency",
    "DO"
]

OFFICIAL_FEATURES = [
    "Water_Temperature_C",
    "Transparency_m",
    "Dissolved_Oxygen_mg_L"
]


def detect_official_anomalies():

    print("\nAQUAGUARD AI - OFFICIAL DATA ANOMALY DETECTION")
    print("=" * 55)

    # 1. Load historical training data
    historical = pd.read_csv(HISTORICAL_FILE)

    # 2. Load newly retrieved official observations
    official = pd.read_csv(OFFICIAL_FILE)

    print("Historical records:", len(historical))
    print("Official observations:", len(official))

    # 3. Prepare historical features
    X_train = historical[HISTORICAL_FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove historical rows with no usable feature values
    X_train = X_train.dropna(how="all")

    # 4. Prepare official features
    X_new = official[OFFICIAL_FEATURES].copy()

    X_new.columns = HISTORICAL_FEATURES

    X_new = X_new.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # 5. Fit imputer using historical data only
    imputer = SimpleImputer(strategy="median")

    X_train_imputed = imputer.fit_transform(X_train)

    X_new_imputed = imputer.transform(X_new)

    # 6. Fit scaler using historical data only
    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train_imputed)

    X_new_scaled = scaler.transform(X_new_imputed)

    # 7. Train Isolation Forest on historical observations
    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42
    )

    model.fit(X_train_scaled)

    # 8. Detect anomalies in newly retrieved observations
    official["Anomaly_Label"] = model.predict(
        X_new_scaled
    )

    official["Anomaly_Score"] = model.decision_function(
        X_new_scaled
    )

    # 9. Convert predictions into readable statuses
    official["Status"] = official["Anomaly_Label"].map({
        1: "Normal",
        -1: "Potential Anomaly"
    })

    # 10. Save results
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    official.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # 11. Display results
    print("\nANOMALY DETECTION COMPLETED")
    print("-" * 45)

    print("\nDetection summary:")
    print(official["Status"].value_counts())

    print("\nOfficial observation results:")

    print(
        official[
            [
                "Observation_Date",
                "Monitoring_Location",
                "Depth",
                "Anomaly_Label",
                "Anomaly_Score",
                "Status"
            ]
        ].to_string(index=False)
    )

    print("\nResults saved to:", OUTPUT_FILE)

    return official


if __name__ == "__main__":
    detect_official_anomalies()