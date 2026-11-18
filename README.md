# AquaGuard AI
### AI-Powered Lake Biwa Water Quality Monitoring and Anomaly Detection System

## 1. Project Overview

AquaGuard AI is an artificial intelligence-based water-quality monitoring and analysis system developed using historical Lake Biwa water-quality data from Shiga Prefecture, Japan.

The project uses machine learning, data analysis, and interactive visualizations to identify unusual water-quality patterns and explore possible environmental changes.

It also provides online retrieval of the latest published water-quality observations from the official Shiga Prefecture website.

## 2. Project Objectives

- Analyse historical Lake Biwa water-quality measurements.
- Identify unusual water-quality patterns using machine learning.
- Explore relationships between different water-quality parameters.
- Analyse historical water-level and earthquake records.
- Experiment with COD prediction using machine learning.
- Retrieve the latest published official water-quality observations.

## 3. Datasets

The project uses the following datasets:

1. **Lake Biwa Water Quality:** Historical water-quality observations from Shiga Prefecture.
2. **Lake Biwa Water Level:** Water-level observations.
3. **Japan Earthquake Records:** Earthquake information used for geographical and historical analysis.
4. **Latest Official Water Quality:** Periodically published observations retrieved from the Shiga Prefecture website.

Official online water-quality source:

https://www.pref.shiga.lg.jp/dk00/2922.html

## 4. Data Preprocessing

Data preprocessing includes:

- Handling missing values.
- Cleaning and transforming datasets.
- Combining monitoring-location data.
- Selecting relevant machine-learning features.
- Preparing datasets for model training and analysis.

## 5. Machine Learning Models

### Isolation Forest – Anomaly Detection

Isolation Forest is used to identify unusual patterns in historical water-quality measurements.

The model classifies observations as normal or unusual based on patterns in the selected parameters.

These unusual observations are potential anomalies and do not automatically confirm pollution.

### Random Forest – COD Prediction

A Random Forest model is used for experimental Chemical Oxygen Demand (COD) prediction.

Model performance is evaluated using:

- MAE – Mean Absolute Error
- RMSE – Root Mean Squared Error
- R² – Coefficient of Determination

Walk-forward validation is used to evaluate the prediction model.

## 6. Dashboard Features

The Streamlit dashboard contains eight sections:

1. Water Quality Analysis
2. AI Anomaly Detection
3. Earthquake Analysis
4. Correlation Analysis
5. Water Level Analysis
6. COD Prediction
7. Latest Official Monitoring
8. Online Data Monitoring

## 7. Online Data Monitoring

AquaGuard AI retrieves the latest published water-quality observations directly from the official Shiga Prefecture website.

The retrieved observations are saved as:

`data/lake_biwa_official_online.csv`

The dashboard provides a refresh option and a CSV download feature.

**Note:** This is periodic official-data retrieval, not continuous real-time sensor monitoring.

## 8. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Requests
- HTML data extraction

## 9. How to Run the Project

### Step 1: Open the project folder

Open the AquaGuard_AI folder in VS Code.

### Step 2: Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Run the Streamlit dashboard

```powershell
python -m streamlit run app.py
```

### Step 4: Open the dashboard

The dashboard will open in your browser, usually at:

`http://localhost:8501`

## 10. Project Limitations

- Anomaly detection identifies unusual patterns, not confirmed pollution events.
- COD predictions are experimental and require further validation.
- Online water-quality observations depend on the publication of official data.
- The system does not currently use continuous physical sensor measurements.
- Earthquake analysis does not provide earthquake predictions or warnings.

## 11. Future Enhancements

- Integrate continuous IoT-based water-quality monitoring.
- Improve anomaly detection using additional historical observations.
- Enhance prediction accuracy with more training data.
- Develop automated environmental alerts.
- Explore earthquake-aware environmental monitoring.

## 12. Conclusion

AquaGuard AI demonstrates how artificial intelligence and machine learning can support environmental monitoring and water-quality analysis.

By combining historical datasets, anomaly detection, experimental prediction, and official online data retrieval, the project provides an interactive platform for exploring water-quality conditions in Lake Biwa.