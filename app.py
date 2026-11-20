import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import re

from io import StringIO
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="AquaGuard AI",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 AquaGuard AI")

st.subheader(
    "AI-Based Water Quality Monitoring and Anomaly Detection"
)

st.write("Lake Biwa, Japan")


# -----------------------------------
# LOAD DATASET PATHS
# -----------------------------------

anomaly_path = Path("outputs/aquaguard_anomaly_results.csv")
earthquake_path = Path("outputs/earthquakes_within_50km.csv")
correlation_path = Path("outputs/correlation_dataset.csv")
water_level_path = Path("outputs/lake_biwa_water_level_cleaned.csv")
prediction_path = Path("outputs/walk_forward_results.csv")

latest_water_path = Path(
    "data/lake_biwa_latest_cleaned.csv"
)


# -----------------------------------
# LOAD ANOMALY DATASET
# -----------------------------------

if not anomaly_path.exists():
    st.error("Anomaly results file not found. Run model.py first.")
    st.stop()

water_df = pd.read_csv(anomaly_path)


# -----------------------------------
# OVERVIEW METRICS
# -----------------------------------

total_records = len(water_df)

unusual_records = (
    water_df["Status"] == "Unusual"
).sum()

normal_records = (
    water_df["Status"] == "Normal"
).sum()

col1, col2, col3 = st.columns(3)

col1.metric("Water Quality Records", total_records)
col2.metric("Normal Records", normal_records)
col3.metric("Unusual Records", unusual_records)

st.divider()


# -----------------------------------
# TABS
# -----------------------------------

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Water Quality",
    "AI Anomaly Detection",
    "Earthquake Analysis",
    "Correlation Analysis",
    "Water Level",
    "COD Prediction",
    "Latest Official Monitoring",
    "Online Data Monitoring"
])


# -----------------------------------
# TAB 1: WATER QUALITY
# -----------------------------------

with tab1:

    st.header("Lake Biwa Water Quality Trends")

    feature = st.selectbox(
        "Select Water Quality Parameter",
        [
            "COD_North_Basin_mg_L",
            "COD_South_Basin_mg_L",
            "Total_Nitrogen_North_Basin_mg_L",
            "Total_Nitrogen_South_Basin_mg_L",
            "Total_Phosphorus_North_Basin_mg_L",
            "Total_Phosphorus_South_Basin_mg_L"
        ]
    )

    fig = px.line(
        water_df,
        x="Year",
        y=feature,
        markers=True,
        title=f"{feature} Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("Water Quality Dataset")

    st.dataframe(
        water_df,
        use_container_width=True
    )


# -----------------------------------
# TAB 2: ANOMALY DETECTION
# -----------------------------------

with tab2:

    st.header("AI-Based Anomaly Detection")

    st.write(
        "Isolation Forest identifies unusual patterns "
        "in the historical water-quality measurements."
    )

    fig = px.scatter(
        water_df,
        x="Year",
        y="COD_North_Basin_mg_L",
        color="Status",
        title="Normal vs Unusual Water Quality Records",
        hover_data=["COD_South_Basin_mg_L"]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Unusual Records")

    unusual_df = water_df[
        water_df["Status"] == "Unusual"
    ]

    st.dataframe(
        unusual_df,
        use_container_width=True
    )

    st.download_button(
        "Download Anomaly Results",
        data=water_df.to_csv(index=False),
        file_name="aquaguard_anomaly_results.csv",
        mime="text/csv"
    )

    st.warning(
        "An unusual record is an AI-flagged observation, "
        "not confirmed pollution."
    )


# -----------------------------------
# TAB 3: EARTHQUAKE ANALYSIS
# -----------------------------------

with tab3:

    st.header("Earthquake Analysis Near Lake Biwa")

    if earthquake_path.exists():

        earthquake_df = pd.read_csv(earthquake_path)

        col1, col2 = st.columns(2)

        col1.metric(
            "Earthquake Records Within 50 km",
            len(earthquake_df)
        )

        col2.metric(
            "Average Magnitude",
            round(earthquake_df["mag"].mean(), 2)
        )

        earthquake_df["time"] = pd.to_datetime(
            earthquake_df["time"],
            errors="coerce",
            utc=True
        )

        earthquake_df["Year"] = (
            earthquake_df["time"].dt.year
        )

        yearly = (
            earthquake_df.groupby("Year")
            .size()
            .reset_index(name="Earthquake_Count")
        )

        fig = px.bar(
            yearly,
            x="Year",
            y="Earthquake_Count",
            title="Recorded Nearby Earthquakes Per Year"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            earthquake_df,
            use_container_width=True
        )

    else:

        st.info("Nearby earthquake dataset not found.")


# -----------------------------------
# TAB 4: CORRELATION ANALYSIS
# -----------------------------------

with tab4:

    st.header("Earthquake and Water Quality Correlation")

    if correlation_path.exists():

        corr_df = pd.read_csv(correlation_path)

        north_corr = corr_df[
            ["Earthquake_Count", "COD_North_Basin_mg_L"]
        ].corr().iloc[0, 1]

        south_corr = corr_df[
            ["Earthquake_Count", "COD_South_Basin_mg_L"]
        ].corr().iloc[0, 1]

        col1, col2 = st.columns(2)

        col1.metric(
            "North Basin Correlation",
            round(north_corr, 3)
        )

        col2.metric(
            "South Basin Correlation",
            round(south_corr, 3)
        )

        fig = px.scatter(
            corr_df,
            x="Earthquake_Count",
            y="COD_North_Basin_mg_L",
            title="Earthquake Activity vs North Basin COD"
        )

        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            corr_df,
            x="Earthquake_Count",
            y="COD_South_Basin_mg_L",
            title="Earthquake Activity vs South Basin COD"
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            "Correlation indicates association, not causation. "
            "The analysis uses only overlapping years with "
            "recorded nearby earthquakes."
        )

    else:

        st.info("Correlation dataset not found.")


# -----------------------------------
# TAB 5: WATER LEVEL
# -----------------------------------

with tab5:

    st.header("Lake Biwa Water Level Monitoring")

    if water_level_path.exists():

        level_df = pd.read_csv(water_level_path)

        level_df["Date_Time"] = pd.to_datetime(
            level_df["Date_Time"],
            errors="coerce"
        )

        fig = px.line(
            level_df,
            x="Date_Time",
            y="Water_Level",
            markers=True,
            title="Lake Biwa Water Level - January 1993"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "This dataset contains 62 valid measurements "
            "from January 1993. It is shown separately and "
            "is not treated as a continuous long-term record."
        )

    else:

        st.info("Water-level dataset not found.")


# -----------------------------------
# TAB 6: COD PREDICTION
# -----------------------------------

with tab6:

    st.header("AI-Based COD Prediction")

    if prediction_path.exists():

        prediction_df = pd.read_csv(prediction_path)

        actual = prediction_df["Actual_COD"]
        predicted = prediction_df["Random_Forest"]

        mae = mean_absolute_error(actual, predicted)

        rmse = mean_squared_error(
            actual,
            predicted
        ) ** 0.5

        r2 = r2_score(actual, predicted)

        st.subheader("Random Forest Model Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric("MAE", round(mae, 4))
        col2.metric("RMSE", round(rmse, 4))
        col3.metric("R² Score", round(r2, 4))

        st.success(
            "Random Forest achieved the best performance "
            "among the tested models in walk-forward validation."
        )

        st.subheader("Actual vs Predicted COD")

        fig = px.line(
            prediction_df,
            x="Year",
            y=["Actual_COD", "Random_Forest"],
            markers=True,
            title="Actual vs Random Forest Predicted COD"
        )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="COD (mg/L)",
            legend_title="Measurements"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Comparison")

        comparison = []

        for model_name in [
            "Linear_Regression",
            "Random_Forest",
            "Dummy_Regressor"
        ]:

            model_predictions = prediction_df[model_name]

            comparison.append({
                "Model": model_name,
                "MAE": mean_absolute_error(
                    actual,
                    model_predictions
                ),
                "RMSE": mean_squared_error(
                    actual,
                    model_predictions
                ) ** 0.5,
                "R2 Score": r2_score(
                    actual,
                    model_predictions
                )
            })

        comparison_df = pd.DataFrame(comparison)

        st.dataframe(
            comparison_df.round(4),
            use_container_width=True
        )

        st.subheader("Walk-Forward Validation Results")

        st.dataframe(
            prediction_df,
            use_container_width=True
        )

        st.download_button(
            "Download Prediction Results",
            data=prediction_df.to_csv(index=False),
            file_name="walk_forward_results.csv",
            mime="text/csv"
        )

        st.info(
            "The model was evaluated using historical data "
            "from 1997 to 2018. These are experimental "
            "predictions, not validated future pollution forecasts."
        )

    else:

        st.warning(
            "Prediction results not found. "
            "Run cod_prediction.py first."
        )


# -----------------------------------
# TAB 7: LATEST OFFICIAL MONITORING
# -----------------------------------

with tab7:

    st.header("Latest Official Lake Biwa Water Quality Monitoring")

    st.write(
        "Explore official water-quality observations from "
        "54 Lake Biwa monitoring locations."
    )

    if latest_water_path.exists():

        latest_df = pd.read_csv(latest_water_path)

        latest_df["Date"] = pd.to_datetime(
            latest_df["Date"],
            errors="coerce"
        )

        latest_df = latest_df.dropna(subset=["Date"])

        if not latest_df.empty:

            # Overview metrics

            latest_date = latest_df["Date"].max()

            total_observations = len(latest_df)

            total_locations = latest_df["Location"].nunique()

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Observations",
                total_observations
            )

            col2.metric(
                "Monitoring Locations",
                total_locations
            )

            col3.metric(
                "Latest Observation",
                latest_date.strftime("%d %b %Y")
            )

            st.divider()

            # Date information

            st.subheader("Dataset Information")

            st.write(
                f"Observation period: "
                f"{latest_df['Date'].min().strftime('%d %b %Y')} "
                f"to {latest_date.strftime('%d %b %Y')}"
            )

            # Location filter

            locations = sorted(
                latest_df["Location"].dropna().unique().tolist()
            )

            selected_location = st.selectbox(
                "Select Monitoring Location",
                ["All Locations"] + locations
            )

            if selected_location == "All Locations":

                filtered_df = latest_df.copy()

            else:

                filtered_df = latest_df[
                    latest_df["Location"] == selected_location
                ].copy()

            # Parameter selection

            st.subheader("Water Quality Analysis")

            parameter_options = {
                "Water Temperature": "Water_Temperature",
                "Air Temperature": "Air_Temperature",
                "Transparency": "Transparency",
                "pH": "pH",
                "Dissolved Oxygen (DO)": "DO",
                "Biochemical Oxygen Demand (BOD)": "BOD",
                "Chemical Oxygen Demand (COD)": "COD",
                "Suspended Solids (SS)": "SS",
                "Total Nitrogen": "Total_Nitrogen",
                "Total Phosphorus": "Total_Phosphorus",
                "Coliform Count": "Coliform_Count"
            }

            selected_parameter = st.selectbox(
                "Choose a Water Quality Parameter",
                list(parameter_options.keys())
            )

            parameter_column = parameter_options[
                selected_parameter
            ]

            # Numeric chart

            chart_df = filtered_df.copy()

            chart_df[parameter_column] = pd.to_numeric(
                chart_df[parameter_column],
                errors="coerce"
            )

            chart_df = chart_df.dropna(
                subset=[parameter_column]
            )

            if not chart_df.empty:

                if selected_location == "All Locations":

                    plot_df = (
                        chart_df.groupby("Date")[parameter_column]
                        .mean()
                        .reset_index()
                    )

                    fig = px.line(
                        plot_df,
                        x="Date",
                        y=parameter_column,
                        markers=True,
                        title=f"Average {selected_parameter} Across Monitoring Locations"
                    )

                else:

                    fig = px.line(
                        chart_df,
                        x="Date",
                        y=parameter_column,
                        markers=True,
                        title=f"{selected_parameter} - {selected_location}"
                    )

                fig.update_layout(
                    xaxis_title="Observation Date",
                    yaxis_title=selected_parameter
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.warning(
                    "No numeric measurements are available "
                    "for this parameter and selection."
                )

            st.caption(
                "Charts use numeric readings only. Censored readings "
                "such as <0.5 and >2.5 are excluded from numerical "
                "calculations but preserved in the original-reading "
                "columns."
            )

            # Latest measurement table

            st.subheader("Official Monitoring Records")

            st.dataframe(
                filtered_df.sort_values("Date", ascending=False),
                use_container_width=True
            )

            # Download button

            st.download_button(
                "Download Latest Water Quality Dataset",
                data=latest_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="lake_biwa_latest_water_quality.csv",
                mime="text/csv"
            )

        else:

            st.warning("No valid observation dates found.")

    else:

        st.error(
            "Latest water-quality dataset not found. "
            "Check data/lake_biwa_latest_cleaned.csv"
        )


# -----------------------------------
# ABOUT AQUAGUARD AI
# -----------------------------------

st.divider()

st.header("About AquaGuard AI")

st.write("""
AquaGuard AI is an exploratory environmental monitoring
project that analyses historical and latest published
Lake Biwa water-quality observations.

The system identifies unusual historical patterns using
Isolation Forest, examines nearby earthquake records,
analyses water-level observations and evaluates experimental
COD predictions using machine learning.

The latest official monitoring dataset provides
location-wise water-quality analysis and visualization.

The system does not currently provide validated
real-time pollution detection or earthquake-based warnings.
""")

st.success(
    "AquaGuard AI Dashboard Loaded Successfully!"
)


# -----------------------------------
# TAB 8: ONLINE DATA MONITORING
# -----------------------------------

with tab8:

    st.header("Online Lake Biwa Water Quality Monitoring")

    st.write(
        "Retrieve the latest published water-quality observations "
        "directly from the official Shiga Prefecture website."
    )

    # REFRESH OFFICIAL DATA AND RUN AI DETECTION

    if st.button("Refresh Official Online Data"):

        with st.spinner(
            "Fetching official data and running AI anomaly detection..."
        ):

            try:

                # Step 1: Fetch official observations

                from fetch_official_data import fetch_official_data

                fetch_official_data()

                # Step 2: Run AI anomaly detection

                from online_anomaly_detection import detect_official_anomalies

                detect_official_anomalies()

                st.success(
                    "Official data retrieved and AI anomaly detection completed!"
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Data retrieval or anomaly detection failed: {e}"
                )

    # DISPLAY OFFICIAL ONLINE DATA

    online_file = Path(
        "data/lake_biwa_official_online.csv"
    )

    if online_file.exists():

        online_df = pd.read_csv(online_file)

        if not online_df.empty:

            st.subheader("Latest Retrieved Observations")

            col1, col2 = st.columns(2)

            col1.metric(
                "Survey Date",
                online_df["Observation_Date"].iloc[0]
            )

            col2.metric(
                "Total Observations",
                len(online_df)
            )

            st.dataframe(
                online_df,
                use_container_width=True
            )

            st.download_button(
                "Download Online Water Quality Data",
                data=online_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="lake_biwa_official_online.csv",
                mime="text/csv"
            )

            # DISPLAY AI ANOMALY DETECTION RESULTS

            anomaly_file = Path(
                "outputs/official_online_anomaly_results.csv"
            )

            if anomaly_file.exists():

                anomaly_df = pd.read_csv(anomaly_file)

                st.divider()

                st.subheader("AI Anomaly Detection Results")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Observations",
                    len(anomaly_df)
                )

                col2.metric(
                    "Normal",
                    (anomaly_df["Status"] == "Normal").sum()
                )

                col3.metric(
                    "Potential Anomalies",
                    (anomaly_df["Status"] == "Potential Anomaly").sum()
                )

                st.dataframe(
                    anomaly_df[
                        [
                            "Observation_Date",
                            "Monitoring_Location",
                            "Depth",
                            "Anomaly_Label",
                            "Anomaly_Score",
                            "Status"
                        ]
                    ],
                    use_container_width=True
                )

                st.download_button(
                    "Download AI Anomaly Results",
                    data=anomaly_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="official_online_anomaly_results.csv",
                    mime="text/csv"
                )

                st.warning(
                    "Potential anomalies are statistically unusual "
                    "observations. They do not confirm pollution."
                )

            else:

                st.info(
                    "AI anomaly results are not available yet. "
                    "Click Refresh Official Online Data to run detection."
                )

        else:

            st.warning(
                "The official online dataset is empty. "
                "Please refresh the official data."
            )

    else:

        st.info(
            "No online observations downloaded yet. "
            "Click Refresh Official Online Data."
        )