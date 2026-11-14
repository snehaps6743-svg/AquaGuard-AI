import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

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
# LOAD DATASETS
# -----------------------------------

anomaly_path = Path("outputs/aquaguard_anomaly_results.csv")
earthquake_path = Path("outputs/earthquakes_within_50km.csv")
correlation_path = Path("outputs/correlation_dataset.csv")
water_level_path = Path("outputs/lake_biwa_water_level_cleaned.csv")

if not anomaly_path.exists():
    st.error("Anomaly results file not found. Run model.py first.")
    st.stop()

water_df = pd.read_csv(anomaly_path)

# -----------------------------------
# OVERVIEW METRICS
# -----------------------------------

total_records = len(water_df)
unusual_records = (water_df["Status"] == "Unusual").sum()
normal_records = (water_df["Status"] == "Normal").sum()

col1, col2, col3 = st.columns(3)

col1.metric("Water Quality Records", total_records)
col2.metric("Normal Records", normal_records)
col3.metric("Unusual Records", unusual_records)

st.divider()

# -----------------------------------
# TABS
# -----------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Water Quality",
    "AI Anomaly Detection",
    "Earthquake Analysis",
    "Correlation Analysis",
    "Water Level"
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
    st.dataframe(water_df, use_container_width=True)

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

    st.dataframe(unusual_df, use_container_width=True)

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

        st.dataframe(earthquake_df, use_container_width=True)

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
# ABOUT
# -----------------------------------

st.divider()

st.header("About AquaGuard AI")

st.write("""
AquaGuard AI is an exploratory environmental monitoring
project that analyses historical Lake Biwa water-quality
measurements, identifies unusual patterns using Isolation
Forest, and examines nearby earthquake records.

The system provides historical data analysis and
visualization. It does not currently provide validated
real-time pollution detection or earthquake-based warnings.
""")

st.success("AquaGuard AI Dashboard Loaded Successfully!")