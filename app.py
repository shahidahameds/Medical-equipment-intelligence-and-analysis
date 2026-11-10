
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="MedEquip Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

DEVICE_PATH = BASE_DIR / "devices-1681209661.csv"
EVENT_PATH = BASE_DIR / "events-1681209680.csv"
MANUFACTURER_PATH = BASE_DIR / "manufacturers-1681209657.csv"


# =====================================================
# PROFESSIONAL LIGHT UI
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background-color: #F3F7FC;
    color: #243B53;
}

/* Main content */
.main .block-container {
    background-color: #F3F7FC;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #E7EFF9;
    border-right: 1px solid #D1DFEF;
}

/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #183153 !important;
}

/* Main headings */
.stApp h1,
.stApp h2,
.stApp h3 {
    color: #15365B !important;
    font-weight: 700;
}

/* General text */
.stApp p,
.stApp label,
.stApp li {
    color: #344B64;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #D6E2F0;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 3px 10px rgba(30, 64, 100, 0.06);
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #536B84 !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #123A63 !important;
    font-weight: 700;
}

/* Hero section */
.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(120deg, #DCEEFF, #E9EAFE);
    border: 1px solid #C7D9F0;
    margin-bottom: 25px;
}

.hero h1 {
    color: #123A63 !important;
    font-size: 32px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #405F80 !important;
    margin-top: 10px;
}

/* Tables */
[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border: 1px solid #D6E2F0;
    border-radius: 12px;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    background-color: #1769AA;
    color: #FFFFFF !important;
    border: none;
    border-radius: 9px;
    font-weight: 600;
    padding: 8px 15px;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background-color: #10558D;
    color: #FFFFFF !important;
}

/* Input fields */
.stTextInput input,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    background-color: #FFFFFF;
    color: #183153;
    border-color: #C9D8E8;
}

/* Captions */
[data-testid="stCaptionContainer"] p,
.small-note {
    color: #63778D !important;
    font-size: 13px;
}

/* Dividers */
hr {
    border-color: #D6E2F0;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# PLOTLY LIGHT THEME
# =====================================================

px.defaults.template = "plotly_white"

CHART_LAYOUT = {
    "font": {
        "family": "Inter",
        "color": "#243B53",
        "size": 12,
    },
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "margin": dict(l=10, r=10, t=25, b=10),
}


def style_chart(fig, height=None):
    fig.update_layout(**CHART_LAYOUT)

    fig.update_xaxes(
        title_font_color="#243B53",
        tickfont_color="#52677F",
        gridcolor="#E1EAF3",
        zerolinecolor="#D6E2F0",
    )

    fig.update_yaxes(
        title_font_color="#243B53",
        tickfont_color="#52677F",
        gridcolor="#E1EAF3",
        zerolinecolor="#D6E2F0",
    )

    if height:
        fig.update_layout(height=height)

    return fig


# =====================================================
# HERO SECTION
# =====================================================

st.markdown("""
<div class="hero">
    <h1>🏥 MedEquip Intelligence</h1>
    <p>
        Medical equipment analytics · Device registry ·
        Historical adverse-event exploration
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data(show_spinner="Loading datasets...")
def load_csv(path):
    return pd.read_csv(path, low_memory=False)


missing = [
    p.name
    for p in [DEVICE_PATH, EVENT_PATH, MANUFACTURER_PATH]
    if not p.exists()
]

if missing:
    st.error("The following dataset files were not found:")

    for filename in missing:
        st.write(f"- {filename}")

    st.info("Place the CSV files beside app.py, then refresh the app.")
    st.stop()


try:
    devices = load_csv(str(DEVICE_PATH))
    events = load_csv(str(EVENT_PATH))
    manufacturers = load_csv(str(MANUFACTURER_PATH))

except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.stop()


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def find_column(df, candidates):
    """Find a column using case-insensitive candidate names."""

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def missing_report(df):
    return pd.DataFrame({
        "Column": df.columns.astype(str),
        "Missing values": df.isna().sum().values,
        "Missing (%)": (df.isna().mean().values * 100).round(2),
        "Unique values": df.nunique(dropna=True).values,
    }).sort_values("Missing (%)", ascending=False)


# =====================================================
# COLUMN DETECTION
# =====================================================

device_id_col = find_column(
    devices, ["id", "device_id"]
)

event_device_col = find_column(
    events, ["device_id", "device id", "deviceid"]
)

reason_col = find_column(
    events, ["reason", "event_reason", "event_type"]
)

risk_col = find_column(
    devices, ["risk_class", "risk class", "risk"]
)

classification_col = find_column(
    devices, ["classification", "device_class", "device type"]
)

manufacturer_id_col = find_column(
    devices, ["manufacturer_id", "manufacturer id", "manufacturerid"]
)

manufacturer_name_col = find_column(
    manufacturers,
    ["name", "manufacturer_name", "company_name", "manufacturer"],
)

manufacturer_table_id_col = find_column(
    manufacturers, ["id", "manufacturer_id", "manufacturer id"]
)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## 🩺 MedEquip")
    st.caption("Analytics workspace")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Equipment Explorer",
            "Event Analytics",
            "Manufacturer Analytics",
            "Data Quality",
        ],
    )

    st.divider()

    st.markdown("### Dataset status")

    st.write(f"Devices: **{len(devices):,}**")
    st.write(f"Events: **{len(events):,}**")
    st.write(f"Manufacturers: **{len(manufacturers):,}**")

    if st.button("🔄 Refresh datasets", width="stretch"):
        st.cache_data.clear()
        st.rerun()


# =====================================================
# OVERVIEW
# =====================================================

if page == "Overview":

    st.subheader("Dashboard Overview")

    st.caption("Summary generated from your actual CSV files.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Device Records", f"{len(devices):,}")
    c2.metric("Event Records", f"{len(events):,}")
    c3.metric("Manufacturers", f"{len(manufacturers):,}")
    c4.metric("Device Attributes", f"{len(devices.columns):,}")

    left, right = st.columns(2)

    # Equipment classification

    with left:

        st.markdown("### Equipment classification")

        if classification_col:

            counts = (
                devices[classification_col]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .head(12)
                .rename_axis("Classification")
                .reset_index(name="Records")
            )

            fig = px.bar(
                counts,
                x="Records",
                y="Classification",
                orientation="h",
                color="Records",
                color_continuous_scale="Blues",
            )

            fig.update_layout(
                coloraxis_showscale=False,
                yaxis={"categoryorder": "total ascending"},
            )

            style_chart(fig, 400)

            st.plotly_chart(fig, width="stretch")

        else:
            st.info("No classification column was detected.")

    # Device risk classes

    with right:

        st.markdown("### Device risk classes")

        if risk_col:

            risk_counts = (
                devices[risk_col]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .rename_axis("Risk class")
                .reset_index(name="Records")
            )

            fig = px.pie(
                risk_counts,
                names="Risk class",
                values="Records",
                hole=0.60,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )

            fig.update_layout(
                height=400,
                legend=dict(orientation="h", y=-0.12),
            )

            style_chart(fig)

            st.plotly_chart(fig, width="stretch")

        else:
            st.info("No risk-class column was detected.")

    st.markdown("### Recent device records")

    st.dataframe(
        devices.head(10),
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "⬇ Download device dataset",
        data=devices.to_csv(index=False).encode("utf-8"),
        file_name="devices_export.csv",
        mime="text/csv",
        width="stretch",
    )


# =====================================================
# EQUIPMENT EXPLORER
# =====================================================

elif page == "Equipment Explorer":

    st.subheader("Equipment Explorer")

    st.caption(
        "Search, filter, inspect, and export equipment records."
    )

    filtered = devices.copy()

    col1, col2 = st.columns([2, 1])

    with col1:

        query = st.text_input(
            "Search all fields",
            placeholder="Enter device ID, manufacturer, or keyword...",
        )

    with col2:

        max_rows = st.selectbox(
            "Maximum displayed rows",
            [100, 500, 1000, 5000],
            index=0,
        )

    if risk_col:

        risk_options = sorted(
            devices[risk_col]
            .fillna("Unknown")
            .astype(str)
            .unique()
        )

        selected_risks = st.multiselect(
            "Filter by risk class",
            risk_options,
            default=[],
        )

        if selected_risks:

            filtered = filtered[
                filtered[risk_col]
                .fillna("Unknown")
                .astype(str)
                .isin(selected_risks)
            ]

    if query.strip():

        mask = filtered.astype(str).apply(
            lambda col: col.str.contains(
                query.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ).any(axis=1)

        filtered = filtered[mask]

    st.metric(
        "Matching equipment records",
        f"{len(filtered):,}",
    )

    st.dataframe(
        filtered.head(max_rows),
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "⬇ Download filtered equipment",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_equipment.csv",
        mime="text/csv",
        width="stretch",
    )


# =====================================================
# EVENT ANALYTICS
# =====================================================

elif page == "Event Analytics":

    st.subheader("Historical Event Analytics")

    st.caption(
        "These records describe reported events. A report does not by itself "
        "establish that the device caused the event."
    )

    a, b = st.columns(2)

    a.metric("Total event records", f"{len(events):,}")
    b.metric("Event attributes", f"{len(events.columns):,}")

    if reason_col:

        reason_counts = (
            events[reason_col]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .head(20)
            .rename_axis("Reported reason")
            .reset_index(name="Records")
        )

        st.markdown("### Most common reported reasons")

        fig = px.bar(
            reason_counts,
            x="Records",
            y="Reported reason",
            orientation="h",
            color="Records",
            color_continuous_scale="Blues",
        )

        fig.update_layout(
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
        )

        style_chart(fig, 560)

        st.plotly_chart(fig, width="stretch")

        st.markdown("### Filter events")

        reason_options = sorted(
            events[reason_col]
            .fillna("Unknown")
            .astype(str)
            .unique()
        )

        chosen_reasons = st.multiselect(
            "Reported reason",
            reason_options,
        )

        filtered_events = events

        if chosen_reasons:

            filtered_events = events[
                events[reason_col]
                .fillna("Unknown")
                .astype(str)
                .isin(chosen_reasons)
            ]

        st.write(
            f"Matching event records: {len(filtered_events):,}"
        )

        st.dataframe(
            filtered_events.head(500),
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "⬇ Download filtered events",
            data=filtered_events.to_csv(index=False).encode("utf-8"),
            file_name="filtered_events.csv",
            mime="text/csv",
            width="stretch",
        )

    else:

        st.warning(
            "No recognized event-reason column was found. "
            "Review the available columns below."
        )

    with st.expander("Inspect event columns"):

        st.write(events.columns.tolist())

        st.dataframe(
            events.head(20),
            width="stretch",
        )


# =====================================================
# MANUFACTURER ANALYTICS
# =====================================================

elif page == "Manufacturer Analytics":

    st.subheader("Manufacturer Analytics")

    left, right = st.columns(2)

    with left:

        st.markdown("### Manufacturer registry")

        if manufacturer_name_col:

            manufacturer_counts = (
                manufacturers[manufacturer_name_col]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .head(15)
                .rename_axis("Manufacturer")
                .reset_index(name="Records")
            )

            fig = px.bar(
                manufacturer_counts,
                x="Records",
                y="Manufacturer",
                orientation="h",
                color="Records",
                color_continuous_scale="Blues",
            )

            fig.update_layout(
                coloraxis_showscale=False,
                yaxis={"categoryorder": "total ascending"},
            )

            style_chart(fig, 470)

            st.plotly_chart(fig, width="stretch")

        else:

            st.info(
                "No manufacturer-name column detected. "
                "You can still explore the full registry below."
            )

    with right:

        st.markdown("### Registry preview")

        st.dataframe(
            manufacturers.head(20),
            width="stretch",
            hide_index=True,
        )

    if manufacturer_id_col and manufacturer_table_id_col:

        st.markdown("### Device records by manufacturer")

        device_counts = (
            devices[manufacturer_id_col]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .head(15)
            .rename_axis("Manufacturer ID")
            .reset_index(name="Device records")
        )

        st.dataframe(
            device_counts,
            width="stretch",
            hide_index=True,
        )

    st.download_button(
        "⬇ Download manufacturer registry",
        data=manufacturers.to_csv(index=False).encode("utf-8"),
        file_name="manufacturers_export.csv",
        mime="text/csv",
        width="stretch",
    )


# =====================================================
# DATA QUALITY
# =====================================================

elif page == "Data Quality":

    st.subheader("Data Quality Center")

    st.caption(
        "Inspect missing values, duplicates, and dataset structure."
    )

    dataset_choice = st.selectbox(
        "Select a dataset",
        ["Devices", "Events", "Manufacturers"],
    )

    dataset_map = {
        "Devices": devices,
        "Events": events,
        "Manufacturers": manufacturers,
    }

    selected_df = dataset_map[dataset_choice]

    report = missing_report(selected_df)

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", f"{len(selected_df):,}")
    c2.metric("Columns", f"{len(selected_df.columns):,}")
    c3.metric(
        "Duplicate rows",
        f"{selected_df.duplicated().sum():,}",
    )

    st.markdown("### Missing values by column")

    missing_chart = report[
        report["Missing values"] > 0
    ].head(20)

    if not missing_chart.empty:

        fig = px.bar(
            missing_chart,
            x="Missing (%)",
            y="Column",
            orientation="h",
            color="Missing (%)",
            color_continuous_scale="Blues",
        )

        fig.update_layout(
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
        )

        style_chart(fig)

        st.plotly_chart(fig, width="stretch")

    else:

        st.success(
            "No missing values were found in this dataset."
        )

    st.markdown("### Column report")

    st.dataframe(
        report,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "⬇ Download data quality report",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name=f"{dataset_choice.lower()}_quality_report.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("View dataset column names"):

        st.write(selected_df.columns.tolist())


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <p class="small-note">
        MedEquip Intelligence · Educational analytics prototype.
        Historical adverse-event records are not validated forecasts
        of future equipment failures.
    </p>
    """,
    unsafe_allow_html=True,
)
