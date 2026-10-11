
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="MedEquip Intelligence",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5edf8;
}
[data-testid="stMetric"] {
    background: #17243a;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #293b56;
}
</style>
""", unsafe_allow_html=True)

st.title("🏥 MedEquip Intelligence")
st.caption("Medical Equipment Failure Prediction & Analytics")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Dashboard", "Dataset Explorer", "Failure Prediction"]
)

if page == "Dashboard":
    st.subheader("Equipment Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Equipment Monitored", "1,250")
    col2.metric("Failure Alerts", "28")
    col3.metric("System Status", "Demo")

    st.info(
        "Upload your real dataset to begin analyzing "
        "medical equipment records."
    )

    chart_data = pd.DataFrame({
        "Status": ["Normal", "Warning", "High Risk"],
        "Equipment": [80, 15, 5]
    })

    fig = px.pie(
        chart_data,
        names="Status",
        values="Equipment",
        hole=0.55,
        title="Sample Equipment Status",
        color_discrete_sequence=[
            "#14b8a6", "#f59e0b", "#ef4444"
        ]
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Dataset Explorer":
    st.subheader("Explore Equipment Data")

    uploaded_file = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.success("Dataset loaded successfully!")
        st.write("Dataset shape:", df.shape)
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Missing Values")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Values"]
        st.dataframe(missing, use_container_width=True)

        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False),
            file_name="equipment_data.csv",
            mime="text/csv"
        )
    else:
        st.info("Upload your Kaggle CSV file to explore the data.")

elif page == "Failure Prediction":
    st.subheader("Equipment Risk Assessment")

    st.warning(
        "Demo interface only. These inputs do not yet connect "
        "to a trained model."
    )

    equipment = st.selectbox(
        "Equipment type",
        ["Ventilator", "Infusion Pump", "Patient Monitor", "Other"]
    )

    usage_hours = st.number_input(
        "Usage hours", min_value=0, max_value=100000, value=1000
    )

    temperature = st.number_input(
        "Operating temperature (°C)",
        min_value=0.0, max_value=100.0, value=25.0
    )

    if st.button("Check Equipment"):
        st.write("Selected equipment:", equipment)
        st.write("Usage hours:", usage_hours)
        st.write("Temperature:", temperature)

        st.info(
            "The machine-learning model will be connected "
            "after we inspect your dataset."
        )

st.divider()
st.caption(
    "Student project · Educational demonstration only · "
    "Not for real-world medical safety decisions"
)

import streamlit as st

st.set_page_config(
    page_title="MedEquip Intelligence",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 MedEquip Intelligence")
st.success("Your Streamlit application is running!")

st.write("Your medical equipment datasets are ready to connect.")
