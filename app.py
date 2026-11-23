
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EstateIQ | House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM PROFESSIONAL CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --green: #153F35;
    --light-green: #DCE9DF;
    --gold: #C9A66B;
    --cream: #F6F4EE;
    --dark: #182A25;
}

.stApp {
    background: #F6F4EE;
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* HERO SECTION */

.hero {
    position: relative;
    min-height: 360px;
    border-radius: 24px;
    overflow: hidden;
    display: flex;
    align-items: center;
    padding: 55px;
    margin-bottom: 35px;
    background:
        linear-gradient(90deg,
        rgba(13, 39, 32, 0.96) 0%,
        rgba(13, 39, 32, 0.80) 48%,
        rgba(13, 39, 32, 0.20) 100%),
        url("https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=1600");
    background-size: cover;
    background-position: center;
}

.hero-content {
    max-width: 650px;
}

.hero-tag {
    display: inline-block;
    color: #E8D2A8;
    border: 1px solid rgba(232,210,168,0.5);
    padding: 8px 16px;
    border-radius: 30px;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 600;
    margin-bottom: 20px;
}

.hero h1 {
    font-family: 'Playfair Display', serif;
    color: white !important;
    font-size: 48px;
    line-height: 1.2;
    margin-bottom: 15px;
}

.hero p {
    color: #E1E9E3 !important;
    font-size: 16px;
    line-height: 1.8;
    max-width: 520px;
}

/* SECTION HEADINGS */

.section-heading {
    font-family: 'Playfair Display', serif;
    font-size: 29px;
    color: #153F35;
    margin-bottom: 7px;
}

.section-description {
    color: #777F79;
    font-size: 14px;
    margin-bottom: 25px;
}

/* FORM CONTAINERS */

[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border: 1px solid #E8E9E2;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 8px 30px rgba(30, 55, 43, 0.045);
}

[data-testid="stWidgetLabel"] p {
    color: #344B40 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

input {
    border-radius: 9px !important;
}

[data-baseweb="select"] > div {
    border-radius: 9px !important;
    background: #FAFBF8 !important;
}

[data-testid="stNumberInput"] button {
    background: #EAF1EA !important;
    color: #153F35 !important;
}

.stSlider [data-baseweb="slider"] {
    color: #153F35 !important;
}

/* PREDICT BUTTON */

.stButton > button {
    width: 100%;
    height: 54px;
    border: none;
    border-radius: 12px;
    background: linear-gradient(100deg, #153F35, #26765A);
    color: white;
    font-size: 16px;
    font-weight: 700;
    transition: 0.25s ease;
    box-shadow: 0 7px 18px rgba(21,63,53,0.18);
}

.stButton > button:hover {
    background: linear-gradient(100deg, #26765A, #153F35);
    color: white;
    transform: translateY(-2px);
}

/* RESULT CARD */

.result-card {
    background: linear-gradient(120deg, #153F35, #245D48);
    border-radius: 20px;
    padding: 32px;
    color: white;
    margin-top: 25px;
    box-shadow: 0 12px 30px rgba(21,63,53,0.16);
}

.result-label {
    color: #D5E4D8;
    font-size: 13px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.result-price {
    color: #F1D6A5;
    font-family: 'Playfair Display', serif;
    font-size: 42px;
    font-weight: 700;
    margin: 10px 0;
}

.result-note {
    color: #DCE9DF;
    font-size: 13px;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #818B82;
    font-size: 12px;
    padding-top: 35px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    df = pd.read_csv("data/train.csv")
    return model, preprocessor, df


try:
    model, preprocessor, df = load_artifacts()
except Exception as e:
    st.error("Unable to load the model. Please check your models and data folders.")
    st.stop()

features = df.drop(columns=["SalePrice", "Id"])


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="hero-tag">ESTATEIQ · SMART PROPERTY VALUATION</div>
        <h1>Find the True Value<br>of Your Dream Home.</h1>
        <p>
            Discover an estimated property value using machine learning.
            Enter your property details and get an intelligent price estimate.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# PROPERTY FORM
# --------------------------------------------------

st.markdown("""
<div class="section-heading">Property Details</div>
<div class="section-description">
Tell us about the property to generate your estimated house price.
</div>
""", unsafe_allow_html=True)

with st.container(border=True):

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 🏠 Property Information")

        overall_qual = st.slider(
            "Overall House Quality",
            1, 10, 5
        )

        living_area = st.number_input(
            "Living Area (sq ft)",
            min_value=300,
            max_value=10000,
            value=int(features["GrLivArea"].median())
        )

        bedrooms = st.number_input(
            "Number of Bedrooms",
            min_value=0,
            max_value=15,
            value=int(features["BedroomAbvGr"].median())
        )

        bathrooms = st.number_input(
            "Full Bathrooms",
            min_value=0,
            max_value=10,
            value=int(features["FullBath"].median())
        )

        lot_area = st.number_input(
            "Lot Area (sq ft)",
            min_value=500,
            max_value=100000,
            value=int(features["LotArea"].median())
        )

    with col2:
        st.markdown("#### 📍 Location & Facilities")

        year_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=2026,
            value=int(features["YearBuilt"].median())
        )

        garage_cars = st.selectbox(
            "Garage Capacity",
            [0, 1, 2, 3, 4],
            index=2
        )

        basement_area = st.number_input(
            "Basement Area (sq ft)",
            min_value=0,
            max_value=5000,
            value=int(features["TotalBsmtSF"].median())
        )

        neighborhood = st.selectbox(
            "Neighborhood",
            sorted(features["Neighborhood"].dropna().unique())
        )

        kitchen_quality = st.selectbox(
            "Kitchen Quality",
            ["Ex", "Gd", "TA", "Fa", "Po"]
        )

    st.markdown("---")

    predict = st.button("✦  Predict Property Value", use_container_width=True)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict:

    input_data = {}

    for column in features.columns:

        if pd.api.types.is_numeric_dtype(features[column]):
            input_data[column] = features[column].median()

        else:
            modes = features[column].mode(dropna=True)
            input_data[column] = (
                modes.iloc[0] if not modes.empty else "None"
            )

    input_data.update({
        "OverallQual": overall_qual,
        "GrLivArea": living_area,
        "BedroomAbvGr": bedrooms,
        "FullBath": bathrooms,
        "LotArea": lot_area,
        "YearBuilt": year_built,
        "GarageCars": garage_cars,
        "TotalBsmtSF": basement_area,
        "Neighborhood": neighborhood,
        "KitchenQual": kitchen_quality
    })

    input_df = pd.DataFrame([input_data], columns=features.columns)

    try:
        processed_input = preprocessor.transform(input_df)
        predicted_log_price = model.predict(processed_input)
        predicted_price = np.expm1(predicted_log_price[0])

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Your Estimated Property Value</div>
            <div class="result-price">${predicted_price:,.2f}</div>
            <div class="result-note">
                AI-powered estimate · Based on the trained machine learning model
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "Note: This is an estimated value, not an official property valuation."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    ESTATEIQ &nbsp; | &nbsp; House Price Prediction System<br><br>
    Developed with Python · Machine Learning · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
