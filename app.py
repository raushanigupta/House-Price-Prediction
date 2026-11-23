import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="EstateIQ | House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# CUSTOM PROFESSIONAL CSS
# --------------------------------------------------
st.markdown(
    """
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
            linear-gradient(
                90deg,
                rgba(13, 39, 32, 0.96) 0%,
                rgba(13, 39, 32, 0.80) 48%,
                rgba(13, 39, 32, 0.20) 100%
            ),
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
        border: 1px solid rgba(232, 210, 168, 0.5);
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
        color: #263D32 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
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

    /* FORM SUBHEADINGS */
    .stApp h4 {
        background: #153F35;
        color: #FFFFFF !important;
        padding: 13px 17px;
        border-radius: 10px;
        font-size: 17px !important;
        font-weight: 600 !important;
        border-left: 5px solid #C9A66B;
        margin-bottom: 22px;
    }

    /* BUTTON */
    .stButton > button {
        width: 100%;
        min-height: 48px;
        border: none;
        border-radius: 12px;
        background: linear-gradient(100deg, #153F35, #26765A);
        color: white;
        font-size: 16px;
        font-weight: 700;
        transition: 0.25s ease;
        box-shadow: 0 7px 18px rgba(21, 63, 53, 0.18);
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
        box-shadow: 0 12px 30px rgba(21, 63, 53, 0.16);
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

    /* ANALYTICS */
    .analytics-heading {
        font-family: 'Playfair Display', serif;
        color: #153F35;
        font-size: 30px;
        margin-bottom: 5px;
    }

    .analytics-description {
        color: #777F79;
        margin-bottom: 22px;
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E8E9E2;
        border-radius: 14px;
        padding: 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #52645A;
    }

    div[data-testid="stMetricValue"] {
        color: #153F35;
        font-weight: 700;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #E7EDE6;
        border-radius: 8px 8px 0 0;
        padding: 12px 16px;
        color: #153F35;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: #153F35 !important;
        color: white !important;
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

    @media (max-width: 768px) {
        .hero {
            min-height: 320px;
            padding: 28px;
            background-position: center;
        }

        .hero h1 {
            font-size: 34px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_path = os.path.join("models", "best_model.pkl")
    preprocessor_path = os.path.join("models", "preprocessor.pkl")
    data_path = os.path.join("data", "train.csv")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    data = pd.read_csv(data_path)

    return model, preprocessor, data


try:
    model, preprocessor, df = load_artifacts()
except Exception as error:
    st.error(
        "Unable to load the model. Please check that the models and data "
        "folders contain the required files."
    )
    st.exception(error)
    st.stop()


if "SalePrice" not in df.columns or "Id" not in df.columns:
    st.error("The training dataset must contain 'SalePrice' and 'Id' columns.")
    st.stop()

features = df.drop(columns=["SalePrice", "Id"])


# --------------------------------------------------
# HELPER FUNCTIONS FOR ANALYTICS
# --------------------------------------------------
def show_report_image(filename, caption):
    image_path = os.path.join("reports", filename)

    if os.path.exists(image_path):
        st.image(
            image_path,
            caption=caption,
            use_container_width=True,
        )
    else:
        st.info(f"{filename} is not available in the reports folder.")


def show_report_csv(filename):
    csv_path = os.path.join("reports", filename)

    if os.path.exists(csv_path):
        report_df = pd.read_csv(csv_path)
        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(f"{filename} is not available in the reports folder.")


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-content">
            <div class="hero-tag">ESTATEIQ &middot; SMART PROPERTY VALUATION</div>
            <h1>Find the True Value<br>of Your Dream Home.</h1>
            <p>Discover an estimated property value using machine learning. Enter your property details and get an intelligent price estimate.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
# --------------------------------------------------
# PROPERTY FORM
# --------------------------------------------------
st.markdown(
    """
    <div class="section-heading">Property Details</div>
    <div class="section-description">
        Tell us about the property to generate your estimated house price.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 🏠 Property Information")

        overall_qual = st.slider(
            "Overall Quality",
            min_value=1,
            max_value=10,
            value=5,
            key="overall_quality_slider",
        )

        living_area = st.number_input(
            "Living Area (sq ft)",
            min_value=300,
            max_value=10000,
            value=int(features["GrLivArea"].median()),
            key="living_area_input",
        )

        bedrooms = st.number_input(
            "Number of Bedrooms",
            min_value=0,
            max_value=15,
            value=int(features["BedroomAbvGr"].median()),
            key="bedrooms_input",
        )

        bathrooms = st.number_input(
            "Full Bathrooms",
            min_value=0,
            max_value=10,
            value=int(features["FullBath"].median()),
            key="bathrooms_input",
        )

        lot_area = st.number_input(
            "Lot Area (sq ft)",
            min_value=500,
            max_value=100000,
            value=int(features["LotArea"].median()),
            key="lot_area_input",
        )

    with col2:
        st.markdown("#### 📍 Location & Facilities")

        year_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=2026,
            value=int(features["YearBuilt"].median()),
            key="year_built_input",
        )

        garage_cars = st.selectbox(
            "Garage Capacity",
            [0, 1, 2, 3, 4],
            index=2,
            key="garage_capacity_select",
        )

        basement_area = st.number_input(
            "Basement Area (sq ft)",
            min_value=0,
            max_value=5000,
            value=int(features["TotalBsmtSF"].median()),
            key="basement_area_input",
        )

        neighborhood_options = sorted(
            features["Neighborhood"].dropna().unique().tolist()
        )

        neighborhood = st.selectbox(
            "Neighborhood",
            neighborhood_options,
            key="neighborhood_select",
        )

        kitchen_quality = st.selectbox(
            "Kitchen Quality",
            ["Ex", "Gd", "TA", "Fa", "Po"],
            key="kitchen_quality_select",
        )

    st.markdown("---")

    predict = st.button(
        "✦  Predict Property Value",
        use_container_width=True,
        key="predict_property_value_button",
    )


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

    input_data.update(
        {
            "OverallQual": overall_qual,
            "GrLivArea": living_area,
            "BedroomAbvGr": bedrooms,
            "FullBath": bathrooms,
            "LotArea": lot_area,
            "YearBuilt": year_built,
            "GarageCars": garage_cars,
            "TotalBsmtSF": basement_area,
            "Neighborhood": neighborhood,
            "KitchenQual": kitchen_quality,
        }
    )

    input_df = pd.DataFrame([input_data], columns=features.columns)

    try:
        processed_input = preprocessor.transform(input_df)
        predicted_value = model.predict(processed_input)[0]

        # The training target was log-transformed using log1p.
        predicted_price = float(np.expm1(predicted_value))

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    Your Estimated Property Value
                </div>
                <div class="result-price">
                    ${predicted_price:,.2f}
                </div>
                <div class="result-note">
                    AI-powered estimate · Based on the trained machine learning model
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Note: This is an estimated value based on the Ames Housing dataset, "
            "not an official property valuation or a live market quote."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# MODEL INSIGHTS AND ANALYTICS DASHBOARD
# --------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <div class="analytics-heading">
        Model Insights &amp; Analytics
    </div>
    <div class="analytics-description">
        Explore the dataset, evaluate model performance, compare algorithms,
        and understand the features influencing house prices.
    </div>
    """,
    unsafe_allow_html=True,
)

tab_eda, tab_evaluation, tab_comparison, tab_tuning, tab_features = st.tabs(
    [
        "📊 EDA",
        "🎯 Model Evaluation",
        "⚖️ Model Comparison",
        "🛠️ Hyperparameter Tuning",
        "💡 Feature Importance",
    ]
)


# --------------------------------------------------
# TAB 1: EXPLORATORY DATA ANALYSIS
# --------------------------------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    st.write(
        "Explore the patterns and relationships present in the house price dataset."
    )

    show_report_image(
        "price_distribution.png",
        "Distribution of House Prices",
    )

    st.markdown("---")

    show_report_image(
        "area_vs_price.png",
        "Living Area vs House Price",
    )

    st.markdown("---")

    show_report_image(
        "correlation_heatmap.png",
        "Correlation Heatmap of Numerical Features",
    )


# --------------------------------------------------
# TAB 2: MODEL EVALUATION
# --------------------------------------------------
with tab_evaluation:
    st.subheader("Model Performance Evaluation")
    st.write(
        "The following metrics are from the model's validation evaluation."
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            label="Mean Absolute Error (MAE)",
            value="$14,900.01",
        )

    with metric2:
        st.metric(
            label="Root Mean Squared Error (RMSE)",
            value="$22,741.00",
        )

    with metric3:
        st.metric(
            label="Validation R² Score",
            value="0.9326",
        )

    st.info(
        "An R² score of 0.9326 means the model explains approximately 93.26% "
        "of the variation in the validation target values. It does not mean "
        "93.26% prediction accuracy."
    )

    st.markdown("---")
    st.subheader("Actual vs Predicted Prices")

    show_report_image(
        "actual_vs_predicted.png",
        "Actual House Prices Compared with Predicted Prices",
    )

    st.markdown("---")
    st.subheader("Residual Distribution")

    show_report_image(
        "residual_distribution.png",
        "Distribution of Prediction Errors (Residuals)",
    )


# --------------------------------------------------
# TAB 3: MODEL COMPARISON
# --------------------------------------------------
with tab_comparison:
    st.subheader("Machine Learning Model Comparison")
    st.write(
        "Compare the performance of the machine learning algorithms "
        "trained for this project."
    )

    show_report_image(
        "model_comparison_visualization.png",
        "Model Performance Comparison",
    )

    st.markdown("---")
    st.subheader("Model Comparison Results")
    show_report_csv("model_comparison.csv")


# --------------------------------------------------
# TAB 4: HYPERPARAMETER TUNING
# --------------------------------------------------
with tab_tuning:
    st.subheader("Hyperparameter Optimization")
    st.write(
        "RandomizedSearchCV with 3-fold cross-validation was used "
        "to experiment with Gradient Boosting and XGBoost parameters."
    )

    st.markdown("#### Tuning Results")
    show_report_csv("tuning_comparison.csv")

    st.markdown("---")

    st.info(
        "In the completed tuning experiment, the existing model remained "
        "the best-performing model. The tuned Gradient Boosting and XGBoost "
        "models did not improve the validation RMSE."
    )


# --------------------------------------------------
# TAB 5: FEATURE IMPORTANCE
# --------------------------------------------------
with tab_features:
    st.subheader("Feature Importance Analysis")
    st.write(
        "Feature importance helps identify which input features contribute "
        "more to the model's predictions."
    )

    show_report_image(
        "feature_importance.png",
        "Feature Importance Visualization",
    )

    st.markdown("---")
    st.subheader("Feature Importance Values")
    show_report_csv("feature_importance.csv")


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------
st.markdown("---")

with st.expander("ℹ️ About EstateIQ"):
    st.write(
        """
        **EstateIQ** is a machine learning-based house price prediction project.

        The project uses the Ames Housing dataset and applies data preprocessing,
        exploratory data analysis, model training, model evaluation,
        hyperparameter tuning, and feature importance analysis.

        The prediction interface accepts selected property details.
        Other model inputs are filled using dataset median or mode values.

        **Dataset:** Ames Housing Dataset

        **Target:** SalePrice (USD)

        **Technologies:** Python, Pandas, NumPy, Scikit-learn,
        XGBoost, Matplotlib, Seaborn, Joblib and Streamlit.
        """
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        ESTATEIQ &nbsp; | &nbsp; House Price Prediction System
        <br><br>
        Developed with Python · Machine Learning · Scikit-learn · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
