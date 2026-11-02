from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model" / "drive_logistic_model.pkl"


# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="DRIVE",
    page_icon="🚗",
    layout="wide",
)


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the saved Logistic Regression model package."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model file not found. Run 'python train_model.py' first."
        )

    return joblib.load(MODEL_FILE)


model_package = load_model()
model = model_package["model"]
features = model_package["features"]
cutoff = model_package["cutoff"]


# ---------------------------------------------------------
# Feature engineering functions
# ---------------------------------------------------------
def create_age_group(age: int) -> int:
    """Convert actual age into the model's age group."""

    if age < 30:
        return 1
    if age <= 45:
        return 2
    if age <= 60:
        return 3
    return 4


def create_age_car_segment(days: int) -> int:
    """Convert days since maintenance into a maintenance segment."""

    if days < 200:
        return 1
    if days <= 360:
        return 2
    if days <= 500:
        return 3
    return 4


def create_income_group(income: float) -> int:
    """
    Convert annual income into an Indian income quartile.

    These thresholds are based on the processed Indian dataset.
    """

    if income <= 856823.75:
        return 1
    if income <= 1125151.50:
        return 2
    if income <= 1438676.50:
        return 3
    return 4


def get_age_group_label(group: int) -> str:
    labels = {
        1: "Below 30 years",
        2: "30 to 45 years",
        3: "45 to 60 years",
        4: "Above 60 years",
    }
    return labels[group]


def get_income_group_label(group: int) -> str:
    labels = {
        1: "Lowest income quartile",
        2: "Lower-middle income quartile",
        3: "Upper-middle income quartile",
        4: "Highest income quartile",
    }
    return labels[group]


def get_age_car_label(segment: int) -> str:
    labels = {
        1: "Less than 200 days",
        2: "200 to 360 days",
        3: "361 to 500 days",
        4: "More than 500 days",
    }
    return labels[segment]


# ---------------------------------------------------------
# App heading
# ---------------------------------------------------------
st.title("🚗 DRIVE")
st.subheader("Data-Driven Insights for Vehicle Entry")

st.write(
    """
    DRIVE is an interactive decision-support application developed to estimate
    customer vehicle purchase likelihood and support ABG Motors in assessing
    potential market entry into India.
    """
)

st.divider()


# ---------------------------------------------------------
# Single customer prediction
# ---------------------------------------------------------
st.header("Single Customer Prediction")

with st.container(border=True):
    st.write("Enter the customer's details below.")

    col1, col2 = st.columns(2)

    with col1:
        current_age = st.number_input(
            "Current Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1,
        )

        annual_income = st.number_input(
            "Annual Income (₹)",
            min_value=0.0,
            value=900000.0,
            step=10000.0,
            format="%.2f",
        )

    with col2:
        days_since_maintenance = st.number_input(
            "Days Since Last Vehicle Maintenance",
            min_value=0,
            value=300,
            step=1,
        )

        gender = st.selectbox(
            "Gender",
            options=["Female", "Male"],
        )


# Convert raw user inputs into model inputs
age_group = create_age_group(current_age)
age_car_segment = create_age_car_segment(days_since_maintenance)
income_group = create_income_group(annual_income)
gender_code = 1 if gender == "Male" else 0


input_data = pd.DataFrame(
    [
        {
            "AGE_CAR SEGMENT": age_car_segment,
            "AGE_GROUP": age_group,
            "INCOME_GROUP": income_group,
            "GENDER_CODE": gender_code,
        }
    ]
)

input_data = input_data[features]


if st.button(
    "Predict Purchase Likelihood",
    type="primary",
):
    probability = model.predict_proba(input_data)[0, 1]
    prediction = int(probability >= cutoff)

    st.divider()

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            label="Purchase Probability",
            value=f"{probability:.2%}",
        )

    with metric_col2:
        st.metric(
            label="Prediction",
            value="Likely Buyer" if prediction == 1 else "Unlikely Buyer",
        )

    st.subheader("Customer Segment Summary")

    segment_col1, segment_col2, segment_col3, segment_col4 = st.columns(4)

    with segment_col1:
        st.metric(
            "Age Group",
            f"Group {age_group}",
        )
        st.caption(get_age_group_label(age_group))

    with segment_col2:
        st.metric(
            "Income Group",
            f"Group {income_group}",
        )
        st.caption(get_income_group_label(income_group))

    with segment_col3:
        st.metric(
            "Maintenance Segment",
            f"Segment {age_car_segment}",
        )
        st.caption(get_age_car_label(age_car_segment))

    with segment_col4:
        st.metric(
            "Gender Code",
            gender_code,
        )
        st.caption(f"{gender} = {gender_code}")

    if prediction == 1:
        st.success(
            "This customer is classified as a likely vehicle buyer."
        )

        st.info(
            "Recommended action: Prioritize the customer for targeted "
            "campaigns, personalized offers, and sales follow-up."
        )

    else:
        st.warning(
            "This customer is classified as an unlikely vehicle buyer."
        )

        st.info(
            "Recommended action: Use awareness campaigns or customer-nurturing "
            "strategies instead of immediate high-cost sales targeting."
        )

    st.caption(f"Classification cut-off used: {cutoff}")


# ---------------------------------------------------------
# Bulk customer prediction
# ---------------------------------------------------------
st.divider()
st.header("Bulk Customer Prediction")

st.write(
    """
    Upload the processed Indian customer dataset to estimate purchase
    probability for all customers at once.
    """
)

uploaded_file = st.file_uploader(
    "Upload the processed India Excel file",
    type=["xlsx"],
)


if uploaded_file is not None:
    try:
        india_data = pd.read_excel(
            uploaded_file,
            sheet_name="IN_Mobiles",
        )

        india_data.columns = india_data.columns.str.strip()

        required_columns = [
            "AGE_CAR SEGMENT",
            "AGE_GROUP",
            "INCOME_GROUP",
            "GENDER_CODE",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in india_data.columns
        ]

        if missing_columns:
            st.error(
                "The uploaded file is missing these required columns: "
                + ", ".join(missing_columns)
            )

        else:
            st.success(
                f"Dataset loaded successfully: {len(india_data):,} customers"
            )

            st.subheader("Dataset Preview")

            st.dataframe(
                india_data.head(10),
                use_container_width=True,
            )

            if st.button(
                "Predict All Customers",
                type="primary",
            ):
                bulk_input = india_data[features].copy()

                india_data["PYTHON_PROBABILITY"] = (
                    model.predict_proba(bulk_input)[:, 1]
                )

                india_data["PYTHON_PREDICTED_PURCHASE"] = (
                    india_data["PYTHON_PROBABILITY"] >= cutoff
                ).astype(int)

                total_customers = len(india_data)

                potential_buyers = int(
                    india_data["PYTHON_PREDICTED_PURCHASE"].sum()
                )

                potential_percentage = (
                    potential_buyers / total_customers
                ) * 100

                average_probability = india_data[
                    "PYTHON_PROBABILITY"
                ].mean()

                high_confidence_buyers = int(
                    (
                        india_data["PYTHON_PROBABILITY"] >= 0.75
                    ).sum()
                )

                # ---------------------------------------------------------
                # Prediction summary
                # ---------------------------------------------------------
                st.subheader("Prediction Summary")

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)

                with kpi1:
                    st.metric(
                        "Total Customers",
                        f"{total_customers:,}",
                    )

                with kpi2:
                    st.metric(
                        "Potential Buyers",
                        f"{potential_buyers:,}",
                    )

                with kpi3:
                    st.metric(
                        "Potential Buyer Percentage",
                        f"{potential_percentage:.2f}%",
                    )

                with kpi4:
                    st.metric(
                        "High-Confidence Buyers",
                        f"{high_confidence_buyers:,}",
                        help=(
                            "Customers with predicted purchase "
                            "probability of at least 75%."
                        ),
                    )

                st.caption(
                    "Average predicted purchase probability: "
                    f"{average_probability:.2%}"
                )

                # ---------------------------------------------------------
                # Prediction results table
                # ---------------------------------------------------------
                st.subheader("Prediction Results")

                result_columns = [
                    column
                    for column in [
                        "ID",
                        "CURR_AGE",
                        "GENDER",
                        "ANN_INCOME",
                        "AGE_CAR",
                        "AGE_CAR SEGMENT",
                        "AGE_GROUP",
                        "INCOME_GROUP",
                        "GENDER_CODE",
                        "PYTHON_PROBABILITY",
                        "PYTHON_PREDICTED_PURCHASE",
                    ]
                    if column in india_data.columns
                ]

                st.dataframe(
                    india_data[result_columns].head(100),
                    use_container_width=True,
                )

                # ---------------------------------------------------------
                # Dashboard preparation
                # ---------------------------------------------------------
                dashboard_data = india_data.copy()

                dashboard_data["PREDICTION_LABEL"] = dashboard_data[
                    "PYTHON_PREDICTED_PURCHASE"
                ].map(
                    {
                        0: "Unlikely Buyer",
                        1: "Likely Buyer",
                    }
                )

                dashboard_data["AGE_GROUP_LABEL"] = dashboard_data[
                    "AGE_GROUP"
                ].map(
                    {
                        1: "Below 30",
                        2: "30–45",
                        3: "45–60",
                        4: "Above 60",
                    }
                )

                dashboard_data["INCOME_GROUP_LABEL"] = dashboard_data[
                    "INCOME_GROUP"
                ].map(
                    {
                        1: "Lowest",
                        2: "Lower-middle",
                        3: "Upper-middle",
                        4: "Highest",
                    }
                )

                # ---------------------------------------------------------
                # Interactive dashboard
                # ---------------------------------------------------------
                st.divider()
                st.header("📊 India Market Prediction Dashboard")

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    probability_chart = px.histogram(
                        dashboard_data,
                        x="PYTHON_PROBABILITY",
                        nbins=20,
                        title="Purchase Probability Distribution",
                        labels={
                            "PYTHON_PROBABILITY": "Purchase Probability",
                            "count": "Number of Customers",
                        },
                    )

                    probability_chart.update_layout(
                        yaxis_title="Number of Customers",
                        bargap=0.05,
                    )

                    st.plotly_chart(
                        probability_chart,
                        use_container_width=True,
                    )

                with chart_col2:
                    gender_summary = (
                        dashboard_data.groupby(
                            ["GENDER", "PREDICTION_LABEL"]
                        )
                        .size()
                        .reset_index(name="CUSTOMER_COUNT")
                    )

                    gender_chart = px.bar(
                        gender_summary,
                        x="GENDER",
                        y="CUSTOMER_COUNT",
                        color="PREDICTION_LABEL",
                        barmode="group",
                        title="Gender-wise Purchase Predictions",
                        labels={
                            "GENDER": "Gender",
                            "CUSTOMER_COUNT": "Number of Customers",
                            "PREDICTION_LABEL": "Prediction",
                        },
                    )

                    st.plotly_chart(
                        gender_chart,
                        use_container_width=True,
                    )

                chart_col3, chart_col4 = st.columns(2)

                with chart_col3:
                    income_summary = (
                        dashboard_data.groupby(
                            [
                                "INCOME_GROUP_LABEL",
                                "PREDICTION_LABEL",
                            ]
                        )
                        .size()
                        .reset_index(name="CUSTOMER_COUNT")
                    )

                    income_chart = px.bar(
                        income_summary,
                        x="INCOME_GROUP_LABEL",
                        y="CUSTOMER_COUNT",
                        color="PREDICTION_LABEL",
                        barmode="stack",
                        title="Income Group-wise Predictions",
                        category_orders={
                            "INCOME_GROUP_LABEL": [
                                "Lowest",
                                "Lower-middle",
                                "Upper-middle",
                                "Highest",
                            ]
                        },
                        labels={
                            "INCOME_GROUP_LABEL": "Income Group",
                            "CUSTOMER_COUNT": "Number of Customers",
                            "PREDICTION_LABEL": "Prediction",
                        },
                    )

                    st.plotly_chart(
                        income_chart,
                        use_container_width=True,
                    )

                with chart_col4:
                    age_summary = (
                        dashboard_data.groupby(
                            [
                                "AGE_GROUP_LABEL",
                                "PREDICTION_LABEL",
                            ]
                        )
                        .size()
                        .reset_index(name="CUSTOMER_COUNT")
                    )

                    age_chart = px.bar(
                        age_summary,
                        x="AGE_GROUP_LABEL",
                        y="CUSTOMER_COUNT",
                        color="PREDICTION_LABEL",
                        barmode="stack",
                        title="Age Group-wise Predictions",
                        category_orders={
                            "AGE_GROUP_LABEL": [
                                "Below 30",
                                "30–45",
                                "45–60",
                                "Above 60",
                            ]
                        },
                        labels={
                            "AGE_GROUP_LABEL": "Age Group",
                            "CUSTOMER_COUNT": "Number of Customers",
                            "PREDICTION_LABEL": "Prediction",
                        },
                    )

                    st.plotly_chart(
                        age_chart,
                        use_container_width=True,
                    )

                # ---------------------------------------------------------
                # Download results
                # ---------------------------------------------------------
                csv_data = india_data.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Prediction Results",
                    data=csv_data,
                    file_name="DRIVE_India_Predictions.csv",
                    mime="text/csv",
                )

    except ValueError:
        st.error(
            "The workbook must contain a sheet named 'IN_Mobiles'."
        )

    except Exception as error:
        st.error(
            f"Unable to process the uploaded file: {error}"
        )


# ---------------------------------------------------------
# About section
# ---------------------------------------------------------
st.divider()
st.header("ℹ️ About DRIVE")

about_col1, about_col2 = st.columns(2)

with about_col1:
    st.subheader("Project Overview")

    st.write(
        """
        **DRIVE — Data-Driven Insights for Vehicle Entry** is an
        interactive predictive analytics application designed to support
        ABG Motors in assessing its potential entry into the Indian
        automobile market.

        The project uses historical Japanese customer purchase data to
        train a Logistic Regression model. The trained model is then
        applied to Indian customer records to estimate purchase
        probability and identify high-potential vehicle buyers.
        """
    )

    st.subheader("Internship Context")

    st.write(
        """
        This project was developed from a business case assigned during
        the Data Science with AI Internship conducted by Internshala.

        The datasets and business objective were provided as part of the
        internship, while data preparation, feature engineering, model
        implementation, evaluation, visualization, application
        development, interpretation, and documentation were completed
        as part of the project work.
        """
    )

with about_col2:
    st.subheader("Technologies Used")

    st.markdown(
        """
        - **Microsoft Excel** — data preparation, feature engineering,
          original model development, and validation
        - **Python** — predictive model implementation
        - **Scikit-learn** — Logistic Regression model
        - **Pandas** — dataset processing and bulk prediction
        - **Streamlit** — interactive decision-support interface
        - **Plotly** — interactive business charts
        - **Tableau** — exploratory analysis and business dashboard
        """
    )

    st.subheader("Model Details")

    st.markdown(
        f"""
        - **Algorithm:** Logistic Regression
        - **Target variable:** Purchase
        - **Classification cut-off:** {cutoff}
        - **Training records:** 28,000
        - **Testing records:** 12,000
        - **Python test accuracy:** 68.26%
        - **Python AUC:** 0.7428
        - **Model inputs:** Age Group, Income Group,
          Maintenance Segment, and Gender Code
        """
    )

st.info(
    """
    DRIVE is an academic decision-support prototype. Its predictions
    estimate customer purchase likelihood based only on the available
    dataset variables and should not be treated as guaranteed purchasing
    outcomes.
    """
)