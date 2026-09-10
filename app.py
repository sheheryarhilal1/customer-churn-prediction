
import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Load Saved Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("churn_pipeline.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load churn_pipeline.pkl")
    st.info("Make sure churn_pipeline.pkl is in the same folder as app.py.")
    st.stop()


# --------------------------------------------------
# Title and Description
# --------------------------------------------------
st.title("📊 Customer Churn Prediction")

st.write(
    """
    This application predicts whether a customer is likely to churn
    based on their demographic, subscription, billing, and usage information.

    Enter the customer details below and click **Predict Churn** to get
    the prediction and probability.
    """
)

st.divider()


# --------------------------------------------------
# User Input
# --------------------------------------------------
st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)


with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    region = st.selectbox(
        "Region",
        ["East", "West", "North", "South"]
    )

    tenure_months = st.slider(
        "Tenure (Months)",
        min_value=1,
        max_value=120,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0,
        step=0.01
    )


with col2:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=600.0,
        step=0.01
    )

    contract_type = st.selectbox(
        "Contract Type",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No"]
    )


with col3:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Bank transfer",
            "Credit card",
            "Mailed check"
        ]
    )

    num_support_calls = st.slider(
        "Number of Support Calls",
        min_value=0,
        max_value=20,
        value=1
    )

    late_payments_last_year = st.slider(
        "Late Payments Last Year",
        min_value=0,
        max_value=20,
        value=1
    )

    avg_monthly_usage_gb = st.number_input(
        "Average Monthly Usage (GB)",
        min_value=0.0,
        value=150.0,
        step=0.1
    )


st.divider()


# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔮 Predict Churn", use_container_width=True):

    try:
        # Create input DataFrame
        new_customer = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "region": region,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "online_security": online_security,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "num_support_calls": num_support_calls,
            "late_payments_last_year": late_payments_last_year,
            "avg_monthly_usage_gb": avg_monthly_usage_gb
        }])

        # Make prediction
        prediction = model.predict(new_customer)[0]

        # Display prediction
        st.subheader("Prediction Result")

        if prediction == "Yes":
            st.error("⚠️ Churn Prediction: YES")

            # Probability
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(new_customer)[0]

                # Find probability of Yes
                classes = model.classes_
                churn_index = list(classes).index("Yes")
                churn_probability = probability[churn_index] * 100

                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.2f}%"
                )

                st.write(
                    f"The model estimates a **{churn_probability:.2f}%** "
                    "probability that this customer will churn."
                )

        else:
            st.success("✅ Churn Prediction: NO")

            # Probability
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(new_customer)[0]

                classes = model.classes_
                churn_index = list(classes).index("Yes")
                churn_probability = probability[churn_index] * 100

                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.2f}%"
                )

                st.write(
                    f"The model estimates a **{churn_probability:.2f}%** "
                    "probability that this customer will churn."
                )

    except Exception as e:
        st.error("An error occurred while making the prediction.")
        st.write("Please check the entered values and saved model.")
        st.exception(e)
