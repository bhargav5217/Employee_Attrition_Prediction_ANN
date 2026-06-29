import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

# -----------------------------
# Load Model & Preprocessor
# -----------------------------
@st.cache_resource
def load_artifacts():

    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    model = tf.keras.models.load_model("model.keras")

    return preprocessor, model


try:
    preprocessor, model = load_artifacts()

except Exception as e:
    st.error(f"Unable to load model.\n\n{e}")
    st.stop()


# -----------------------------
# Title
# -----------------------------
st.title("👨‍💼 Employee Attrition Prediction")
st.write(
    "Predict whether an employee is **likely to leave** the company "
    "based on employee information."
)

st.divider()

# -----------------------------
# Input Form
# -----------------------------
with st.form("employee_form"):

    col1, col2 = st.columns(2)

    # ---------------- Left Column ----------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=70,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        years_at_company = st.number_input(
            "Years at Company",
            0,
            50,
            5
        )

        job_role = st.selectbox(
            "Job Role",
            [
                "Education",
                "Media",
                "Healthcare",
                "Technology",
                "Finance",
                "Sales",
                "Legal"
            ]
        )

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=1000,
            max_value=100000,
            value=5000
        )

        work_life_balance = st.selectbox(
            "Work-Life Balance",
            ["Poor", "Fair", "Good", "Excellent"]
        )

        job_satisfaction = st.selectbox(
            "Job Satisfaction",
            ["Low", "Medium", "High", "Very High"]
        )

        performance_rating = st.selectbox(
            "Performance Rating",
            ["Low", "Below Average", "Average", "High"]
        )

        promotions = st.number_input(
            "Number of Promotions",
            0,
            10,
            0
        )

        overtime = st.selectbox(
            "Overtime",
            ["Yes", "No"]
        )

        distance = st.number_input(
            "Distance from Home",
            1,
            100,
            10
        )

    # ---------------- Right Column ----------------

    with col2:

        education = st.selectbox(
            "Education Level",
            [
                "High School",
                "Associate Degree",
                "Bachelor’s Degree",
                "Master’s Degree",
                "PhD"
            ]
        )

        marital = st.selectbox(
            "Marital Status",
            [
                "Single",
                "Married",
                "Divorced"
            ]
        )

        dependents = st.number_input(
            "Number of Dependents",
            0,
            10,
            0
        )

        job_level = st.selectbox(
            "Job Level",
            [
                "Entry",
                "Mid",
                "Senior"
            ]
        )

        company_size = st.selectbox(
            "Company Size",
            [
                "Small",
                "Medium",
                "Large"
            ]
        )

        company_tenure = st.number_input(
            "Company Tenure",
            0,
            100,
            10
        )

        remote_work = st.selectbox(
            "Remote Work",
            [
                "Yes",
                "No"
            ]
        )

        leadership = st.selectbox(
            "Leadership Opportunities",
            [
                "Yes",
                "No"
            ]
        )

        innovation = st.selectbox(
            "Innovation Opportunities",
            [
                "Yes",
                "No"
            ]
        )

        reputation = st.selectbox(
            "Company Reputation",
            [
                "Poor",
                "Fair",
                "Good",
                "Excellent"
            ]
        )

        recognition = st.selectbox(
            "Employee Recognition",
            [
                "Low",
                "Medium",
                "High",
                "Very High"
            ]
        )

    threshold = st.slider(
        "Decision Threshold",
        0.1,
        0.9,
        0.5,
        0.05
    )

    submitted = st.form_submit_button(
        "Predict Attrition",
        use_container_width=True
    )

# -----------------------------
# Prediction
# -----------------------------
if submitted:

    employee = pd.DataFrame({

        "age":[age],
        "gender":[gender],
        "years_at_company":[years_at_company],
        "job_role":[job_role],
        "monthly_income":[monthly_income],
        "work_life_balance":[work_life_balance],
        "job_satisfaction":[job_satisfaction],
        "performance_rating":[performance_rating],
        "number_of_promotions":[promotions],
        "overtime":[overtime],
        "distance_from_home":[distance],
        "education_level":[education],
        "marital_status":[marital],
        "number_of_dependents":[dependents],
        "job_level":[job_level],
        "company_size":[company_size],
        "company_tenure":[company_tenure],
        "remote_work":[remote_work],
        "leadership_opportunities":[leadership],
        "innovation_opportunities":[innovation],
        "company_reputation":[reputation],
        "employee_recognition":[recognition]

    })

    try:

        X = preprocessor.transform(employee)

        probability = float(model.predict(X, verbose=0)[0][0])

        prediction = int(probability >= threshold)

        st.divider()

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Probability of Attrition",
                f"{probability:.2%}"
            )

        with col2:

            confidence = max(probability, 1-probability)

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2%}"
            )

        st.progress(probability)

        # -----------------------------
        # Risk Level
        # -----------------------------

        if probability < 0.35:
            risk = "🟢 Low Risk"

        elif probability < 0.65:
            risk = "🟡 Medium Risk"

        else:
            risk = "🔴 High Risk"

        st.write(f"### Risk Level: {risk}")

        # -----------------------------
        # Prediction
        # -----------------------------

        if prediction == 1:

            st.error(
                f"""
                ### ⚠ Employee Likely to Leave

                Estimated Probability:

                **{probability:.2%}**
                """
            )

            st.info(
                """
                ### Suggested HR Actions

                • Schedule a one-on-one discussion

                • Review career growth opportunities

                • Evaluate workload

                • Consider compensation review

                • Increase employee recognition
                """
            )

        else:

            st.success(
                f"""
                ### ✅ Employee Likely to Stay

                Probability of Staying:

                **{1-probability:.2%}**
                """
            )

        # -----------------------------
        # Employee Summary
        # -----------------------------

        with st.expander("Employee Information"):

            st.dataframe(employee, use_container_width=True)

    except Exception as e:

        st.error(e)

# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "Employee Attrition Prediction using a Deep Neural Network "
    "built with TensorFlow and deployed using Streamlit."
)