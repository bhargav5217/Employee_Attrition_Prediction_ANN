import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

st.set_page_config(page_title="Employee Attrition Prediction", page_icon="👨‍💼", layout="wide")

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

st.title("👨‍💼 Employee Attrition Prediction")
st.caption("Fill in the employee details below and click Predict.")
st.divider()

with st.form("employee_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 70, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        years_at_company = st.number_input("Years at Company", 0, 50, 5)
        job_role = st.selectbox("Job Role", ["Education", "Media", "Healthcare", "Technology", "Finance", "Sales", "Legal"])
        monthly_income = st.number_input("Monthly Income", 1000, 100000, 5000)
        work_life_balance = st.selectbox("Work-Life Balance", ["Poor", "Fair", "Good", "Excellent"])
        job_satisfaction = st.selectbox("Job Satisfaction", ["Low", "Medium", "High", "Very High"])
        performance_rating = st.selectbox("Performance Rating", ["Low", "Below Average", "Average", "High"])
        promotions = st.number_input("Number of Promotions", 0, 10, 0)
        overtime = st.selectbox("Overtime", ["Yes", "No"])
        distance = st.number_input("Distance from Home (km)", 1, 100, 10)

    with col2:
        education = st.selectbox("Education Level", ["High School", "Associate Degree", "Bachelor's Degree", "Master's Degree", "PhD"])
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        dependents = st.number_input("Number of Dependents", 0, 10, 0)
        job_level = st.selectbox("Job Level", ["Entry", "Mid", "Senior"])
        company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
        company_tenure = st.number_input("Company Tenure (years)", 0, 100, 10)
        remote_work = st.selectbox("Remote Work", ["Yes", "No"])
        leadership = st.selectbox("Leadership Opportunities", ["Yes", "No"])
        innovation = st.selectbox("Innovation Opportunities", ["Yes", "No"])
        reputation = st.selectbox("Company Reputation", ["Poor", "Fair", "Good", "Excellent"])
        recognition = st.selectbox("Employee Recognition", ["Low", "Medium", "High", "Very High"])

    submitted = st.form_submit_button("Predict Attrition", use_container_width=True)

if submitted:
    employee = pd.DataFrame({
        "age": [age], "gender": [gender], "years_at_company": [years_at_company],
        "job_role": [job_role], "monthly_income": [monthly_income],
        "work_life_balance": [work_life_balance], "job_satisfaction": [job_satisfaction],
        "performance_rating": [performance_rating], "number_of_promotions": [promotions],
        "overtime": [overtime], "distance_from_home": [distance],
        "education_level": [education], "marital_status": [marital],
        "number_of_dependents": [dependents], "job_level": [job_level],
        "company_size": [company_size], "company_tenure": [company_tenure],
        "remote_work": [remote_work], "leadership_opportunities": [leadership],
        "innovation_opportunities": [innovation], "company_reputation": [reputation],
        "employee_recognition": [recognition]
    })

    try:
        X = preprocessor.transform(employee)
        probability = float(model.predict(X, verbose=0)[0][0])
        prediction = int(probability >= 0.5)

        st.divider()

        if prediction == 1:
            st.error("⚠️ This employee is likely to **leave**.")
        else:
            st.success("✅ This employee is likely to **stay**.")

        with st.expander("View Employee Data"):
            st.dataframe(employee, use_container_width=True)

    except Exception as e:
        st.error(e)