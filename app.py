import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.title("🏦 Loan Approval Prediction App")

# Load saved model
model = joblib.load("loan_prediction.pkl")

# --- Input fields ---
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Co-applicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0)
Loan_Amount_Term = st.number_input("Loan Term (in days)", min_value=0)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# --- Preprocessing based on training ---
def preprocess():
    # Convert categorical to match training encoding
    data = {
        "Gender": 1 if Gender == "Male" else 0,
        "Married": 1 if Married == "Yes" else 0,
        "Dependents": 3 if Dependents == "3+" else int(Dependents),
        "Education": 0 if Education == "Graduate" else 1,
        "Self_Employed": 1 if Self_Employed == "Yes" else 0,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": 2 if Property_Area == "Urban" else (1 if Property_Area == "Semiurban" else 0)
    }

    # Convert to DataFrame **with exact correct column order**
    df = pd.DataFrame([data], columns=[
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ])

    return df

# --- Predict ---
if st.button("Predict Loan Approval"):
    input_df = preprocess()
    result = model.predict(input_df)[0]

    if result == 1:
        st.success("🎉 Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")
