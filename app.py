import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load("loan_prediction.pkl")

st.title("💰 Loan Approval Prediction App")

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0, step=1)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0, step=1)
LoanAmount = st.number_input("Loan Amount", min_value=0, step=1)
Loan_Amount_Term = st.number_input("Loan Amount Term (in days)", min_value=0, step=1)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

def preprocess():

    gender = 1 if Gender == "Male" else 0
    married = 1 if Married == "Yes" else 0
    dependents = 3 if Dependents == "3+" else int(Dependents)
    education = 1 if Education == "Graduate" else 0
    self_emp = 1 if Self_Employed == "Yes" else 0

    property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    property_area = property_map[Property_Area]

    features = np.array([[gender, married, dependents, education, self_emp,
                          ApplicantIncome, CoapplicantIncome, LoanAmount,
                          Loan_Amount_Term, Credit_History, property_area]])

    return features

if st.button("Check Loan Status"):
    input_data = preprocess()
    result = model.predict(input_data)[0]  

    if result == "Y":
        st.success("🎉 Loan Approved!")
    else:
        st.error("❌ Loan Rejected")
