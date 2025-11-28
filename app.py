import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('loan_prediction.pkl', 'rb'))

st.title("Loan Approval Prediction")

gender = st.selectbox('Gender', ['Male','Female'])
married = st.selectbox('Married', ['No','Yes'])
dependents = st.selectbox('Dependents', ['0','1','2','3+'])
education = st.selectbox('Education', ['Graduate','Not Graduate'])
self_employed = st.selectbox('Self Employed', ['No','Yes'])
applicant_income = st.number_input('Applicant Income', min_value=0)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0)
loan_amount = st.number_input('Loan Amount', min_value=0)
loan_amount_term = st.number_input('Loan Amount Term (in days)', min_value=0)
credit_history = st.selectbox('Credit History', ['0','1'])
property_area = st.selectbox('Property Area', ['Urban','Rural','Semiurban'])

def preprocess_input():
    gender_val = 1 if gender == 'Male' else 0
    married_val = 1 if married == 'Yes' else 0
    dep = 3 if dependents == '3+' else int(dependents)
    education_val = 1 if education == 'Graduate' else 0
    self_emp_val = 1 if self_employed == 'Yes' else 0
    credit_history_val = int(credit_history)
    prop_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
    prop_area_val = prop_map[property_area]

    features = [gender_val, married_val, dep, education_val, self_emp_val,
                applicant_income, coapplicant_income, loan_amount,
                loan_amount_term, credit_history_val, prop_area_val]
    return np.array(features).reshape(1, -1)

if st.button('Predict Loan Status'):
    input_data = preprocess_input()
    result = model.predict(input_data)[0]
    
    if result == 'Y' or result == 1:
        st.success('🎉 Loan Approved')
    else:
        st.error('❌ Loan Rejected')
