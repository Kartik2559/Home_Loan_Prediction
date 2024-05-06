import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the machine learning model
loan_model = pickle.load(open('home_app_model.pkl', 'rb'))

# Define a function to preprocess the input data
def preprocess_input(data):
    to_numeric = {'Male': 1, 'Female': 0,
                  'Yes': 1, 'No': 0,
                  'Graduate': 1, 'Not Graduate': 0,
                  'Urban': 2, 'Semiurban': 1, 'Rural': 0,
                  '1': 1, '0': 0,
                  '3+': 3}
    # Map categorical variables to numeric values
    data = data.applymap(lambda label: to_numeric[label] if label in to_numeric else label)
    # Fill missing values with 0
    data.fillna(0, inplace=True)
    return data

# Streamlit app
def main():
    st.title('Loan Approval Prediction by Kartik')
    st.subheader('Enter Applicant Details:')

    # Input fields
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Marital Status', ['Yes', 'No'])
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    applicant_income = st.number_input('Applicant Income')
    coapplicant_income = st.number_input('Coapplicant Income')
    loan_amount = st.number_input('Loan Amount')
    loan_amount_term = st.number_input('Loan Amount Term')
    credit_history = st.selectbox('Credit History', ['1', '0'])
    property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])
    dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, step=1)

    # Predict button
    if st.button('Predict Loan Approval'):
        # Create a dictionary with user input
        input_data = {
            'Gender': [gender],
            'Married': [married],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area],
            'Dependents': [dependents]
        }

        # Convert dictionary to DataFrame
        input_df = pd.DataFrame(input_data)

        # Preprocess input data
        preprocessed_input = preprocess_input(input_df)

        # Make prediction
        loan_prediction = loan_model.predict(preprocessed_input)

        # Display prediction result
        if loan_prediction[0] == 1:
            st.success('Congratulations! Your loan is likely to be approved.')
        else:
            st.error('Sorry, your loan is unlikely to be approved.')

if __name__ == '__main__':
    main()