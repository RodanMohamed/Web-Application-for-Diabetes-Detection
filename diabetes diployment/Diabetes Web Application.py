# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 10:26:05 2024

@author: Rodan Mohamed
"""

import numpy as np
import pickle
import streamlit as st

# Load the pre-trained model
model = pickle.load(open('C:/Users/ALRWOAD LABTOB/Documents/Study/diabetes diployment/trained_model.save', 'rb'))

# Function to make a diabetes prediction
def predict_diabetes(user_input):
    input_data_array = np.array(user_input)
    reshaped_data = input_data_array.reshape(1, -1)
    result = model.predict(reshaped_data)

    if result[0] == 0:
        return 'The individual is not diabetic. 😊'
    else:
        return 'The individual is diabetic. ⚠️'

# Main function for Streamlit app
def main():
    # Set page configuration
    st.set_page_config(page_title='Diabetes Detection App', page_icon='🩺', layout='wide')

    # Add sidebar for navigation and additional info
    st.sidebar.title("About the App")
    st.sidebar.info("""
    This is a simple web app designed to predict the likelihood of a person being diabetic based on their health metrics.
    Please provide the required information to check the diabetes status. ⚕️
    """)

    # Set the title with some color and alignment
    st.markdown("<h1 style='text-align: center; color: blue;'>Diabetes Detection Application 🩺🩸</h1>", unsafe_allow_html=True)

    # Description of the app
    st.markdown("""
    <p style='text-align: center;'>This app uses a machine learning model to predict whether a person has diabetes based on their health information. Enter the following details and click the button below to get your result:</p>
    """, unsafe_allow_html=True)

    # Create input fields with placeholders and emojis for better UI
    pregnancies = st.text_input('🤰 Number of Pregnancies:')
    glucose_level = st.text_input('🍬 Glucose Level:')
    blood_pressure = st.text_input('💉 Blood Pressure Reading:')
    skin_thickness = st.text_input('📏 Skin Thickness:')
    insulin_level = st.text_input('💉 Insulin Level:')
    bmi_value = st.text_input('⚖️ BMI Value:')
    pedigree_function = st.text_input('🔬 Diabetes Pedigree Function:')
    age = st.text_input('👤 Age:')

    # Initialize the variable for the diagnosis
    diagnosis_result = ''

    # Create a button for prediction with a progress bar
    if st.button('Check Diabetes Status 🧪'):
        with st.spinner('Processing... Please wait ⏳'):
            # Validate inputs (you can add more checks if needed)
            if not all([pregnancies, glucose_level, blood_pressure, skin_thickness, insulin_level, bmi_value, pedigree_function, age]):
                st.error('⚠️ Please fill in all the fields before proceeding!')
            else:
                diagnosis_result = predict_diabetes([pregnancies, glucose_level, blood_pressure, skin_thickness, insulin_level, bmi_value, pedigree_function, age])
                st.success(diagnosis_result)

    # Display diagnosis result
    if diagnosis_result:
        st.markdown(f"<h3 style='text-align: center; color: green;'>{diagnosis_result}</h3>", unsafe_allow_html=True)

    # Add a footer with some credits or description
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.text("Developed by Rodan Mohamed")
    st.sidebar.markdown("<small>Machine Learning Model trained with PIMA Indians Diabetes Database</small>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
