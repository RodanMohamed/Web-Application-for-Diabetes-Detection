# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 10:21:45 2024

@author: Rodan Mohamed
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('C:/Users/ALRWOAD LABTOB/Documents/Study/diabetes diployment/trained_model.save', 'rb'))


# Input data for prediction (you can modify these values)
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Convert the input data to a numpy array
input_data_as_numpy_array = np.array(input_data)

# Reshape the array for a single prediction
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make the prediction using the loaded model
prediction = loaded_model.predict(input_data_reshaped)

# Print the predicted value
print(f"Prediction result: {prediction[0]}")

# Determine the result based on the prediction
if prediction[0] == 0:
    print('The person is not diabetic.')
else:
    print('The person is diabetic.')













