#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:04:54 2022

@author: soundaryasubramanian
"""

import numpy as np 
import pickle 
import streamlit as st

loaded_model = pickle.load(open('/Users/soundaryasubramanian/Practise_Dataset/Diabetes Prediction using Machine Learning/trained_model_diabetics.sav','rb'))

#creating a function for prediction

def diab_prediction(input_data):

    # change the input data into numpy array 

    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance

    input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshape)
    print(prediction)

    if(prediction[0] == 0):
        return 'The person is  not diabetic'
    else : 
        return 'The person is diabetic'
        
    
    
def main():
    
    # Title
    st.title('Diabetes predictive web app')
    
    # input data from the user 
    
    Pregnancies = st.text_input('Count of Pregnancies :')
    Glucose = st.text_input('Glucose level : ')
    BloodPressure = st.text_input('Blood Pressure value : ')
    SkinThickness = st.text_input('SkinThickness value : ')
    Insulin = st.text_input('Insulin level :')
    BMI = st.text_input('BMI Value : ')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction : ')
    Age = st.text_input('Age of the person : ')
    
    
    
    
    # code for prediction 
    diagnosis = ''
    
    # creating a button for prediction 
    
    if st.button('Diabetes Test Result'):
        diagnosis = diab_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
        
    st.success(diagnosis)
    
    
    
    
if __name__ == '__main__' : 
    
    main()
    
    
    
    
    
    
