import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('tpot_model.pkl')

st.title('Machine Learning Model Deployment with Streamlit')

# Sidebar for user input
st.sidebar.header('Input Features')

def user_input_features():
    feature1 = st.sidebar.number_input('Feature 1', min_value=0.0, max_value=100.0, value=50.0)
    feature2 = st.sidebar.number_input('Feature 2', min_value=0.0, max_value=100.0, value=50.0)
    # Add more features as necessary
    data = {'feature1': [feature1], 'feature2': [feature2]}
    return pd.DataFrame(data)

input_df = user_input_features()

st.write('Input Features:')
st.write(input_df)

if st.button('Predict'):
    prediction = model.predict(input_df)
    st.write(f'Prediction: {prediction[0]}')

if st.checkbox('Show model details'):
    st.write(model)
