import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved scaler and TPOT model 
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('tpot_model.pkl', 'rb') as f:
    tpot_model = pickle.load(f)

# Define the input fields for user to enter
st.title("Fraud Detection Prediction")
st.write("Enter the following details to predict whether a transaction is fraudulent:")

# Dropdown for categorical selection
category = st.selectbox(
    "Select a category:",
    ['category_gas_transport', 'category_grocery_pos', 'category_shopping_pos', 
     'category_home', 'category_shopping_net', 'category_entertainment']
)

# Manually set other categories to 0
categories = {
    'category_gas_transport': 0,
    'category_grocery_pos': 0,
    'category_shopping_pos': 0,
    'category_home': 0,
    'category_shopping_net': 0,
    'category_entertainment': 0
}
categories[category] = 1  # Set selected category to 1

# Input for numeric features
cc_num = st.number_input("Credit Card Number")
amt = st.number_input("Amount")
zip_code = st.number_input("ZIP Code")
lat = st.number_input("Latitude")
long = st.number_input("Longitude")
city_pop = st.number_input("City Population")
merch_lat = st.number_input("Merchant Latitude")
merch_long = st.number_input("Merchant Longitude")

# Combine all inputs into a dataframe
input_data = pd.DataFrame({
    'category_gas_transport': [categories['category_gas_transport']],
    'category_grocery_pos': [categories['category_grocery_pos']],
    'category_shopping_pos': [categories['category_shopping_pos']],
    'category_home': [categories['category_home']],
    'category_shopping_net': [categories['category_shopping_net']],
    'category_entertainment': [categories['category_entertainment']],
    'cc_num': [cc_num],
    'amt': [amt],
    'zip': [zip_code],
    'lat': [lat],
    'long': [long],
    'city_pop': [city_pop],
    'merch_lat': [merch_lat],
    'merch_long': [merch_long],
})

# Apply scaling only to the numeric columns
numeric_columns = ['cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])

# Predict using the TPOT model
if st.button("Predict"):
    prediction = tpot_model.predict(input_data)
    result = "Fraudulent" if prediction[0] == 1 else "Legitimate"
    st.write(f"The transaction is likely: {result}")
