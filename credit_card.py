import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('best_tpot_model.pkl')

# Streamlit app
st.title('Fraud Detection Model Deployment')

# User inputs for each feature
category_gas_transport = st.number_input('Category Gas Transport', min_value=0, max_value=1)
category_grocery_pos = st.number_input('Category Grocery POS', min_value=0, max_value=1)
category_shopping_pos = st.number_input('Category Shopping POS', min_value=0, max_value=1)
category_home = st.number_input('Category Home', min_value=0, max_value=1)
category_shopping_net = st.number_input('Category Shopping Net', min_value=0, max_value=1)
category_entertainment = st.number_input('Category Entertainment', min_value=0, max_value=1)
cc_num = st.number_input('Credit Card Number')
amt = st.number_input('Amount')
zip_code = st.number_input('Zip Code')
lat = st.number_input('Latitude')
long = st.number_input('Longitude')
city_pop = st.number_input('City Population')
merch_lat = st.number_input('Merchant Latitude')
merch_long = st.number_input('Merchant Longitude')

# Create DataFrame with user inputs
input_data = pd.DataFrame({
    'category_gas_transport': [category_gas_transport],
    'category_grocery_pos': [category_grocery_pos],
    'category_shopping_pos': [category_shopping_pos],
    'category_home': [category_home],
    'category_shopping_net': [category_shopping_net],
    'category_entertainment': [category_entertainment],
    'cc_num': [cc_num],
    'amt': [amt],
    'zip': [zip_code],
    'lat': [lat],
    'long': [long],
    'city_pop': [city_pop],
    'merch_lat': [merch_lat],
    'merch_long': [merch_long]
})

# Predict using the model
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f'Prediction: {prediction[0]}')
