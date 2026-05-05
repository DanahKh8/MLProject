import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Luxury Timepiece Valuation Engine", page_icon="⌚", layout="wide")

# Load the saved model and features
model = joblib.load('watch_model.pkl')
model_features = joblib.load('model_features.pkl')

# Set up the web page
st.title("Luxury Timepiece Valuation Engine")
st.write("Enter the specifications of the watch to predict its secondary market value.")

# Create the user input fields
col1, col2 = st.columns(2)

with col1:
    # Used the top brands kept during our Cardinality Reduction
    brand = st.selectbox("Brand", ['Rolex', 'Omega', 'Patek Philippe', 'Audemars Piguet', 'Breitling', 'Tudor', 'Cartier', 'Panerai', 'Other'])
    case_material = st.selectbox("Case Material", ['Steel', 'Yellow gold', 'Rose gold', 'White gold', 'Titanium', 'Platinum', 'Other'])
    year = st.slider("Year of Production", 1950, 2024, 2015)

with col2:
    movement = st.selectbox("Movement", ['Automatic', 'Manual winding', 'Quartz'])
    condition = st.selectbox("Condition", ['New', 'Unworn', 'Very good', 'Good', 'Fair'])

# The Prediction Button
if st.button("Predict Valuation"):
    # 1. Create a blank dictionary with all our model's features set to 0
    input_data = {feature: 0 for feature in model_features}
    
    # 2. Add the numeric input
    input_data['Year of production'] = year
    
    # 3. Handle the categorical inputs
    if f'Brand_Cleaned_{brand}' in input_data:
        input_data[f'Brand_Cleaned_{brand}'] = 1
    if f'Case material_Cleaned_{case_material}' in input_data:
        input_data[f'Case material_Cleaned_{case_material}'] = 1
    if f'Movement_{movement}' in input_data:
        input_data[f'Movement_{movement}'] = 1
    if f'Condition_{condition}' in input_data:
        input_data[f'Condition_{condition}'] = 1
        
    # 4. Convert to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 5. Predict and convert from Log_Price back to actual Dollars
    log_prediction = model.predict(input_df)[0]
    actual_price = np.expm1(log_prediction)
    
    # 6. Display the result
    st.success(f"### Estimated Market Value: ${actual_price:,.2f}")
    st.balloons()