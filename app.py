import pandas as pd
import numpy as np
import pickle
import os
import streamlit as st
from sklearn.preprocessing import LabelEncoder
# Load the trained model
model_path = os.path.join(os.getcwd(), "classifier.pkl")
model = pickle.load(open(model_path, "rb"))
# model = pickle.load(open('C:/Users/user/ashish/ml/Model_big_mart/classifier.pkl', 'rb'))
encoder=LabelEncoder()

def predict_sales(input_data):
    data=encoder.fit_transform(input_data)
    data=np.array(data).reshape(1,-1)
    prediction = model.predict(data)
    return prediction[0]

# Streamlit app
def main():
    st.title("Big Mart Sales Prediction")
    # Input fields for user
    item_identifier = st.text_input("Item Identifier")
    item_weight = st.number_input("Item Weight", min_value=0.0, step=0.1)
    item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
    item_visibility = st.number_input("Item Visibility", min_value=0.0, step=0.01)
    item_type = st.selectbox("Item Type", ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household", "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast"])
    item_mrp = st.number_input("Item MRP", min_value=0.0, step=0.1)
    outlet_identifier = st.text_input("Outlet Identifier")
    outlet_establishment_year = st.number_input("Outlet Establishment Year", min_value=1900, max_value=2024, step=1)
    outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
    outlet_location_type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
    outlet_type = st.selectbox("Outlet Type", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]) 

    # Predict button    
    if st.button("Predict Sales"):
        input_data = [item_identifier, item_weight, item_fat_content, item_visibility, item_type, item_mrp, outlet_identifier, outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]
        prediction = predict_sales(input_data)
        st.success(f"Predicted Sales: {prediction}")
if __name__ == "__main__":
    main()
