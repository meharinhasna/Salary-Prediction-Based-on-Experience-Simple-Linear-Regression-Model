# import libary
import pandas 
import pickle
from sklearn.metrics import r2_score
import streamlit as st

# set config
st.set_page_config(page_title="Salary Prediction Model" , page_icon=":robot:")


# load model
with open("model.pickle" , "rb") as file:
    model = pickle.load(file)

# load scale
with open("scale.pickle" , "rb") as file:
    scale = pickle.load(file)

# title
st.title("Salary Prediction Model")


# user input
year_ex = st.number_input("Enter Years of Experience:" , min_value=0.0, max_value=50.0, step=0.5)

# Predict button
if st.button("Predict Salary"):
    new_data = [[year_ex]]
    new_data_scaled = scale.transform(new_data)
    # prediction
    prediction = model.predict(new_data_scaled)
    st.success(f"Predicted Salary: {prediction[0]:,.2f}")