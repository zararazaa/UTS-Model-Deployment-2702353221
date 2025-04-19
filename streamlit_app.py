import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

raw_dataset_path = 


def main():
    st.title('UTS Model Deployement - Zara Abigail Budiman - 2702353221)

    st.subheader("Please input data :3")
    user_input = {}

    # User input  
    for feature in feature_columns:
        if df[feature].dtype == 'object':
            user_input[feature] = st.selectbox(f"Select {feature}", df[feature].unique())
        else:
            min_val, max_val = float(df[feature].min()), float(df[feature].max())

            if feature == "Height":  
                value = st.slider(f"Enter {feature}", min_val, max_val, (min_val + max_val) / 2)
                user_input[feature] = round(value,2)
                
            else:  
                user_input[feature] = st.slider(f"Enter {feature}", int(min_val), int(max_val), int((min_val + max_val) / 2), step=1)

    if st.button("Predict"):
      model_filename = 'xgb.pkl'
      model=load_model(model_filename)
      processed_input = pre
