import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load encoders and model
oe = joblib.load("oe.pkl")
ohe = joblib.load("ohe.pkl")
model = joblib.load("xgb.pkl")

# Load dataset to get feature info
df = pd.read_csv("Dataset_A_loan.csv")
feature_columns = df.drop(columns=["loan_status"]).columns

st.title('UTS Model Deployment - Zara Abigail Budiman - 2702353221')

st.subheader("Please input data")

# Get input from user
user_input = {}
for feature in feature_columns:
    if df[feature].dtype == 'object':
        user_input[feature] = st.selectbox(f"{feature}", df[feature].dropna().unique())
    else:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        user_input[feature] = st.slider(f"{feature}", min_val, max_val, (min_val + max_val) / 2)

if st.button("Predict"):
    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])

    # Preprocessing same as in OOP model
    # Binary encoding
    gender_map = {"male": 0, "female": 1}
    default_map = {"No": 0, "Yes": 1}
    input_df["person_gender"] = input_df["person_gender"].str.lower().map(gender_map)
    input_df["previous_loan_defaults_on_file"] = input_df["previous_loan_defaults_on_file"].map(default_map)

    # Ordinal Encoding
    input_df["person_education"] = oe.transform(input_df[["person_education"]])

    # One-hot encoding
    for col in ["person_home_ownership", "loan_intent"]:
        encoded = ohe.transform(input_df[[col]])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out([col]))
        input_df = input_df.drop(columns=[col])
        input_df = pd.concat([input_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Predict
    prediction = model.predict(input_df)
    st.success(f"Prediction: {prediction[0]}")
