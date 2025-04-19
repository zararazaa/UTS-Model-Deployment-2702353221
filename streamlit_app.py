import streamlit as st
import pandas as pd
import joblib

# Path to dataset and model
dataset_path = "Dataset_A_loan.csv"
model_filename = "trained_model.pkl"

# Function to load the dataset
def load_data():
    return pd.read_csv(dataset_path)

# Function to load the model
def load_model(filename):
    model = joblib.load(filename)
    return model

# Function to make predictions
def predict_with_model(model, user_input):
    prediction = model.predict([user_input])
    return prediction[0]

# Streamlit app
def main():
    st.title("UTS Model Deployment")
    st.info("Predicting Loan Status")

    st.subheader("Please Input the Data")
    
    data = load_data()
    feature_columns = [
        col for col in data.columns if col != 'loan_status'
    ]  # Exclude target column

    user_input = {}
    
    for feature in feature_columns:
        if data[feature].dtype == 'object':
            options = data[feature].dropna().unique().tolist()
            user_input[feature] = st.selectbox(f"{feature}", options)
        else:
            min_val = data[feature].min()
            max_val = data[feature].max()
            default_val = data[feature].median()
            user_input[feature] = st.slider(f"{feature}", float(min_val), float(max_val), float(default_val))

    model = load_model(model_filename)
    if st.button("Predict"):
        # Convert to the same order as model expects
        input_list = [user_input[feature] for feature in feature_columns]
        prediction = predict_with_model(model, input_list)
        st.success(f"Loan Status Prediction: {prediction}")

# Fix the typo in if-statement
if __name__ == "__main__":
    main()
