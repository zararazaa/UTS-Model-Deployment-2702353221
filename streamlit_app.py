import streamlit as st
import pandas as pd
import pickle

# Load the encoders and model
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def load_encoder(filename):
    with open(filename, 'rb') as file:
        encoder = pickle.load(file)
    return encoder

def predict_with_model(model, user_input):
    prediction = model.predict(user_input)
    return prediction[0]

def preprocess_input(user_input, oe, ohe):
    # Apply ordinal encoding to 'person_education'
    user_input['person_education'] = oe.transform(user_input[['person_education']])

    # Apply one-hot encoding to 'person_home_ownership' and 'loan_intent'
    user_input = pd.get_dummies(user_input, columns=['person_home_ownership', 'loan_intent'], drop_first=True)

    # Return the preprocessed data
    return user_input

def main():
    st.title("UTS Model Deployment")
    st.info("Predicting Loan Status")

    st.subheader("Please Input the Data")

    # Input fields
    age = st.slider("Age", 20, 80, step=1)
    gender = st.selectbox("Gender", ("male", "female"))
    education = st.selectbox("Last Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    income = st.slider("Annual Income", 8000, 500000, step=1000)
    exp = st.slider("Years of Work Experience", 0, 75, step=1)
    ownership = st.selectbox("Home Ownership Status", ["Rent", "Own", "Mortgage", "Other"])
    amount = st.slider("Loan Amount Requested", 500, 35000, step=100)
    intent = st.selectbox("Purpose of the loan", ["Personal", "Education", "Medical", "Venture", "Home Improvement", "Debt Consolidation"])
    rate = st.slider("Loan Interest Rate (%)", 5, 20, step=1)
    loan_percent = st.slider("Loan Amount As % of Annual Income", 0, 100, step=1)
    history = st.slider("Credit History Length (Years)", 2, 30, step=1)
    score = st.slider("Credit Score", 350, 850, step=1)
    prev = st.selectbox("Have you failed to repay a loan before?", ("Yes", "No"))

    

    # Preprocess input data
    prev2 = 1 if prev == "Yes" else 0
    gender2 = 1 if gender == "female" else 0

    user_input = pd.DataFrame([{
        "person_age": age,
        "person_gender": gender2,
        "person_education": education,
        "person_income": income,
        "person_emp_exp": exp,
        "person_home_ownership": ownership,
        "loan_amnt": amount,
        "loan_intent": intent,
        "loan_int_rate": rate,
        "loan_percent_income": loan_percent,  # if model expects 0–1
        "cb_person_cred_hist_length": history,
        "credit_score": score,
        "previous_loan_defaults_on_file": prev2
    }])

    user_input = user_input[model.get_booster().feature_names]
    
    # Load encoders and model
    oe = load_encoder("oe.pkl")  # Ordinal Encoder
    ohe = load_encoder("ohe.pkl")  # OneHotEncoder
    model = load_model("xgb.pkl")  # XGBoost Model

    # Preprocess user input (apply ordinal and one-hot encoding)
    user_input = preprocess_input(user_input, oe, ohe)
    user_input = user_input[model.get_booster().feature_names]
    
    if st.button("Predict"):
        # Make prediction
        prediction = predict_with_model(model, user_input)
        prediction_label = "Accepted" if prediction == 1 else "Rejected"
        st.success(f"Loan status prediction: **{prediction_label}**")

if __name__ == "__main__":
    main()
