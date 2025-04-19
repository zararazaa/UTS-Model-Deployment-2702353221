import streamlit as st
import pandas as pd
import joblib
import pickle

dataset_path = "Dataset_A_loan.csv"


def load_data():
    return pd.read_csv(dataset_path)

def load_model(filename):
    with open(filename, 'rb') as file:
        model = joblib.load(file)
    return model

def predict_with_model(model, user_input):
    prediction = model.predict([user_input])
    return prediction[0]

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

    prev2 = 1 if prev == "Yes" else 0
    gender2 = 1 if gender == "female" else 0
    education2 = 0 if education == "High School" else (1 if education == "Bachelor" else (2 if education == "Associate" else (3 if education == "Master" else 4)))
    ownership2 = 0 if ownership == "Rent" else (1 if ownership == "Own" else (2 if ownership == "Mortgage" else 3))
    intent2 = 0 if intent == "Venture" else (1 if intent == "Education" else (2 if intent == "Medical" else (3 if intent == "Personal" else (4 if intent == "Home Improvement" else 5))))


    st.write("Check if it is already correct:", user_input)
    model = load_model(model_filename)
    
    if st.button("Predict"):
        user_input = pd.DataFrame({
        "person_age": age,
        "person_gender": gender2,
        "person_education": education2,
        "person_income": income,
        "person_emp_exp": exp,
        "person_home_ownership": ownership2,
        "loan_amnt": amount,
        "loan_intent": intent2,
        "loan_int_rate": rate,
        "loan_percent_income": loan_percent / 100,  # if model expects 0–1
        "cb_person_cred_hist_length": history,
        "credit_score": score,
        "previous_loan_defaults_on_file": prev2
    })

        model_path = "xgb.pkl"
        prediction = model.predict(user_input)
        st.success(f"Loan status prediction: **{prediction}**")

if __name__ == "__main__":
    main()
