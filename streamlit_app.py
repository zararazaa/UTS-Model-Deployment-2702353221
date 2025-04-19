import streamlit as st
import pandas as pd
import joblib

dataset_path = "Dataset_A_loan.csv"
model_filename = "trained_model.pkl"

def load_data():
    return pd.read_csv(dataset_path)

def load_model(filename):
    model = joblib.load(filename)
    return model

def predict_with_model(model, user_input):
    prediction = model.predict([user_input])
    return prediction[0]

def main():
    st.title("UTS Model Deployment")
    st.info("Predicting Loan Status")

    st.subheader("Please Input the Data")
    
    data = load_data()


    age = st.slider("Age", 20, 80, step=1)
    gender = st.selectbox("Gender", ("male", "female")
    education = st.selectbox("Last Education", ["High School", "Bachelor", "Associate", "Master", "Doctorate"])
    income = st.slider("Income", 8000, 500000, step=1)
    exp = st.slider("Years of Work Experience", 0, 75, step = 1)
    ownership = st.selectbox("Home Ownership Status", ["Rent", "Own", "Mortgage", "Other"])
    amount = st.slider("Loan Amount Requested", 500, 35000, step=1)
    intent = st.selectbox("Purpose of the loan", ["Personal", "Education", "Medical", "Venture", "Home Improvement", "Debt Consolidation"])
    rate = st.slider("Loan Interest Rate", 5, 20)
    income = st.selectbox("Loan Amount As a Percentage Of Annual Income",
    history = st.slider("Credit History Length In Years",2, 30, step=1)
    score = st.slider("Credit Score", 350, 850, step=1)
    prev = st.selectbox("Have you failed to repay a loan?", "Yes", "No")
    
                    
    user_input = {
        person_age = age
        person_gender = gender
        person_education = education
        person_income = income
        person_emp_exp = exp
        person_home_ownership = ownership
        loan_amnt = amount
        loan_intent = intent
        loan_int_rate = rate
        loan_percent_income = income
        cb_person_cred_hist_length = history
        credit_score = score
        precivious_loan_defaults_on_file = prev
    }

if st.button("Predict"):

    prev2 = 1 if prev == "Yes" else 0



    
    # for feature in feature_columns:
    #     if data[feature].dtype == 'object':
    #         options = data[feature].dropna().unique().tolist()
    #         user_input[feature] = st.selectbox(f"{feature}", options)
    #     else:
    #         min_val = data[feature].min()
    #         max_val = data[feature].max()
    #         default_val = data[feature].median()
    #         user_input[feature] = st.slider(f"{feature}", float(min_val), float(max_val), float(default_val))


if __name__ == "__main__":
    main()
