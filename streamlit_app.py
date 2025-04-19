import streamlit as st
import joblib

dataset_path = "Dataset_A_loan.csv"

def load_data():
  return pd.read_csv(dataset_path)  

def load_model(filename):
  model = joblib.load(filename)
  return model

def predict_with_model(model,user_input):
  prediction = model.predict([user_input])
  return prediction[0]

def main():
  st.title("UTS Model Deployment")
  st.info('Predicting Loan Status')

  st.subheader("Please Input the Data :3")
  user_input = {}

  data = load_data()
  for feature in feature_columns:
    if data[feature].dtype == 'object':
      user_input[feature] = st.selectbox(f"Select {feature}", data[feature].max())
    else:
      user_input[feature] = st.slider(f"Enter {feature}", int(min_val), int(max_val), int((min_val + max_val) / 2, step=1)

  model_filename = 'trained_model.pkl'
  model = load_model(model_filename)
  prediction = predict_with_model(model, user_input)
  st.write("Loan status is", prediction)

if__name__=="__main__":
main()

