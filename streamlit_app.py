import streamlit as st
import joblib

def load_model(filename):
  model = joblib.load(filename)
  return model

def predict_with_model(model,user_input):
  prediction = model.predict([user_input])
  return prediction[0]


