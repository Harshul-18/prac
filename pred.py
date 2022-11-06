import pandas as pd
import streamlit as st
import pickle

model = pickle.load(open("model_ag.pkl", "rb"))

text_to_predict = pd.DataFrame({
  "title": [
    "Google may spur cloud cybersecurity M&A with $5.4B Mandiant buy",
    "Europe struggles to meet mounting needs of Ukraine's fleeing millions",
    "How the pandemic housing market spurred buyer's remorse across America",
  ]
})

predictions, output_directory = model.predict(text_to_predict)

print(predictions)
st.write(f"{predictions}")