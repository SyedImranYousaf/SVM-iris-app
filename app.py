import streamlit as st
import pandas as pd
import pickle

MODEL_PATH = "models/svm_iris.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# Load the model once
model = load_model()

st.title("Iris SVM Classifier")

# User inputs
sepal_length = st.number_input("Sepal Length", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal Width", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal Width", 0.0, 10.0, 0.2)

# Match training feature names
data = {
    'Id': 0,
    'SepalLengthCm': sepal_length,
    'SepalWidthCm': sepal_width,
    'PetalLengthCm': petal_length,
    'PetalWidthCm': petal_width
}

df = pd.DataFrame([data])

# Predict button
if st.button("Predict"):
    pred = model.predict(df)[0]
    st.success(f"Predicted Class: {pred}")
