import streamlit as st
import pickle
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer

# Load XGBoost model from JSON
model = XGBClassifier()
model.load_model("xgb_best_model.json")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load CountVectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.title("Sentence Transformation Classifier")

original_sentence = st.text_input("Enter Original Sentence:")
transformed_sentence = st.text_input("Enter Transformed Sentence:")

if st.button("Predict"):
    if original_sentence.strip() == "" or transformed_sentence.strip() == "":
        st.warning("Please enter both sentences.")
    else:
        # Combine sentences
        combined_text = original_sentence + " " + transformed_sentence

        # Vectorize the input (CountVectorizer → 407-dim)
        embedding = vectorizer.transform([combined_text])

        # Predict
        prediction = model.predict(embedding)

        # Decode label
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        st.success(f"Predicted Class: {predicted_label}")
