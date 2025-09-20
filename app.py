import streamlit as st

st.set_page_config(page_title="Twitter Sentiment Classifier", layout="centered")

st.title("✈️ Twitter Sentiment Classifier")
st.write("This app predicts sentiment from airline-related tweets.")

tweet = st.text_input("Enter a tweet:")
if tweet:
    st.success("Prediction: Positive (placeholder)")
