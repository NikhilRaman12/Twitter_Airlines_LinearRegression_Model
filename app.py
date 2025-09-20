import streamlit as st
import pickle
import pandas as pd

# Page config
st.set_page_config(
    page_title="Twitter Sentiment Score Predictor",
    layout="centered"
)

st.title("Twitter Sentiment Score Predictor")
st.subheader("Predict a sentiment score from airline-related tweets using Linear Regression")

st.markdown("""
This application uses a **Linear Regression** model trained on TF-IDF features and one-hot encoded metadata  
to predict a **numerical sentiment score** for airline-related tweets.
""")

# Load model and vectorizer
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Sample tweets
sample_tweets = [
    "I had a great flight with Delta!",
    "United lost my luggage again.",
    "The boarding process was okay, nothing special.",
    "Southwest staff were incredibly helpful.",
    "Terrible experience with American Airlines."
]

selected_sample = st.sidebar.selectbox("Choose a sample tweet:", [""] + sample_tweets)
tweet_input = st.text_input("Enter a tweet for sentiment scoring:")

tweet = tweet_input if tweet_input else selected_sample

# Prediction
if tweet:
    try:
        # Vectorize text only (assuming model was trained on text vectors)
        transformed = vectorizer.transform([tweet])
        prediction = model.predict(transformed)[0]
        st.markdown(f"**Predicted Sentiment Score:** `{round(prediction, 3)}`")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("""
**Author**: Nikhil Raman  
**Model**: Linear Regression  
**Deployment**: Streamlit Cloud  
**Stack**: Python, scikit-learn, pandas, TF-IDF  
**Purpose**: Numeric sentiment scoring for AIML portfolio
""")
