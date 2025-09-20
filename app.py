import streamlit as st
import pickle
import pandas as pd

# Page config
st.set_page_config(
    page_title="Twitter Sentiment Classifier",
    layout="centered"
)

# Title and description
st.title("Twitter Sentiment Classifier")
st.subheader("Analyze sentiment from airline-related tweets using machine learning")

st.markdown("""
This application uses a logistic regression model trained on TF-IDF features to classify tweets into **positive**, **negative**, or **neutral** sentiments.  
Enter a tweet below or select a sample to see the prediction.
""")

# Load model and vectorizer
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    st.error("Model files not found. Please ensure 'model.pkl' and 'vectorizer.pkl' are in the root directory.")
    st.stop()

# Sample tweets
sample_tweets = [
    "I had a great flight with Delta!",
    "United lost my luggage again.",
    "The boarding process was okay, nothing special.",
    "Southwest staff were incredibly helpful.",
    "Terrible experience with American Airlines."
]

selected_sample = st.selectbox("Or choose a sample tweet:", [""] + sample_tweets)

# Text input
tweet_input = st.text_input("Enter a tweet for sentiment analysis:")

# Use sample if selected
tweet = tweet_input if tweet_input else selected_sample

# Prediction
if tweet:
    try:
        transformed = vectorizer.transform([tweet])
        prediction = model.predict(transformed)[0]
        st.markdown(f"**Predicted Sentiment:** `{prediction}`")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("""
**Author**: Nikhil Raman  
**Model**: Logistic Regression  
**Deployment**: Streamlit Cloud  
**Stack**: Python, scikit-learn, pandas, TF-IDF
""")
