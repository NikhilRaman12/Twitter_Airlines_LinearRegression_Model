import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Twitter Sentiment Classifier",
    layout="centered"
)

# Header
st.title("Twitter Sentiment Classifier")
st.subheader("Real-time sentiment analysis of airline-related tweets using Logistic Regression")

st.markdown("""
This application uses a **Logistic Regression** model trained on **TF-IDF** features to classify tweets into  
**positive**, **neutral**, or **negative** sentiments.  
Enter a tweet below or select a sample to see the prediction.
""")

# Load model and vectorizer
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'model.pkl' and 'vectorizer.pkl' are present.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sample tweets
sample_tweets = [
    "I had a great flight with Delta!",
    "United lost my luggage again.",
    "The boarding process was okay, nothing special.",
    "Southwest staff were incredibly helpful.",
    "Terrible experience with American Airlines.",
    "JetBlue was punctual and clean.",
    "Flight delayed for 5 hours. Never flying again."
]

# Sidebar for sample selection
st.sidebar.header("Sample Tweets")
selected_sample = st.sidebar.selectbox("Choose a sample tweet:", [""] + sample_tweets)

# Main input
tweet_input = st.text_input("Enter a tweet for sentiment analysis:")

# Use sample if no manual input
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
**Accuracy**: 79.5%  
**Deployment**: Streamlit Cloud  
**Stack**: Python, scikit-learn, pandas, TF-IDF  
**Purpose**: Recruiter-ready NLP showcase for AIML career advancement
""")
