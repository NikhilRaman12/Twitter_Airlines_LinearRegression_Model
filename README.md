# ✈️ Twitter Airlines Sentiment Classifier

A machine learning web application that classifies airline-related tweets into **positive**, **neutral**, or **negative** sentiments. This project demonstrates practical NLP techniques, model experimentation, and full-stack deployment using **Python**, **scikit-learn**, and **Streamlit**.

🚀 **Live Demo**  
🔗 [Streamlit App](https://your-streamlit-link-here)

---

## 📋 Project Overview

- **Objective**: Build and deploy a sentiment analysis model for airline-related tweets using supervised machine learning.  
- **Dataset**: [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) – over 14,000 labeled tweets.  
- **Approach**:  
  - Preprocessing with **TF-IDF vectorization**  
  - Model experimentation with both **Linear Regression** and **Logistic Regression**  
  - Final deployment using **Logistic Regression**, as sentiment is a **categorical** output, not continuous.

---

## 🔍 Modeling Journey

| Model              | Purpose                            | Outcome                         |
|-------------------|-------------------------------------|----------------------------------|
| Linear Regression | Initial baseline for sentiment scoring | Accuracy: ~9.6% (poor fit)       |
| Logistic Regression | Final model for classification     | Accuracy: **79.5%** (robust fit) |

✅ Logistic Regression was selected for deployment due to its superior performance on categorical sentiment labels.

---

## ✨ Key Features

- 📝 **Custom Tweet Analysis** – Enter any tweet and receive instant sentiment prediction.  
- ⚡ **Real-Time Classification** – Immediate output for positive, neutral, or negative sentiment.  
- ☁️ **Cloud-Hosted Web App** – Accessible anywhere via Streamlit Cloud.  
- 🔗 **Reproducible Workflow** – Clean, documented code with Git-based version control.

---

## 🛠️ Tech Stack

| Category         | Tools Used                          |
|------------------|-------------------------------------|
| Programming      | Python                              |
| ML Libraries     | scikit-learn, pandas, joblib        |
| Deployment       | Streamlit Cloud                     |
| Version Control  | Git & GitHub                        |

---

## 📂 Repository Structure

```
Twitter_Airlines_LinearRegression_Model/
├── app.py                 # Streamlit web application  
├── requirements.txt       # Dependencies  
├── model.pkl              # Trained Logistic Regression model  
├── vectorizer.pkl         # TF-IDF vectorizer  
├── README.md              # Documentation  
└── data/                  # (Optional) Dataset or samples  
```

---

## ⚙️ Running the Project Locally

```bash
# 1️⃣ Clone the repository
git clone https://github.com/NikhilRaman12/Twitter_Airlines_LinearRegression_Model.git
cd Twitter_Airlines_LinearRegression_Model

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the Streamlit app
streamlit run app.py
```

---

## 👤 Author

**Nikhil Raman** – Data Analyst | AIML Engineer in Progress  
📧 Email: nikhilraman1203@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/nikhilraman)  
💻 [GitHub](https://github.com/NikhilRaman12)  
📊 [Tableau](https://public.tableau.com/profile/nikhil.raman)  
📈 [Kaggle](https://kaggle.com/nikhilramank)

---

## 🤝 Contributing

Contributions, suggestions, and feature requests are welcome.  
Fork this repository, open an issue, or submit a pull request to collaborate.

---

##  Why This Project Matters

This project showcases:
- End-to-end ML workflow from data preprocessing to deployment  
- Model selection based on problem type (regression vs classification)  
- Real-time NLP application using open-source tools  
- Resilience through debugging, version control, and iterative improvement

> 🔥 Built with precision, deployed with purpose—this is more than a demo. It’s a recruiter-ready showcase of AIML engineering.
