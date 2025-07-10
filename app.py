import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    negation_patterns = {
        "not good": "bad", "not happy": "sad", "not bad": "good",
        "not great": "poor", "not recommend": "avoid", "not working": "broken",
        "not worth": "waste", "not useful": "useless", "not nice": "awful"
    }
    for phrase, replacement in negation_patterns.items():
        text = text.replace(phrase, replacement)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Load model, vectorizer, and dataset
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
df = pd.read_csv("product_reviews_clean_3000.csv")
sentiment_counts = df["label"].value_counts()

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

#  CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(to bottom right, #00bfa5, #80deea);
        color: #003d33;
    }
    .main {
        padding: 2rem;
        background-color: #ffffff10;
        border-radius: 12px;
    }
    .stTextArea textarea {
        border-radius: 10px;
        font-size: 16px;
        background-color: #e0f7fa;
        color: #004d40;
    }
    .stButton > button {
        background-color: #00796b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with pie chart
with st.sidebar:
    st.markdown("## 📊 Sentiment Stats")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
           startangle=90, colors=['#004d40', '#26c6da', '#b2ebf2'])
    ax.axis('equal')
    st.pyplot(fig)
    st.markdown("---")
    st.markdown("✅ **Model**: Naive Bayes")
    st.markdown("📦 **Dataset**: 3000 product reviews")


# Main content
st.title("Sentiment Analyzer")
st.subheader("🔍 Check the sentiment of a product review")
st.markdown("Enter your review below to find out whether it's **Positive**, **Negative**, or **Neutral**.")

user_input = st.text_area("💬 Your product review:")

if st.button("🔎 Analyze Review"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        cleaned = clean_text(user_input)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0].lower()

        if prediction in ["positive", "good", "great", "happy", "nice"]:
            st.success("✔️ **Sentiment: POSITIVE** 😊")
        elif prediction in ["negative", "bad", "worst", "poor", "awful"]:
            st.error("❌ **Sentiment: NEGATIVE** 😞")
        else:
            st.info("😐 **Sentiment: NEUTRAL**")

st.markdown("---")
st.markdown("<center>🌈 Made with ❤️ using Python + Streamlit + ML</center>", unsafe_allow_html=True)









