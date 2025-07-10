# 🧠 Sentiment Analysis of Product Reviews

This project is a **machine learning web app** that analyzes the sentiment of product reviews using a **Naive Bayes Classifier** and **CountVectorizer**. The app is built with **Streamlit** and provides a clean and interactive UI for sentiment classification.

---

## 🔗 Live App

👉 **Try it here**: [Sentiment Analysis App on Streamlit](https://yashisaraswat86-pbel-project-app-cvox4b.streamlit.app/)

---

## 🔍 Features

- Predict sentiment (Positive, Negative, Neutral) of user-written product reviews
- Built-in text cleaning and negation handling
- Uses `CountVectorizer` for word frequency-based features
- Trained with `MultinomialNB` for fast and effective classification
- Pie chart showing sentiment distribution in sidebar
- Deployed via **Streamlit Cloud**

---

## 🛠️ Technologies Used

- **Python 3**
- **Streamlit**
- **Pandas**, **Scikit-learn**
- **NLTK** for text preprocessing
- **Matplotlib**, **Seaborn**
- **MultinomialNB** from `sklearn.naive_bayes`
- **CountVectorizer** for text vectorization

---

## 🗂️ Project Structure

```
📁 sentiment-analysis-app/
├── app.py # Streamlit UI
├── model.py # Model training and evaluation
├── product_reviews_clean_3000.csv # Dataset
├── sentiment_model.pkl # Trained Naive Bayes model
├── vectorizer.pkl # Count vectorizer
├── sentiment_distribution.png # Sentiment pie chart
├── requirements.txt # List of dependencies
└── README.md # Project documentation
```

---

## 🚀 Deployment

Deployed using [Streamlit Cloud](https://streamlit.io/cloud)

To run locally:

```bash
git clone https://github.com/yashi/sentiment-analysis-app.git
cd sentiment-analysis-app
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Model Accuracy

- Vectorization: CountVectorizer
- Classifier: CountVectorizer
- Accuracy: ~85% on test set
- Evaluation: Classification report and confusion matrix shown in terminal

---

## 🙋‍♀️ Author

**Yashi Saraswat**  
20-year-old aspiring ML developer  
Project created for academic and portfolio purposes

---

## 📬 Feedback

Feel free to fork, contribute, or submit issues.  
This is a beginner-friendly open-source project for learning sentiment analysis and deployment.
