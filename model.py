import pandas as pd
import string
import re
import joblib
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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

df = pd.read_csv("product_reviews_clean_3000.csv")
df['cleaned_review'] = df['review'].apply(clean_text)

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='label', palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig("sentiment_distribution.png")
plt.close()

cv = CountVectorizer()
X = cv.fit_transform(df['cleaned_review']).toarray()
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


joblib.dump(model, "sentiment_model.pkl")
joblib.dump(cv, "vectorizer.pkl")
print("Model and vectorizer saved.")







