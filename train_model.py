import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load data
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

fake['label'] = 0
real['label'] = 1
news = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)

X = news['text']
y = news['label']

# Vectorize
vectorizer = CountVectorizer(stop_words='english', max_df=0.7)
X_vectors = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vectors, y)

# Save model and vectorizer
joblib.dump(model, "model/fake_news_model.joblib")
joblib.dump(vectorizer, "model/vectorizer.joblib")

print("✅ Model and vectorizer saved successfully!")
