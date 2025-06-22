import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# Load fake and real news
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

# Add label column: 0 for fake, 1 for real
fake['label'] = 0
real['label'] = 1

# Combine datasets
news = pd.concat([fake, real])

# Shuffle rows
news = news.sample(frac=1).reset_index(drop=True)

# Show basic info
print("Total entries:", len(news))
print("Columns:", news.columns)
print(news.head())
