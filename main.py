from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load saved model and vectorizer
model = joblib.load("model/fake_news_model.joblib")
vectorizer = joblib.load("model/vectorizer.joblib")

# Init FastAPI
app = FastAPI()

# Input format
class NewsInput(BaseModel):
    text: str

# Prediction route
@app.post("/predict")
def predict(news: NewsInput):
    vector = vectorizer.transform([news.text])
    prediction = model.predict(vector)[0]
    result = "Real News 🟢" if prediction == 1 else "Fake News 🔴"
    return {"prediction": result}
