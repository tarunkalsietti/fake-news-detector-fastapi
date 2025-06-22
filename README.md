# 📰 Fake News Detection API using FastAPI

A machine learning project to detect fake news articles using Natural Language Processing and FastAPI.

##  Features

- Detects fake vs. real news articles
- Built using FastAPI and Naive Bayes classifier
- Trained on 44,000+ real-world news samples
- Real-time prediction via REST API (localhost)
- Interactive Swagger UI support

##  Tech Stack

- Python, pandas, scikit-learn
- FastAPI, Uvicorn
- Joblib (for model saving)
- Dataset: Kaggle (Fake & True news)

<pre> ## 📂 File Structure ``` ├── data/ │ ├── Fake.csv │ └── True.csv ├── model/ │ ├── fake_news_model.joblib │ └── vectorizer.joblib ├── main.py ├── train_model.py ├── startw.py ``` </pre>


## 🔧 Run Locally

```bash
python train_model.py
uvicorn main:app --reload

## 👤 Author

**Tarun Kalisetti**  
[LinkedIn](https://www.linkedin.com/in/tarunkalisetti) • [GitHub](https://github.com/tarunkalsietti)
