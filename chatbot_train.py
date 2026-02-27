import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

data = pd.read_csv("chatbot_dataset.csv")

X = data["user_input"]
y = data["intent"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

joblib.dump(model,"chatbot_model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")

print("Chatbot trained successfully")