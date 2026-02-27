from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib


app = Flask(__name__)
CORS(app)
# ---- CHATBOT MODEL ----

chat_model = joblib.load("chatbot_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

responses = {
    "stress": "Take a deep breath and focus on one task at a time.",
    "tired": "You might need rest and hydration.",
    "sleep": "Low sleep affects focus and productivity.",
    "motivation": "Small progress every day leads to success.",
    "workload": "Try prioritizing the most important task first."
}

@app.route("/chat", methods=["POST"])
def chat():

    data = request.json
    message = data["message"]

    msg = vectorizer.transform([message])

    intent = chat_model.predict(msg)[0]

    reply = responses.get(intent,"I'm here to help you stay productive.")

    return jsonify({"reply":reply})


# ---- PRODUCTIVITY MODEL ----

prod_model = joblib.load("productivity_model.pkl")

@app.route("/predict-productivity", methods=["POST"])
def predict_productivity():

    data = request.json

    features = np.array([[
        data["completed_tasks"],
        data["total_tasks"],
        data["sleep_hours"],
        data["mood_score"]
    ]])

    prediction = prod_model.predict(features)[0]

    return jsonify({"productivity":round(float(prediction),2)})


# ---- BURNOUT MODEL ----

burn_model = joblib.load("burnout_model.pkl")

@app.route("/predict-burnout", methods=["POST"])
def predict_burnout():

    data = request.json

    features = np.array([[
        data["sleep_hours"],
        data["mood_score"],
        data["total_tasks"]
    ]])

    prediction = burn_model.predict(features)[0]

    risk = "High" if prediction == 1 else "Low"

    return jsonify({"burnout_risk":risk})


@app.route("/")
def home():
    return "LifeOS AI API Running"


if __name__ == "__main__":
    app.run(debug=True)