from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

prod_model = joblib.load("productivity_model.pkl")
burn_model = joblib.load("burnout_model.pkl")

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
    
    return jsonify({"productivity": round(float(prediction),2)})

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
    
    return jsonify({"burnout_risk": risk})

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/")
def home():
    return "LifeOS AI API Running"
