from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# ==========================
# LOAD ML MODELS
# ==========================

prod_model = joblib.load("productivity_model.pkl")
burn_model = joblib.load("burnout_model.pkl")
health_model = joblib.load("health_model.pkl")   # NEW MODEL


# ==========================
# HOME ROUTE
# ==========================

@app.route("/")
def home():
    return "LifeOS AI API Running"


# ==========================
# PRODUCTIVITY PREDICTION
# ==========================

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
    
    return jsonify({
        "productivity": round(float(prediction),2)
    })


# ==========================
# BURNOUT PREDICTION
# ==========================

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
    
    return jsonify({
        "burnout_risk": risk
    })


# ==========================
# HEALTH RISK PREDICTION
# ==========================

@app.route("/predict-health", methods=["POST"])
def predict_health():
    
    data = request.json
    
    features = np.array([[
        data["pregnancies"],
        data["glucose"],
        data["blood_pressure"],
        data["skin_thickness"],
        data["insulin"],
        data["bmi"],
        data["diabetes_pedigree"],
        data["age"]
    ]])
    
    prediction = health_model.predict(features)[0]

    if prediction == 1:
        result = "High Risk of Diabetes"
        advice = [
            "Reduce sugar intake",
            "Exercise regularly",
            "Maintain healthy body weight",
            "Check blood glucose regularly"
        ]
    else:
        result = "Low Risk"
        advice = [
            "Maintain healthy diet",
            "Continue regular exercise",
            "Monitor health regularly"
        ]

    return jsonify({
        "health_prediction": result,
        "advice": advice
    })


# ==========================
# RUN SERVER
# ==========================

if __name__ == "__main__":
    app.run(debug=True)