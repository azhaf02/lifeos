import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# Dummy training dataset
data = pd.DataFrame({
    "completed_tasks": [2,5,8,1,7,6,3,9],
    "total_tasks": [5,6,10,4,8,7,5,10],
    "sleep_hours": [6,8,7,5,8,6,7,9],
    "mood_score": [5,8,7,4,9,6,6,9],
    "burnout": [1,0,0,1,0,1,0,0]  # 1=High Risk
})

# Productivity Model
X_reg = data[["completed_tasks","total_tasks","sleep_hours","mood_score"]]
y_reg = (data["completed_tasks"]/data["total_tasks"])*100

reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)

joblib.dump(reg_model, "productivity_model.pkl")

# Burnout Model
X_clf = data[["sleep_hours","mood_score","total_tasks"]]
y_clf = data["burnout"]

clf_model = RandomForestClassifier()
clf_model.fit(X_clf, y_clf)

joblib.dump(clf_model, "burnout_model.pkl")

print("Models trained and saved!")
