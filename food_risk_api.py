from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load models
model = joblib.load("food_risk_predictor_rf.pkl")
scaler = joblib.load("food_scaler.pkl")
ingredient_encoder = joblib.load("ingredient_encoder.pkl")
category_encoder = joblib.load("category_encoder.pkl")

class RiskRequest(BaseModel):
    ingredients: list
    category: str
    sugar_g: float
    fat_g: float
    salt_g: float
    fiber_g: float
    protein_g: float
    calories: float

@app.post("/predict-risk")
def predict_risk(data: RiskRequest):
    try:
        # One-hot encode ingredients
        known_ingredients = list(ingredient_encoder.classes_)
        ingredient_vector = [1 if ing in data.ingredients else 0 for ing in known_ingredients]

        # Encode category
        encoded_category = category_encoder.transform([data.category]).tolist()

        # Add nutrition info
        nutrition = [
            data.sugar_g, data.fat_g, data.salt_g,
            data.fiber_g, data.protein_g, data.calories
        ]

        # Merge all features
        features = np.array(encoded_category + ingredient_vector + nutrition).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Predict
        score = model.predict_proba(features_scaled)[0][1]
        risk_level = "Low" if score < 0.33 else "Moderate" if score < 0.66 else "High"
        color = "Green" if risk_level == "Low" else "Orange" if risk_level == "Moderate" else "Red"

        return {
            "risk_score": round(float(score), 2),
            "risk_level": risk_level,
            "color": color
        }

    except Exception as e:
        return {"error": str(e)}
