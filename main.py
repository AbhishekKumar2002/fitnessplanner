import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import os
from pathlib import Path

from src.utils.exercise_generator import generate_exercise_plan
from src.utils.meal_planner import generate_meal_plan

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "src" / "data" / "models"

# Pydantic input model
class UserInput(BaseModel):
    age: int
    weight: float
    height: float
    gender: str
    activity_level: str
    goal: str

# Load models
def load_models(model_dir=MODEL_DIR):
    models = {}
    for target in ['target_calories', 'protein_ratio', 'carb_ratio', 'fat_ratio', 'exercise_intensity']:
        model_path = model_dir / f"{target}_model.pkl"
        if not model_path.exists():
            logging.error(f"Model file for {target} not found at {model_path}")
            raise HTTPException(status_code=500, detail=f"Model file for {target} not found")
        with open(model_path, 'rb') as f:
            models[target] = pickle.load(f)

    preprocessing_path = model_dir / "preprocessing.pkl"
    if not preprocessing_path.exists():
        logging.error(f"Preprocessing pipeline not found at {preprocessing_path}")
        raise HTTPException(status_code=500, detail="Preprocessing pipeline not found")

    with open(preprocessing_path, 'rb') as f:
        preprocessing = pickle.load(f)

    return models, preprocessing

# Health check
@app.get("/")
def root():
    return {"message": "Fitness Plan API is live. Use POST /api/fitness-plan."}

# Prediction endpoint
@app.post("/api/fitness-plan")
async def get_fitness_plan(user_input: UserInput):
    logging.info("Received POST request at /api/fitness-plan")
    try:
        # Load models
        models, preprocessing = load_models()

        # Prepare input
        input_df = pd.DataFrame([user_input.dict()])
        logging.debug(f"Input DataFrame:\n{input_df}")

        # Transform
        X_transformed = preprocessing.transform(input_df)
        numeric_features = ['age', 'weight', 'height']
        cat_features = ['gender', 'activity_level', 'goal']
        cat_feature_names = preprocessing.transformers_[1][1].named_steps['encoder'].get_feature_names_out(cat_features)
        all_feature_names = numeric_features + list(cat_feature_names)
        X_input_df = pd.DataFrame(X_transformed, columns=all_feature_names)
        logging.debug(f"Transformed DataFrame:\n{X_input_df}")

        # Predict
        predictions = {}
        for target, model in models.items():
            predictions[target] = float(model.predict(X_input_df)[0])

        # Generate plans
        exercise_plan = generate_exercise_plan(predictions['exercise_intensity'])
        meal_plan = generate_meal_plan(
            predictions['target_calories'],
            predictions['protein_ratio'],
            predictions['carb_ratio'],
            predictions['fat_ratio']
        )

        return {
            "predictions": {
                "target_calories": round(predictions['target_calories']),
                "protein_ratio": round(predictions['protein_ratio'] * 100, 1),
                "carb_ratio": round(predictions['carb_ratio'] * 100, 1),
                "fat_ratio": round(predictions['fat_ratio'] * 100, 1),
                "exercise_intensity": round(predictions['exercise_intensity'], 1)
            },
            "exercise_plan": exercise_plan,
            "meal_plan": meal_plan
        }

    except Exception as e:
        logging.error(f"Unhandled error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
