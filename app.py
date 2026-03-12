from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = pickle.load(open("wine_model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Wine Quality Prediction API"}

@app.post("/predict")
def predict(
    fixed_acidity: float,
    volatile_acidity: float,
    citric_acid: float,
    residual_sugar: float,
    chlorides: float,
    free_sulfur_dioxide: float,
    total_sulfur_dioxide: float,
    density: float,
    pH: float,
    sulphates: float,
    alcohol: float
):

    features = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                          residual_sugar, chlorides,
                          free_sulfur_dioxide, total_sulfur_dioxide,
                          density, pH, sulphates, alcohol]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Good Quality Wine"
    else:
        result = "Bad Quality Wine"

    return {"prediction": result}