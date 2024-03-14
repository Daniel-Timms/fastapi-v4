from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.model.model import predict_pipeline
from app.model.model import __date__ as model_version

app = FastAPI()

# Define a request payload that expects an array of integers or floats
class ArrayInput(BaseModel):
    data: List[float]  # or List[int] if your input is integers

# Define the output format of your prediction endpoint
class PredictionOut(BaseModel):
    prediction_result: float  # or int/str, depending on what your model returns

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}

# Update the predict endpoint to accept an array and return a prediction
@app.post("/predict", response_model=PredictionOut)
def predict(payload: ArrayInput):
    prediction = predict_pipeline(payload.data)  # Assuming your model's predict_pipeline function accepts a list
    return {"prediction result:": prediction}  # <- change here, in the text

# To run, in the terminal, navigate to the directory containing main.py and run: uvicorn app.main:app --reload