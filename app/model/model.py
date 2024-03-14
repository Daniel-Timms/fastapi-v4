import numpy as np
import pickle
from pathlib import Path

__date__ = "140324"

BASE_DIR = Path(__file__).resolve(strict=True).parent

# Load the saved model
with open(f"{BASE_DIR}/finalized_model_{__date__}.pkl", "rb") as f:
    model = pickle.load(f)

# Load the saved scaler
with open(f"{BASE_DIR}/scaler_{__date__}.pkl", "rb") as f:
    scaler = pickle.load(f)


# Define a function to make predictions using the loaded model
def predict_pipeline(features: list):
    # Convert the features into a numpy array and reshape it
    features_array = np.array(features).reshape(1, -1)
    
    # Apply the same scaling transformation as during training
    scaled_features = scaler.transform(features_array)
    
    # Make a prediction using the loaded model
    prediction = model.predict(scaled_features)
    
    # Return the prediction
    return int(prediction[0])