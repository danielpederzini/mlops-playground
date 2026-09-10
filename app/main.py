import os
from pydantic import BaseModel, ConfigDict, Field
import mlflow
import mlflow.sklearn
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI

app = FastAPI(
    title="Fetal Health Classification API",
    description="This API provides endpoints for classifying fetal health based on input features.",
    version="0.0.1",
    openapi_tags=[
        {
            "name": "Application Health",
            "description": "Endpoints for checking the health of the API."
        },
        {
            "name": "Inference",
            "description": "Endpoints for classifying fetal health based on input features."
        }
    ]
)

def load_env():
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        print(f"Loaded environment variables from {dotenv_path}")
    else:
        print(f"Warning: .env file not found at {dotenv_path}. Using default environment variables.")


def load_model():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "fetal_health_classifier")
    model_version = os.getenv("MLFLOW_MODEL_VERSION")

    print(f"Loading model from MLflow tracking URI: {tracking_uri}, model name: {model_name}")
    mlflow.set_tracking_uri(tracking_uri)

    if not model_version:
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        versions = client.search_model_versions(
            f"name='{model_name}'", max_results=1,
            order_by=["version_number DESC"])
        if not versions:
            raise RuntimeError(
                f"No versions registered for model {model_name!r}")
        model_version = versions[0].version

    logged_model_uri = f"models:/{model_name}/{model_version}"
    logged_model = mlflow.sklearn.load_model(logged_model_uri)

    print(f"Model loaded successfully from {logged_model_uri}")

    return logged_model


@app.on_event("startup")
def on_startup():
    print("Starting up the Fetal Health Classification API...")
    load_env()
    
    global model
    model = load_model()
    

@app.get(
    path="/health",
    tags=["Application Health"],
    summary="Check the health of the API",
    description="This endpoint returns the health status of the API."
)
async def health():
    return {"status": "HEALTHY"}

class FetalHealthInferenceRequest(BaseModel):
    baseline_value: float = Field(alias="baseline value")
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    light_decelerations: float
    severe_decelerations: float
    prolongued_decelerations: float
    abnormal_short_term_variability: float
    mean_value_of_short_term_variability: float
    percentage_of_time_with_abnormal_long_term_variability: float
    mean_value_of_long_term_variability: float
    histogram_width: float
    histogram_min: float
    histogram_max: float
    histogram_number_of_peaks: float
    histogram_number_of_zeroes: float
    histogram_mode: float
    histogram_mean: float
    histogram_median: float
    histogram_variance: float
    histogram_tendency: float

    model_config = ConfigDict(populate_by_name=True)


FEATURE_ORDER = list(FetalHealthInferenceRequest.model_fields)

CLASS_MAP = {
    0: "Normal",
    1: "Suspect",
    2: "Pathological"
}

@app.post(
    path="/predictions",
    tags=["Inference"],
    summary="Predict fetal health",
    description="This endpoint predicts the health of a fetus based on input features."
)
async def predict(input_data: FetalHealthInferenceRequest):
    features = input_data.model_dump()
    input_array = np.array([[features[name] for name in FEATURE_ORDER]],
                           dtype="float32")

    raw_prediction = model.decision_function(input_array)
    predicted_class = int(model.predict(input_array)[0])

    prediction_result = {
        "predicted_class": CLASS_MAP.get(predicted_class, "Unknown"),
        "raw_prediction": raw_prediction.tolist()
    }
    
    print(f"Prediction made for input {input_data}: {prediction_result['predicted_class']}")
    return prediction_result