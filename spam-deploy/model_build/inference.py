import json
import os
import joblib
import numpy as np


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {
        "model": model,
        "metadata": metadata
    }


def input_fn(request_body, request_content_type):
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    payload = json.loads(request_body)

    instances = payload.get("instances")
    if instances is None:
        raise ValueError("JSON must contain an 'instances' key.")

    if isinstance(instances, str):
        instances = [instances]

    if isinstance(instances, list):
        if len(instances) == 0:
            raise ValueError("'instances' cannot be empty.")
        if all(isinstance(x, str) for x in instances):
            return instances

    raise ValueError(
        "Expected 'instances' to be either a single string or a list of strings."
    )


def predict_fn(input_data, model_bundle):
    model = model_bundle["model"]

    predicted_labels = model.predict(input_data).tolist()

    result = {
        "predicted_labels": predicted_labels
    }

    # Optional: include confidence-like info if available
    if hasattr(model, "predict_proba"):
        try:
            result["probabilities"] = model.predict_proba(input_data).tolist()
        except Exception:
            pass

    return result


def output_fn(prediction, accept):
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction)