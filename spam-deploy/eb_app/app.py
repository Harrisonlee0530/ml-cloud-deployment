import json
import os
import random

import boto3
import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
AWS_REGION = os.environ.get("AWS_REGION", "ca-central-1")

# Simple honeypot field name
HONEYPOT_FIELD = "website"


def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=AWS_REGION)


def predict_message(text: str):
    runtime = get_runtime_client()
    payload = json.dumps({"instances": [text]})

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload,
    )

    result = json.loads(response["Body"].read().decode("utf-8"))
    label = result["predicted_labels"][0]
    return label, result


def get_random_ham_message():
    # Free no-key API
    # https://api.adviceslip.com/advice
    # Returns human-readable advice text
    try:
        r = requests.get(
            "https://api.adviceslip.com/advice",
            timeout=5,
            headers={"Cache-Control": "no-cache"}
        )
        r.raise_for_status()
        data = r.json()
        advice = data.get("slip", {}).get("advice")
        if advice:
            return advice
    except Exception:
        pass

    fallback = [
        "Remember to call your parents this evening.",
        "Take a short break and stretch your legs.",
        "Please submit your assignment before midnight.",
        "Let's meet at the library after class."
    ]
    return random.choice(fallback)


def get_random_spam_message():
    # Free no-key API
    # https://fakestoreapi.com/products
    # We turn a product into an obviously promotional spam-like message
    # It keeps going down, so maybe try sth different!
    try:
        r = requests.get("https://fakestoreapi.com/products", timeout=5)
        r.raise_for_status()
        products = r.json()
        if isinstance(products, list) and products:
            product = random.choice(products)
            title = product.get("title", "Exclusive product")
            price = product.get("price", "9.99")
            return (
                f"LIMITED TIME OFFER! Get {title} today for just ${price}! "
                f"Click now to claim your exclusive deal and win bonus rewards!"
            )
    except Exception:
        pass

    fallback = [
        "URGENT! You have been selected for a cash reward. Click now to claim it.",
        "Congratulations! Exclusive offer just for you. Reply now to win prizes.",
        "Final notice: your account qualifies for a premium gift. Act immediately.",
        "Special promotion! Buy now and unlock your free bonus reward."
    ]
    return random.choice(fallback)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Honeypot: bots often fill hidden fields
    if request.form.get(HONEYPOT_FIELD):
        return render_template(
            "index.html",
            error="Bot-like submission detected. Please try again."
        )

    message = request.form.get("message", "").strip()

    if not message:
        return render_template(
            "index.html",
            error="Please enter a message."
        )

    try:
        label, raw_result = predict_message(message)
        return render_template(
            "index.html",
            message=message,
            prediction=label,
            raw_result=raw_result
        )
    except Exception as e:
        return render_template(
            "index.html",
            message=message,
            error=str(e)
        )


@app.route("/random-message", methods=["GET"])
def random_message():
    # 50/50 spam vs ham
    if random.random() < 0.5:
        label = "ham"
        message = get_random_ham_message()
    else:
        label = "spam"
        message = get_random_spam_message()

    return jsonify({
        "message": message,
        "source_label": label
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    # JSON API for curl/Postman/etc.
    try:
        payload = request.get_json(force=True)
        message = payload.get("message", "").strip()

        if not message:
            return jsonify({"error": "Missing 'message'"}), 400

        label, raw_result = predict_message(message)

        return jsonify({
            "message": message,
            "prediction": label,
            "raw_result": raw_result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
