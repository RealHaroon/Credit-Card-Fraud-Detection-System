# api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, create_model
import joblib
import numpy as np

app = FastAPI(title="Fraud Detection API")

model = joblib.load("/run/media/haroon/Local Disk (D:)/Github/Credit-Card-Fraud-Detection-System/model/fraud_model.joblib")
meta = joblib.load("/run/media/haroon/Local Disk (D:)/Github/Credit-Card-Fraud-Detection-System/model/model_meta.joblib")
THRESHOLD = meta["threshold"]
FEATURE_NAMES = meta["feature_names"]

fields = {name: (float, ...) for name in FEATURE_NAMES}
Transaction = create_model("Transaction", **fields)

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold_used: float

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    row = [getattr(transaction, name) for name in FEATURE_NAMES]
    X = np.array(row).reshape(1, -1)
    prob = float(model.predict_proba(X)[0, 1])
    return PredictionResponse(
        fraud_probability=round(prob, 6),
        is_fraud=prob >= THRESHOLD,
        threshold_used=THRESHOLD,
    )

@app.get("/", response_class=HTMLResponse)
def home():
    # a zeroed-out example so the textarea is never empty / user has a valid shape to edit
    sample_json = "{\n" + ",\n".join(f'  "{name}": 0.0' for name in FEATURE_NAMES) + "\n}"

    return f"""
    <html>
    <head>
        <title>Fraud Detector</title>
        <style>
            body {{ font-family: sans-serif; max-width: 640px; margin: 40px auto; }}
            textarea {{ width: 100%; height: 320px; font-family: monospace; font-size: 13px; }}
            button {{ margin-top: 10px; padding: 8px 16px; cursor: pointer; }}
            #result {{ margin-top: 16px; padding: 12px; border-radius: 6px; }}
            .fraud {{ background: #fdd; color: #900; }}
            .clean {{ background: #dfd; color: #060; }}
        </style>
    </head>
    <body>
        <h2>Credit Card Fraud Detection</h2>
        <p>Paste a transaction as JSON (all {len(FEATURE_NAMES)} features required):</p>
        <textarea id="jsonInput">{sample_json}</textarea>
        <br>
        <button onclick="predict()">Predict</button>
        <div id="result"></div>

        <script>
            async function predict() {{
                const resultDiv = document.getElementById("result");
                let payload;
                try {{
                    payload = JSON.parse(document.getElementById("jsonInput").value);
                }} catch (e) {{
                    resultDiv.className = "fraud";
                    resultDiv.innerText = "Invalid JSON: " + e.message;
                    return;
                }}

                const res = await fetch("/predict", {{
                    method: "POST",
                    headers: {{"Content-Type": "application/json"}},
                    body: JSON.stringify(payload)
                }});

                if (!res.ok) {{
                    const err = await res.json();
                    resultDiv.className = "fraud";
                    resultDiv.innerText = "Error: " + JSON.stringify(err.detail);
                    return;
                }}

                const json = await res.json();
                resultDiv.className = json.is_fraud ? "fraud" : "clean";
                resultDiv.innerText =
                    `Fraud Probability: ${{json.fraud_probability}} | Flagged as Fraud: ${{json.is_fraud}} | Threshold: ${{json.threshold_used}}`;
            }}
        </script>
    </body>
    </html>
    """