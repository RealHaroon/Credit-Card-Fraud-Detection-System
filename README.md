
# Credit Card Fraud Detection System

A machine learning system for detecting fraudulent credit card transactions on a severely imbalanced dataset (0.17% fraud rate), deployed as a FastAPI service with both a web UI and terminal client for real-time predictions.

## Overview

This project benchmarks bagging vs. boosting ensemble methods on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 492 fraud cases), selects the empirically best-performing model, and deploys it behind a REST API.

## Project Structure

```
Credit-Card-Fraud-Detection-System/
├── README.md
├── main.py                 # FastAPI app (API + web UI)
├── client_terminal.py       # Terminal script to send predictions to the API
├── sample_fraud.json        # Real fraud transaction (ground truth: Class=1)
├── sample_legit.json        # Real legitimate transaction (ground truth: Class=0)
├── model/
│   ├── fraud_model.joblib   # Trained RandomForest model
│   └── model_meta.joblib    # Threshold + feature name metadata
└── notebook/
    └── notebook.ipynb       # Full experimentation: EDA, model comparison, tuning
```

## Dataset

- 284,807 transactions, 492 labeled fraud (0.1727% positive class)
- Features `V1`–`V28`: PCA-transformed components (anonymized for privacy)
- `Time`, `Amount`: raw, untransformed
- **Severe class imbalance** — accuracy is a misleading metric here (predicting "not fraud" for every transaction already scores ~99.83% accuracy). Evaluation instead centers on **precision, recall, F1, ROC-AUC, and PR-AUC** (PR-AUC specifically, since it's far more informative than ROC-AUC under extreme imbalance).

## Experiment: Bagging vs. Boosting

The notebook documents a full comparison between **RandomForest** (bagging) and **GradientBoosting** (boosting), since boosting is often assumed to be the default "industry standard" for fraud detection. The experiment was run in stages rather than assuming that going in:

1. **Baseline (untuned) comparison** — RandomForest immediately outperformed default `GradientBoostingClassifier` by a wide margin (PR-AUC 0.82 vs 0.44). Investigation also surfaced a practical bottleneck: sklearn's classic `GradientBoostingClassifier` is exact-greedy and impractically slow at this data scale (280K rows) — switched to `HistGradientBoostingClassifier` (histogram-binned) for all subsequent boosting runs.

2. **Hyperparameter tuning** — Ran a cross-validated randomized search (`RandomizedSearchCV`, 25 candidates × 3-fold CV, scored on PR-AUC) over `learning_rate`, `max_leaf_nodes`, `max_depth`, `max_iter`, `l2_regularization`, and `min_samples_leaf`. This closed most of the gap (PR-AUC 0.44 → 0.79).

3. **Decision threshold tuning** — Swept the precision-recall curve to find the F1-optimal threshold for both models, instead of using the default 0.5 cutoff. This mattered more than the algorithm choice itself for GradientBoost: same PR-AUC (0.7905), but F1 rose from 0.598 (threshold=0.5) to 0.812 (threshold=0.963) — proof that PR-AUC measures ranking quality while the threshold is a separate decision that determines how that ranking becomes an actual classification.

### Final result

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **RandomForest** (threshold=0.43) | 0.9516 | 0.7973 | **0.8676** | 0.9365 | **0.8235** |
| GradientBoost, tuned (threshold=0.963) | 0.8943 | 0.7432 | 0.8118 | 0.9612 | 0.7905 |

**RandomForest wins on PR-AUC and F1 even after GradientBoost was fully tuned and threshold-optimized.**

### Why RandomForest was chosen over GradientBoost

- **Features are already PCA-transformed** — decorrelated, no raw interaction structure left. Boosting's core advantage (sequentially modeling residual errors to capture complex interactions) has less to exploit here; bagging's advantage (averaging many independent trees to cancel noise) doesn't depend on that structure.
- **RandomForest's defaults were already close to optimal** for this data — bagging is inherently more forgiving of hyperparameters than boosting, which needs shrinkage/depth/regularization tuned carefully to perform well.
- **Clean, low-dimensional, moderate-size tabular data** (284K rows, 30 features, no missing values, no raw messy interactions) is close to ideal terrain for RandomForest. Boosting's real production edge tends to show up on larger, messier, higher-dimensional data with genuine raw non-linear interactions, usually paired with much heavier tuning infrastructure than used here.
- **The takeaway**: "boosting is the industry standard" is true conditionally — for tuned models on the right kind of data — not universally. Model selection should follow the evidence from rigorous evaluation, not a fixed assumption about which algorithm family is "supposed" to win.

## Model

- **Algorithm**: RandomForestClassifier (`n_estimators=200`)
- **Decision threshold**: 0.43 (F1-optimized via precision-recall curve sweep, not the default 0.5)
- Trained model and metadata (threshold, feature names) are saved in `model/` via `joblib`.

## Running the API

```bash
pip install fastapi uvicorn joblib scikit-learn pandas requests
uvicorn main:app --reload
```

- **Web UI**: `http://127.0.0.1:8000/` — paste a transaction as JSON (all 30 features) into the textarea and get an instant fraud probability + flagged decision.
- **Interactive API docs**: `http://127.0.0.1:8000/docs` (auto-generated by FastAPI)
- **Predict endpoint**: `POST /predict` — accepts a JSON body with all feature values, returns `fraud_probability`, `is_fraud`, and `threshold_used`.

## Predicting from the terminal

```bash
python3 client_terminal.py
```

Sends `sample_fraud.json` and `sample_legit.json` to the running API and prints the model's prediction for each — useful as a quick sanity check that the deployed model correctly flags a known fraud case and clears a known legitimate one.

Or with curl directly:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_fraud.json
```

## Tech Stack

- **Modeling**: scikit-learn (RandomForestClassifier, HistGradientBoostingClassifier)
- **Evaluation**: precision, recall, F1, ROC-AUC, PR-AUC, precision-recall threshold tuning
- **Tuning**: RandomizedSearchCV with StratifiedKFold cross-validation
- **Deployment**: FastAPI, Pydantic, joblib
- **Full experimentation notebook**: `notebook/notebook.ipynb`
