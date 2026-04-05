

import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
from datasets import load_dataset
from huggingface_hub import HfApi, login

# config
HF_TOKEN     = os.environ["HF_TOKEN"]
HF_USERNAME  = os.environ["HF_USERNAME"]
DATASET_REPO = f"{HF_USERNAME}/visit-with-us-data"
MODEL_REPO   = f"{HF_USERNAME}/visit-with-us-model"
TARGET_COL   = "ProdTaken"
RANDOM_STATE = 42

# Load train and test data
print("Logging in to Hugging Face...")
login(token=HF_TOKEN)
api = HfApi()

print("Loading train split from Hugging Face...")
train_ds = load_dataset(DATASET_REPO, data_files="train.csv", split="train")
train_df = train_ds.to_pandas()

print("Loading test split from Hugging Face...")
test_ds  = load_dataset(DATASET_REPO, data_files="test.csv",  split="train")
test_df  = test_ds.to_pandas()

print(f"Train shape : {train_df.shape}")
print(f"Test  shape : {test_df.shape}")

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]

X_test  = test_df.drop(columns=[TARGET_COL])
y_test  = test_df[TARGET_COL]

# define model
xgb_model = XGBClassifier(
    eval_metric="logloss",
    random_state=RANDOM_STATE
)

param_grid = {
    "n_estimators"    : [100, 200],
    "max_depth"       : [3, 5, 7],
    "learning_rate"   : [0.05, 0.1, 0.2],
    "subsample"       : [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

gs = GridSearchCV(
    xgb_model,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

# tune model
print("\nTuning XGBoost with GridSearchCV...")
gs.fit(X_train, y_train)

best_model  = gs.best_estimator_
best_params = gs.best_params_
print(f"\nBest Parameters : {best_params}")

# evaluate
y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print(f"\nXGBoost Results")
print(f"  Accuracy : {acc:.4f}")
print(f"  F1 Score : {f1:.4f}")
print(f"  ROC-AUC  : {roc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Purchased", "Purchased"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# log params in mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("visit-with-us-xgboost")

with mlflow.start_run(run_name="XGBoost_GridSearchCV"):
    # Log best hyperparameters
    for param_name, param_value in best_params.items():
        mlflow.log_param(param_name, param_value)

    # Log evaluation metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc",  roc)

    # Log the best model artifact with MLflow
    mlflow.sklearn.log_model(best_model, artifact_path="xgboost_model")

    print(f"\nMLflow run logged: Accuracy: {acc:.4f}  F1: {f1:.4f}  ROC-AUC: {roc:.4f}")

# Save artifacts locally and register on Hugging Face
os.makedirs("model_artifacts", exist_ok=True)

# Save pickled model
with open("model_artifacts/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save feature names so the Streamlit app can reconstruct the input DataFrame
with open("model_artifacts/feature_names.json", "w") as f:
    json.dump(list(X_train.columns), f)

print("\nXGBoost model and feature names saved locally.")

# Create model repo on HF (skip if already exists)
try:
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)
    print(f"Model repo created: https://huggingface.co/models/{MODEL_REPO}")
except Exception as e:
    print(f"Repo already exists or error: {e}")

# Upload both artifacts to the HF model hub
for fname in ["best_model.pkl", "feature_names.json"]:
    api.upload_file(
        path_or_fileobj=f"model_artifacts/{fname}",
        path_in_repo=fname,
        repo_id=MODEL_REPO,
        repo_type="model"
    )
    print(f"Registered {fname}  ,  https://huggingface.co/models/{MODEL_REPO}")

print("\nModel training and registration complete.")
