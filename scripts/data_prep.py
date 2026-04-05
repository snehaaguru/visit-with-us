
import os
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from huggingface_hub import HfApi, login

# Config 
HF_TOKEN     = os.environ["HF_TOKEN"]
HF_USERNAME  = os.environ["HF_USERNAME"]
DATASET_REPO = f"{HF_USERNAME}/visit-with-us-data"
TARGET_COL   = "ProdTaken"
RANDOM_STATE = 42

# Login and load dataset from Hugging Face
print("Logging in to Hugging Face...")
login(token=HF_TOKEN)
api = HfApi()

print("Loading dataset from Hugging Face...")
dataset = load_dataset(DATASET_REPO, data_files="tourism.csv", split="train")
df = dataset.to_pandas()
print(f"Loaded data with shape: {df.shape}")

# Data Cleaning

# Drop unnecessary identifier columns
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Check and confirm no missing values remain
missing = df.isna().sum()
print("Missing values per column:")
print(missing[missing > 0] if missing.any() else "  None — all columns are complete.")

# Label-encode all categorical (object) columns
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
print(f"\nEncoded categorical columns: {cat_cols}")

# Train / Test Split and local save
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Re-attach target column
train_df = X_train.copy()
train_df[TARGET_COL] = y_train.values

test_df = X_test.copy()
test_df[TARGET_COL] = y_test.values

print(f"\nTrain shape : {train_df.shape}")
print(f"Test  shape : {test_df.shape}")

# Save splits locally
os.makedirs("./data", exist_ok=True)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv",   index=False)
print("Saved train.csv and test.csv to ./data/")

# Upload splits back to Hugging Face
for fname in ["train.csv", "test.csv"]:
    api.upload_file(
        path_or_fileobj=f"data/{fname}",
        path_in_repo=fname,
        repo_id=DATASET_REPO,
        repo_type="dataset"
    )
    print(f"Uploaded {fname}  ,  https://huggingface.co/datasets/{DATASET_REPO}")

print("Data preparation complete.")
