
import os
from huggingface_hub import HfApi, login

login(token=os.environ["HF_TOKEN"])
api = HfApi()
HF_USERNAME  = os.environ["HF_USERNAME"]
DATASET_REPO = f"{HF_USERNAME}/visit-with-us-data"

# Create dataset repo 
try:
    api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=False)
    print(f"Dataset repo created: https://huggingface.co/datasets/{DATASET_REPO}")
except Exception as e:
    print(f"Repo already exists or error: {e}")

# Create local data folder and upload raw CSV
os.makedirs("./data", exist_ok=True)
api.upload_file(
    path_or_fileobj="tourism.csv",
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset"
)
print(f"Raw dataset registered at: https://huggingface.co/datasets/{DATASET_REPO}")

