
import os
from huggingface_hub import HfApi, login

login(token=os.environ["HF_TOKEN"])
api = HfApi()
HF_USERNAME = os.environ["HF_USERNAME"]
SPACE_REPO  = f"{HF_USERNAME}/visit-with-us-app"

# Create Space
try:
    api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="streamlit", private=False)
    print(f"Space created: https://huggingface.co/spaces/{SPACE_REPO}")
except Exception as e:
    print(f"Space already exists: {e}")

# Push deployment files
for fname in ["app.py", "requirements.txt", "Dockerfile"]:
    api.upload_file(
        path_or_fileobj=f"deployment/{fname}",
        path_in_repo=fname,
        repo_id=SPACE_REPO,
        repo_type="space"
    )
    print(f"Pushed {fname} , https://huggingface.co/spaces/{SPACE_REPO}")

print(f"Streamlit App Live at: https://huggingface.co/spaces/{SPACE_REPO}")

