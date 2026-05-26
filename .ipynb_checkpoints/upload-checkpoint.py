from pathlib import Path
from huggingface_hub import HfApi
import os

# ============================================================
# CONFIG
# ============================================================

HF_DATASET_REPO = "ShreyashDhoot/Guardpaint"
HF_TOKEN = os.getenv("HF_TOKEN")

# List folders you want to upload
FOLDERS_TO_UPLOAD = [
    "tournaments",
    "jailbreak_results.json"
]

# ============================================================
# INIT
# ============================================================

api = HfApi(token=HF_TOKEN)

root_dir = Path.cwd()

files_to_upload = []

# Add all files from selected folders recursively
for folder in FOLDERS_TO_UPLOAD:

    folder_path = root_dir / folder

    if not folder_path.exists():
        print(f"Folder not found: {folder}")
        continue

    for file_path in folder_path.rglob("*"):
        if file_path.is_file():
            files_to_upload.append(file_path)

print(f"Found {len(files_to_upload)} files.\n")

# ============================================================
# UPLOAD
# ============================================================

for file_path in files_to_upload:

    relative_path = file_path.relative_to(root_dir)

    print(f"Uploading: {relative_path}")

    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=str(relative_path).replace("\\", "/"),
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )

print("\nDone.")