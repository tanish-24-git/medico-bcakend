from pathlib import Path
import shutil
import uuid

# ------------------------------
# File Utilities
# ------------------------------

def save_uploaded_file(uploaded_file, upload_dir: str) -> str:
    """
    Save an uploaded file to the specified directory with a unique name.
    Returns the saved file path.
    """
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)

    unique_filename = f"{uuid.uuid4()}_{uploaded_file.filename}"
    file_path = upload_path / unique_filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)

    return str(file_path)

def delete_file(file_path: str):
    """
    Delete a file if it exists.
    """
    path = Path(file_path)
    if path.exists():
        path.unlink()
        return True
    return False

def clean_directory(dir_path: str):
    """
    Delete all files in a directory.
    """
    path = Path(dir_path)
    if not path.exists():
        return

    for file in path.iterdir():
        if file.is_file():
            file.unlink()
