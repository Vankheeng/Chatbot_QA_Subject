from __future__ import annotations

import os
from pathlib import Path

from fastapi import UploadFile


def save_file(uploaded_file: UploadFile, save_file_dir: str) -> str:
    """
    Saves an uploaded file to the specified directory.

    This function ensures that the target directory exists, then saves the uploaded file to it.

    Returns:
        str: The full path of the saved file.

    Raises:
        FileExistsError: If a file with the same name already exists in the directory.
    """
    Path(save_file_dir).mkdir(parents=True, exist_ok=True)

    file_name = uploaded_file.filename
    file_path = os.path.join(save_file_dir, file_name)

    if os.path.exists(file_path):
        raise FileExistsError(f"File {file_name} already exists")

    with open(file_path, 'wb') as file:
        file.write(uploaded_file.file.read())

    return file_path
