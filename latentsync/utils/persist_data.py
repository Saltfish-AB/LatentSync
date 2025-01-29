import os
import shutil
from typing import Optional

def save_on_persistent_disk(local_file_path: str, new_filename: Optional[str] = None):
    """
    Moves a file to the persistent disk location at /latent-sync-data.
    
    Args:
        local_file_path (str): Path to the local file that needs to be moved.
        new_filename (str, optional): New filename if renaming is required. Defaults to None.

    Returns:
        str: New file path if successful, or an error message.
    """
    try:
        # Ensure the persistent disk mount exists
        persistent_disk_path = "/latent-sync-data"
        if not os.path.exists(persistent_disk_path):
            raise FileNotFoundError(f"Persistent disk path '{persistent_disk_path}' does not exist.")

        # Ensure the source file exists
        if not os.path.isfile(local_file_path):
            raise FileNotFoundError(f"File '{local_file_path}' not found.")

        # Use new filename if provided, else keep original filename
        filename = new_filename if new_filename else os.path.basename(local_file_path)
        new_path = os.path.join(persistent_disk_path, filename)

        # Move the file
        shutil.move(local_file_path, new_path)

        print(f"✅ File moved successfully to: {new_path}")
        return new_path

    except Exception as e:
        print(f"❌ Error: {e}")
        return str(e)
