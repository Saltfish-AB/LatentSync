import requests
from tqdm import tqdm
import os
import shutil

def download_file(url: str, destination: str, chunk_size: int = 8192) -> None:
    """
    Downloads a file from the given URL and saves it to the specified destination.
    
    Parameters:
        url (str): The URL of the file to download.
        destination (str): The path where the downloaded file will be saved.
        chunk_size (int): The size of each chunk in bytes. Default is 8192 bytes.
    
    Raises:
        Exception: If the download fails or the URL is unreachable.
    """
    try:
        # Make a request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

        # Get the total file size from headers
        total_size = int(response.headers.get("content-length", 0))

        # Open the destination file and write the content in chunks
        with open(destination, "wb") as file, tqdm(
            desc=f"Downloading {destination}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
                bar.update(len(chunk))
        
        print(f"Download completed: {destination}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download file from {url}: {e}")
        raise
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
        raise


def cleanup_file(filepath: str) -> None:
    """
    Deletes the specified file from the filesystem.

    Parameters:
        filepath (str): The path of the file to delete.

    Returns:
        None
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"File {filepath} has been deleted.")
        else:
            print(f"File {filepath} does not exist. No action taken.")
    except Exception as e:
        print(f"Error while trying to delete {filepath}: {e}")
        raise


def cleanup_folder(folder_path: str, keep_folder: bool = True) -> None:
    """
    Removes all files within a folder.
    
    Parameters:
        folder_path (str): The path to the folder whose contents should be deleted.
        keep_folder (bool): If True, keeps the folder structure but removes all contents.
                           If False, removes the folder entirely. Default is True.
    
    Raises:
        Exception: If the folder deletion fails.
    """
    try:
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist. No action taken.")
            return
            
        if not os.path.isdir(folder_path):
            print(f"{folder_path} is not a directory. No action taken.")
            return
            
        if keep_folder:
            # Count the number of files for reporting
            file_count = sum([len(files) for _, _, files in os.walk(folder_path)])
            
            # Remove all contents but keep the folder
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    
            print(f"Removed all contents ({file_count} files) from {folder_path}.")
        else:
            # Remove the entire folder and its contents
            shutil.rmtree(folder_path)
            print(f"Removed folder {folder_path} and all its contents.")
    
    except Exception as e:
        print(f"Error while cleaning up folder {folder_path}: {e}")
        raise