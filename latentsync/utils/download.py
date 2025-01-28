import requests
from tqdm import tqdm
import os

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