from google.cloud import storage
from google.oauth2 import service_account
import os

# Read the environment variable
service_account_path = os.getenv("SERVICE_ACCOUNT_KEY_PATH")

# Check if the environment variable exists
if service_account_path:
    print(f"Service account path: {service_account_path}")
else:
    print("Environment variable SERVICE_ACCOUNT_KEY_PATH is not set.")

# Initialize a GCS client
credentials = service_account.Credentials.from_service_account_file(service_account_path or "/secrets/saltfish-434012-8c642217c8e8.json")

# Initialize a GCS client with the credentials
storage_client = storage.Client(credentials=credentials)

def upload_video_to_gcs(bucket_name: str, source_file_path: str, destination_blob_name: str):
    """
    Uploads a video file to Google Cloud Storage.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_file_path (str): Path to the video file to upload.
        destination_blob_name (str): The destination path/name in the GCS bucket.
    """
    try:
        # Get the bucket
        bucket = storage_client.bucket(bucket_name)

        # Create a blob (object) in the bucket
        blob = bucket.blob(destination_blob_name)

        # Upload the video file
        blob.upload_from_filename(source_file_path)

        print(f"File {source_file_path} uploaded to {destination_blob_name} in bucket {bucket_name}.")
    except Exception as e:
        print(f"An error occurred: {e}")
