import subprocess
import os
from src.logger.logging import logger

class GCloudSync:
    
    def sync_folder_from_gcloud(self, gcp_bucket_url, filename, destination):
        # Ensure destination directory exists
        os.makedirs(destination, exist_ok=True)
        
        # Construct full path where the file should be saved
        file_path = os.path.join(destination, filename)
        
        # Construct the gsutil command
        command = [
            "gsutil", 
            "cp", 
            f"gs://{gcp_bucket_url}/{filename}", 
            file_path
        ]
        
        logger.info(f"Executing command: {' '.join(command)}")
        
        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully downloaded {filename} from {gcp_bucket_url} to {destination}")
        else:
            logger.error(f"Failed to download {filename} from {gcp_bucket_url}: {result.stderr}")
   
            
    def sync_folder_to_gcloud(self, gcp_bucket_url, filepath, filename):
        """
        Uploads a file to Google Cloud Storage.

        Args:
            gcp_bucket_url (str): The URL of the Google Cloud Storage bucket.
            filepath (str): The local path of the file to upload.
            filename (str): The name of the file to upload.
        """
        try:
            # Construct the gsutil command
            command = [
                "gsutil",
                "cp",
                os.path.join(filepath, filename),
                f"gs://{gcp_bucket_url}/"
            ]

            # Run the command and capture the output
            result = subprocess.run(command, check=True, text=True, capture_output=True)

            # Log the output
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            # Handle errors in the upload process
            print(f"Error occurred: {e.stderr}")
            raise        


    