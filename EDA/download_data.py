import os
from dotenv import load_dotenv
import kagglehub

# Load environment variables from .env file
load_dotenv()

# Verify that the required environment variables are set
if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
    raise ValueError("KAGGLE_USERNAME and KAGGLE_KEY must be set in the .env file")

def fetch_kaggle_data():
    """
    Fetches data from Kaggle and saves it to the specified path.

    Parameters:
    dataset (str): The Kaggle dataset identifier (e.g., 'username/dataset-name').
    download_path (str): The local path where the file will be saved.
    """
    endpoint = os.getenv("DATASET_ENDPOINT")
    download_path = "./data/brats2020-training-data"  # Default download path

    path = kagglehub.dataset_download("awsaf49/brats2020-training-data", output_dir=download_path)
    print(f"Dataset downloaded to: {path}")

if __name__ == "__main__":
    fetch_kaggle_data()