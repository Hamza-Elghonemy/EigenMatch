import kagglehub
import time
import os

def download_dataset_with_retry(dataset_name, max_retries=3, timeout_delay=30):
    """Download dataset with retry logic and better error handling"""
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} to download dataset...")
            path = kagglehub.dataset_download(dataset_name)
            print("Path to dataset files:", path)
            return path
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Waiting {timeout_delay} seconds before retry...")
                time.sleep(timeout_delay)
            else:
                print("All download attempts failed.")
                raise e

# Try to download the dataset
try:
    path = download_dataset_with_retry("kostastokis/simpsons-faces/")
except Exception as e:
    print(f"Failed to download dataset: {e}")
    print("Please check your internet connection and Kaggle API credentials.")