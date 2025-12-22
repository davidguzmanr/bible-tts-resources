import os
import requests
import zipfile
from io import BytesIO


def download_and_extract_texts(language: str, urls: dict[str, str], base_path: str = "dataset"):
    """
    Download zip files and extract them to organized folders.
    
    Args:
        language: Language name (used for folder structure)
        urls: Dictionary mapping keys to download URLs
        base_path: Base directory for downloads (default: "dataset")
    """
    # Create base directories
    base_dir = f"{base_path}/{language}"
    text_dir = f"{base_dir}/text"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    for key, file_url in urls.items():
        print(f"Downloading {key} from {file_url}...")
        
        # Download the zip file
        response = requests.get(file_url)
        response.raise_for_status()
        
        # Save zip file to dataset/{language}
        zip_path = f"{base_dir}/{key}.zip"
        with open(zip_path, "wb") as f:
            f.write(response.content)
        print(f"Saved zip to {zip_path}")
        
        # Create target directory for extraction
        extract_dir = f"{text_dir}/{key}"
        os.makedirs(extract_dir, exist_ok=True)
        
        # Unzip to dataset/{language}/text/{key}
        with zipfile.ZipFile(BytesIO(response.content), 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")

    print("Done!")


# Example usage
if __name__ == "__main__":
    LANGUAGE = "Ahirani"
    url = {
        "usx": "https://openbible-api-1.biblica.com/artifactContent/68de89d141a2a80e0f03b71b",
        "usfm": "https://openbible-api-1.biblica.com/artifactContent/68de8f2841a2a80e0f043529"
    }
    
    download_and_extract(LANGUAGE, url)