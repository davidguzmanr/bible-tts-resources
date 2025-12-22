import os
import time
from pathlib import Path
from utils import download_audios

# Directory containing HTML files
html_dir = Path("html_files")
output_base_dir = Path("audios")

# Get all HTML files
html_files = sorted(html_dir.glob("*.html"))

print(f"Found {len(html_files)} languages to process")

for i, html_path in enumerate(html_files):
    lang_name = html_path.stem  # Get filename without extension
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(html_files)}] Processing: {lang_name}")
    print(f"{'='*60}")
    
    try:
        # 1) Parse HTML -> dict(name -> URL)
        links = download_audios.extract_artifact_links(str(html_path))
        print(f"Found {len(links)} artifacts")
        
        # 2) Download + unzip into folders
        output_dir = output_base_dir / lang_name
        download_audios.download_and_unzip_all(links, str(output_dir), overwrite=False, timeout=60)
        
        print(f"✓ Successfully processed {lang_name}")
        
    except Exception as e:
        print(f"✗ Failed to process {lang_name}: {e}")
        continue
    
    # Sleep 1 minute between languages (except after the last one)
    if i < len(html_files) - 1:
        print(f"\nSleeping for 1 minute before next language...")
        time.sleep(60)

print(f"\n{'='*60}")
print("All languages processed!")
print(f"{'='*60}")

