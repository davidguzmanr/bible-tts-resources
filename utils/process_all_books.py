# Script to process all Bible books for Lingala
import os
import pandas as pd
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Base paths
BASE_PATH = "data/audios/Lingala"
TIMING_FOLDER = os.path.join(BASE_PATH, "Timing Files/Timing Files Bundle")
USFM_FOLDER = "data/texts/Lingala/Paratext (USFM)/release/USX_1"
OUTPUT_BASE = os.path.join(BASE_PATH, "Alignment")

# Testaments to process
TESTAMENTS = ["New Testament - mp3", "Old Testament - mp3"]

def get_book_code(book_folder_path):
    """Extract book code from mp3 files in the folder (e.g., '1CO' from '1CO_001.mp3')"""
    for f in os.listdir(book_folder_path):
        if f.endswith('.mp3') and '_' in f:
            return f.split('_')[0]
    return None

def build_dataframe():
    """Build a dataframe with all books to process"""
    rows = []
    
    for testament in TESTAMENTS:
        testament_path = os.path.join(BASE_PATH, testament)
        if not os.path.exists(testament_path):
            print(f"Warning: {testament_path} does not exist")
            continue
            
        for book_folder in os.listdir(testament_path):
            book_folder_path = os.path.join(testament_path, book_folder)
            
            if not os.path.isdir(book_folder_path):
                continue
                
            # Get book code from mp3 files
            book_code = get_book_code(book_folder_path)
            if book_code is None:
                print(f"Warning: Could not find book code for {book_folder}")
                continue
            
            # Build paths
            wav_folder = book_folder_path
            timing_folder = TIMING_FOLDER
            book_sfm = os.path.join(USFM_FOLDER, f"{book_code}.usfm")
            output = os.path.join(OUTPUT_BASE, book_folder)
            
            # Check if USFM file exists
            usfm_exists = os.path.exists(book_sfm)
            
            rows.append({
                'book_name': book_folder,
                'book_code': book_code,
                'testament': testament,
                'wav_folder': wav_folder,
                'timing_folder': timing_folder,
                'book_sfm': book_sfm,
                'output': output,
                'usfm_exists': usfm_exists
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['testament', 'book_name']).reset_index(drop=True)
    return df

def process_single_book(row, script_path="utils/split_verse_lingala.py"):
    """Process a single book - used by parallel executor"""
    if not row['usfm_exists']:
        return f"Skipped {row['book_name']}: USFM file not found"
    
    cmd = [
        "python", script_path,
        "-wav_folder", row['wav_folder'],
        "-timing_folder", row['timing_folder'],
        "-book_sfm", row['book_sfm'],
        "-output", row['output']
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        return f"✓ Completed {row['book_name']} ({row['book_code']})"
    else:
        return f"✗ Failed {row['book_name']}: {result.stderr[:200]}"

def run_processing(df, script_path="utils/split_verse_lingala.py", max_workers=4):
    """Run the split_verse script for each row in the dataframe in parallel"""
    # Filter to only books with USFM files
    df_to_process = df[df['usfm_exists']].copy()
    
    print(f"\nProcessing {len(df_to_process)} books with {max_workers} parallel workers...")
    print("="*60)
    
    completed = 0
    total = len(df_to_process)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_book = {
            executor.submit(process_single_book, row, script_path): row['book_name']
            for _, row in df_to_process.iterrows()
        }
        
        # Process results as they complete
        for future in as_completed(future_to_book):
            book_name = future_to_book[future]
            completed += 1
            try:
                result = future.result()
                print(f"[{completed}/{total}] {result}")
            except Exception as e:
                print(f"[{completed}/{total}] ✗ Error processing {book_name}: {e}")

if __name__ == '__main__':
    # Build and display the dataframe
    df = build_dataframe()
    
    print("\n=== Books to Process ===")
    print(f"Total books: {len(df)}")
    print(f"USFM files found: {df['usfm_exists'].sum()}")
    print(f"USFM files missing: {(~df['usfm_exists']).sum()}")
    
    print("\n=== DataFrame Preview ===")
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.width', None)
    print(df[['book_name', 'book_code', 'testament', 'usfm_exists']].to_string())
    
    # Show any missing USFM files
    missing = df[~df['usfm_exists']]
    if len(missing) > 0:
        print("\n=== Missing USFM Files ===")
        print(missing[['book_name', 'book_code', 'book_sfm']].to_string())
    
    # Ask for confirmation before processing
    print("\n" + "="*60)
    response = input("Do you want to run processing for all books? (yes/no): ")
    if response.lower() == 'yes':
        workers = input("How many parallel workers? (default: 4): ").strip()
        max_workers = int(workers) if workers else 4
        run_processing(df, max_workers=max_workers)
        print("\n=== All Processing Complete ===")
    else:
        print("Processing cancelled. You can inspect the dataframe above.")
        # Save dataframe to CSV for reference
        csv_path = "data/audios/Lingala/books_to_process.csv"
        df.to_csv(csv_path, index=False)
        print(f"DataFrame saved to {csv_path}")
