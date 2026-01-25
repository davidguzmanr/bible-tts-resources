# Script to process all Bible books for Lingala
import os
import argparse
import pandas as pd
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Testaments to process
TESTAMENTS = ["New Testament - mp3", "Old Testament - mp3"]

def find_audio_files(folder_path):
    """Recursively find audio files (mp3/wav) and return (folder_containing_them, list_of_files)"""
    for root, dirs, files in os.walk(folder_path):
        audio_files = [f for f in files if (f.endswith('.mp3') or f.endswith('.wav')) and '_' in f]
        if audio_files:
            return root, audio_files
    return None, []

def get_book_code(book_folder_path):
    """Extract book code from audio files in the folder (e.g., '1CO' from '1CO_001.mp3')
    Searches recursively to handle nested folder structures."""
    audio_folder, audio_files = find_audio_files(book_folder_path)
    if audio_files:
        # Return the book code from the first audio file
        return audio_files[0].split('_')[0]
    return None

def get_audio_folder(book_folder_path):
    """Get the actual folder containing audio files (handles nested structures)"""
    audio_folder, audio_files = find_audio_files(book_folder_path)
    return audio_folder

def build_dataframe(base_path):
    """Build a dataframe with all books to process"""
    timing_folder = os.path.join(base_path, "Timing Files/Timing Files Bundle")
    usfm_folder = os.path.join(os.path.dirname(os.path.dirname(base_path)), "texts", os.path.basename(base_path), "Paratext (USFM)/release/USX_1")
    output_base = os.path.join(base_path, "Alignment")
    
    rows = []
    
    for testament in TESTAMENTS:
        testament_path = os.path.join(base_path, testament)
        if not os.path.exists(testament_path):
            print(f"Warning: {testament_path} does not exist")
            continue
            
        for book_folder in os.listdir(testament_path):
            book_folder_path = os.path.join(testament_path, book_folder)
            
            if not os.path.isdir(book_folder_path):
                continue
                
            # Get book code from audio files (searches recursively)
            book_code = get_book_code(book_folder_path)
            if book_code is None:
                print(f"Warning: Could not find book code for {book_folder}")
                continue
            
            # Build paths - get the actual folder containing audio files
            wav_folder = get_audio_folder(book_folder_path)
            if wav_folder is None:
                print(f"Warning: Could not find audio folder for {book_folder}")
                continue
            book_sfm = os.path.join(usfm_folder, f"{book_code}.usfm")
            output = os.path.join(output_base, book_folder)
            
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
    if len(df) > 0:
        df = df.sort_values(['testament', 'book_name']).reset_index(drop=True)
    return df

def process_single_book(row, script_path="utils/split_verse_with_timing.py"):
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
        # Get the last traceback and error message for better debugging
        stderr = result.stderr.strip()
        error_lines = stderr.split('\n')
        
        # Find the actual error (usually starts with a known error type)
        error_types = ['ValueError', 'KeyError', 'FileNotFoundError', 'TypeError', 'IndexError', 'Exception', 'Error']
        error_msg = None
        traceback_file = None
        
        for i, line in enumerate(error_lines):
            # Look for the file/line info from traceback
            if 'File "' in line and ', line ' in line:
                traceback_file = line.strip()
            # Look for the actual error message
            for err_type in error_types:
                if line.strip().startswith(err_type):
                    error_msg = line.strip()
                    break
            if error_msg:
                break
        
        # Build informative error message
        if error_msg:
            if traceback_file:
                return f"✗ Failed {row['book_name']} ({row['book_code']}): {error_msg}\n    Location: {traceback_file}\n    WAV folder: {row['wav_folder']}\n    Timing folder: {row['timing_folder']}"
            else:
                return f"✗ Failed {row['book_name']} ({row['book_code']}): {error_msg}"
        else:
            # Fallback: show last few meaningful lines
            meaningful_lines = [line for line in error_lines 
                               if line.strip() and not line.startswith('Processing') 
                               and not line.startswith('Splitting') and '|' not in line]
            error_msg = meaningful_lines[-1] if meaningful_lines else stderr[-500:]
            return f"✗ Failed {row['book_name']} ({row['book_code']}): {error_msg}"

def run_processing(df, script_path="utils/split_verse_with_timing.py", max_workers=4):
    """Run the split_verse_with_timing script for each row in the dataframe in parallel"""
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
    parser = argparse.ArgumentParser(description='Process all Bible books for a given language')
    parser.add_argument('-base_path', '--base_path', type=str, required=True,
                        help='Base path to audio files (e.g., data/audios/Lingala)')
    parser.add_argument('-workers', '--workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Processing: {args.base_path}")
    print(f"{'='*60}")
    
    # Build and display the dataframe
    df = build_dataframe(args.base_path)
    
    if len(df) == 0:
        print("ERROR: No valid books found. Check folder structure and audio files.")
        exit(1)
    
    # Save dataframe to CSV
    csv_path = os.path.join(args.base_path, "books_to_process.csv")
    df.to_csv(csv_path, index=False)
    print(f"DataFrame saved to {csv_path}")
    
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
    
    # Run processing
    print("\n" + "="*60)
    run_processing(df, max_workers=args.workers)
    print("\n=== All Processing Complete ===")
