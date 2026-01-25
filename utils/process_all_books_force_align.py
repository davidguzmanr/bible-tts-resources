"""
Process all Bible books for a language using force alignment.

This script is similar to process_all_books.py but uses force alignment
via ReadAlongs Studio instead of timing files.

Usage:
    python utils/process_all_books_force_align.py \
        -base_path "data/audios/LanguageName" \
        -language "und" \
        -workers 4

The expected directory structure:
    data/audios/LanguageName/
        New Testament - mp3/
            1 Corinthians/
                1CO_001.mp3
                1CO_002.mp3
                ...
            Matthew/
                MAT_001.mp3
                ...
        Old Testament - mp3/
            Genesis/
                GEN_001.mp3
                ...
    
    data/texts/LanguageName/
        Paratext (USFM)/release/USX_1/
            1CO.usx
            MAT.usx
            GEN.usx
            ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# Testaments to process
TESTAMENTS = ["New Testament - mp3", "Old Testament - mp3"]


def get_book_code(book_folder_path: str) -> str | None:
    """
    Extract book code from audio files in the folder.
    
    Looks for files like '1CO_001.mp3' and extracts '1CO'.
    """
    for f in os.listdir(book_folder_path):
        if f.lower().endswith(('.mp3', '.wav')) and '_' in f:
            return f.split('_')[0]
    return None


def find_usx_file(usfm_folder: str, book_code: str) -> str | None:
    """
    Find the USX/USFM file for a given book code.
    
    Checks for .usx, .usfm, and .sfm extensions.
    """
    for ext in ['.usx', '.usfm', '.sfm', '.USX', '.USFM', '.SFM']:
        path = os.path.join(usfm_folder, f"{book_code}{ext}")
        if os.path.exists(path):
            return path
    return None


def build_dataframe(base_path: str) -> pd.DataFrame:
    """
    Build a dataframe with all books to process.
    
    Args:
        base_path: Base path to audio files (e.g., data/audios/Lingala)
    
    Returns:
        DataFrame with columns: book_name, book_code, testament, audio_folder,
                               book_usx, output, usx_exists
    """
    # Look for USX files in the expected location
    usfm_folder = os.path.join(
        os.path.dirname(os.path.dirname(base_path)), 
        "texts", 
        os.path.basename(base_path), 
        "Paratext (USFM)/release/USX_1"
    )
    
    # Alternative locations to check
    alt_usfm_folders = [
        os.path.join(os.path.dirname(os.path.dirname(base_path)), "texts", os.path.basename(base_path), "USX"),
        os.path.join(os.path.dirname(os.path.dirname(base_path)), "texts", os.path.basename(base_path)),
        os.path.join(base_path, "USX"),
        os.path.join(base_path, "texts"),
    ]
    
    # Find the first existing USX folder
    if not os.path.exists(usfm_folder):
        for alt_folder in alt_usfm_folders:
            if os.path.exists(alt_folder):
                usfm_folder = alt_folder
                break
    
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
            
            # Get book code from audio files
            book_code = get_book_code(book_folder_path)
            if book_code is None:
                print(f"Warning: Could not find book code for {book_folder}")
                continue
            
            # Find USX file
            book_usx = find_usx_file(usfm_folder, book_code)
            usx_exists = book_usx is not None
            
            if book_usx is None:
                book_usx = os.path.join(usfm_folder, f"{book_code}.usx")  # Expected path
            
            # Build output path
            output = os.path.join(output_base, book_folder)
            
            rows.append({
                'book_name': book_folder,
                'book_code': book_code,
                'testament': testament,
                'audio_folder': book_folder_path,
                'book_usx': book_usx,
                'output': output,
                'usx_exists': usx_exists
            })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['testament', 'book_name']).reset_index(drop=True)
    return df


def process_single_book(
    row: dict, 
    script_path: str = "utils/force_align_book.py",
    language: str = "und",
) -> str:
    """
    Process a single book using force alignment.
    
    Args:
        row: DataFrame row with book information
        script_path: Path to the force_align_book.py script
        language: Language code for g2p
    
    Returns:
        Status message
    """
    if not row['usx_exists']:
        return f"Skipped {row['book_name']}: USX file not found"
    
    cmd = [
        "python", script_path,
        "-audio_folder", row['audio_folder'],
        "-book_usx", row['book_usx'],
        "-output", row['output'],
        "-language", language,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        return f"✓ Completed {row['book_name']} ({row['book_code']})"
    else:
        # Extract a brief error message
        error_lines = result.stderr.strip().split('\n')
        error_msg = error_lines[-1] if error_lines else "Unknown error"
        return f"✗ Failed {row['book_name']}: {error_msg[:100]}"


def run_processing(
    df: pd.DataFrame, 
    script_path: str = "utils/force_align_book.py",
    language: str = "und",
    max_workers: int = 4,
):
    """
    Run force alignment for all books in the dataframe.
    
    Args:
        df: DataFrame with book information
        script_path: Path to the force_align_book.py script
        language: Language code for g2p
        max_workers: Number of parallel workers
    """
    # Filter to only books with USX files
    df_to_process = df[df['usx_exists']].copy()
    
    if df_to_process.empty:
        print("No books with USX files found to process.")
        return
    
    print(f"\nProcessing {len(df_to_process)} books with {max_workers} parallel workers...")
    print("=" * 60)
    
    completed = 0
    total = len(df_to_process)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_book = {
            executor.submit(
                process_single_book, row.to_dict(), script_path, language
            ): row['book_name']
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


def main():
    parser = argparse.ArgumentParser(
        description='Process all Bible books using force alignment'
    )
    parser.add_argument(
        '-base_path', '--base_path',
        type=str, 
        required=True,
        help='Base path to audio files (e.g., data/audios/Lingala)'
    )
    parser.add_argument(
        '-language', '--language',
        type=str,
        default="und",
        help='Language code for g2p (default: "und" for undetermined)'
    )
    parser.add_argument(
        '-workers', '--workers',
        type=int, 
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    
    args = parser.parse_args()
    
    # Build the dataframe
    print("=" * 60)
    print("Force Alignment - Process All Books")
    print("=" * 60)
    print(f"Base path: {args.base_path}")
    print(f"Language: {args.language}")
    
    df = build_dataframe(args.base_path)
    
    if df.empty:
        print("No books found to process.")
        return 1
    
    # Save dataframe to CSV
    csv_path = os.path.join(args.base_path, "books_to_process_force_align.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDataFrame saved to {csv_path}")
    
    # Display summary
    print("\n=== Books to Process ===")
    print(f"Total books: {len(df)}")
    print(f"USX files found: {df['usx_exists'].sum()}")
    print(f"USX files missing: {(~df['usx_exists']).sum()}")
    
    print("\n=== DataFrame Preview ===")
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.width', None)
    print(df[['book_name', 'book_code', 'testament', 'usx_exists']].to_string())
    
    # Show missing USX files
    missing = df[~df['usx_exists']]
    if len(missing) > 0:
        print("\n=== Missing USX Files ===")
        print(missing[['book_name', 'book_code', 'book_usx']].to_string())
    
    if args.dry_run:
        print("\n=== Dry Run Mode - No processing performed ===")
        return 0
    
    # Run processing
    print("\n" + "=" * 60)
    run_processing(
        df, 
        language=args.language,
        max_workers=args.workers
    )
    print("\n=== All Processing Complete ===")
    
    return 0


if __name__ == '__main__':
    exit(main())
