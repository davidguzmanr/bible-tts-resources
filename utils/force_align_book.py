"""
Force align a Bible book using ReadAlongs Studio.

This script aligns audio and text from Bible books using force alignment
(via the readalongs library) when timing files are not available.

Usage:
    python utils/force_align_book.py \
        -audio_folder "data/audios/Language/New Testament - mp3/1 Corinthians" \
        -book_usx "data/texts/Language/Paratext/USX/1CO.usx" \
        -output "data/audios/Language/Alignment/1 Corinthians" \
        -language "und"

    # Also works with USFM files:
    python utils/force_align_book.py \
        -audio_folder "data/audios/Language/New Testament - mp3/1 Corinthians" \
        -book_usx "data/texts/Language/Paratext/1CO.usfm" \
        -output "data/audios/Language/Alignment/1 Corinthians" \
        -language "und"

The script will:
1. Parse the USX or USFM file to extract verse text
2. For each chapter audio file, run force alignment with readalongs
3. Split the audio into individual verse segments
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Import the scripture parser from the same directory (handles both USX and USFM)
from usx_parser import scripture_to_dataframe


def strip_numbers_from_text(text: str) -> str:
    """
    Remove standalone numbers from text to avoid g2p issues.
    
    Numbers like "365", "120", "600" cause g2p conversion failures.
    This removes them while preserving the rest of the text structure.
    
    Args:
        text: Input text that may contain numbers
        
    Returns:
        Text with standalone numbers removed
    """
    # Remove standalone numbers (surrounded by whitespace or punctuation)
    # This pattern matches numbers that are whole words (not part of other words)
    stripped = re.sub(r'\b\d+\b', '', text)
    # Clean up any double spaces left behind
    stripped = re.sub(r'  +', ' ', stripped)
    # Clean up spaces before punctuation
    stripped = re.sub(r' ([,;:.!?])', r'\1', stripped)
    return stripped.strip()


def get_chapter_audio_files(audio_folder: str) -> Dict[int, str]:
    """
    Find all audio files in the folder and map them to chapter numbers.
    
    Assumes audio files are named like 'BOOK_001.mp3' or 'BOOK_001.wav'
    where BOOK is the book code and 001 is the chapter number.
    
    Returns:
        Dict mapping chapter number to full file path
    """
    audio_folder = Path(audio_folder)
    chapter_files = {}
    
    for f in audio_folder.iterdir():
        if f.suffix.lower() in ('.mp3', '.wav'):
            # Extract chapter number from filename like "1CO_001.mp3"
            match = re.match(r'^([A-Z0-9]+)_(\d+)\.(mp3|wav)$', f.name, re.IGNORECASE)
            if match:
                chapter_num = int(match.group(2))
                chapter_files[chapter_num] = str(f)
    
    return chapter_files


def prepare_verse_text_file(
    df: pd.DataFrame, 
    chapter: int, 
    output_path: str,
    include_headings: bool = False,
    strip_numbers: bool = True,
) -> List[Tuple[int, str]]:
    """
    Prepare a text file for readalongs alignment from the verse DataFrame.
    
    Each verse is written as a separate line (which becomes a sentence in readalongs).
    Numbers can be optionally stripped from the alignment text to avoid g2p issues,
    while the returned verses list contains the original text with numbers.
    
    Args:
        df: DataFrame with columns ['book', 'chapter', 'verse', 'text']
        chapter: Chapter number to extract
        output_path: Path to write the text file
        include_headings: Whether to include verse 0 (chapter headings)
        strip_numbers: Whether to strip numbers from the alignment text (default True)
    
    Returns:
        List of (verse_number, original_text) tuples for verses included in the file.
        Note: The returned text is the ORIGINAL text (with numbers), while the file
        written to output_path has numbers stripped if strip_numbers=True.
    """
    chapter_df = df[df['chapter'] == chapter].copy()
    
    if not include_headings:
        chapter_df = chapter_df[chapter_df['verse'] > 0]
    
    chapter_df = chapter_df.sort_values('verse')
    
    verses = []
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in chapter_df.iterrows():
            original_text = str(row['text']).strip()
            if original_text:  # Only include non-empty verses
                # Write stripped text for alignment (numbers removed)
                if strip_numbers:
                    alignment_text = strip_numbers_from_text(original_text)
                else:
                    alignment_text = original_text
                f.write(alignment_text + '\n')
                # Return original text (with numbers) for transcript saving
                verses.append((int(row['verse']), original_text))
    
    return verses


def run_readalongs_alignment(
    text_file: str,
    audio_file: str,
    output_dir: str,
    languages: List[str] = None,
    save_temps: bool = False,
) -> Tuple[bool, str]:
    """
    Run readalongs alignment on a text and audio file.
    
    Args:
        text_file: Path to the plain text file
        audio_file: Path to the audio file
        output_dir: Directory to save alignment output
        languages: List of language codes for g2p (first is primary, rest are fallbacks)
        save_temps: Whether to save temporary files
    
    Returns:
        Tuple of (success, message)
    """
    if languages is None:
        languages = ["und"]
    
    try:
        # Import readalongs API
        from readalongs.api import align
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Run alignment
        status, exception, log = align(
            textfile=text_file,
            audiofile=audio_file,
            output_base=output_dir,
            language=languages,  # Pass list of languages for g2p cascade/fallback
            output_formats=["textgrid"],  # TextGrid gives us word timings
            save_temps=save_temps,
            force_overwrite=True,
        )
        
        if status == 0:
            return True, "Alignment successful"
        else:
            # Use repr() to safely convert exception, as some exceptions have broken __str__
            exc_str = repr(exception) if exception else "Unknown error"
            return False, f"Alignment failed: {exc_str}\n{log}"
            
    except ImportError:
        # Fall back to CLI if API import fails
        return run_readalongs_cli(text_file, audio_file, output_dir, languages, save_temps)


def run_readalongs_cli(
    text_file: str,
    audio_file: str,
    output_dir: str,
    languages: List[str] = None,
    save_temps: bool = False,
) -> Tuple[bool, str]:
    """
    Run readalongs alignment using the command line interface.
    
    Fallback method if the Python API is not available.
    """
    if languages is None:
        languages = ["und"]
    
    cmd = [
        "readalongs", "align",
        text_file,
        audio_file,
        output_dir,
        "-l", ",".join(languages),  # Multiple languages separated by comma for g2p cascade
        "-o", "textgrid",
        "-f",  # force overwrite
    ]
    
    if save_temps:
        cmd.append("-s")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        return True, "Alignment successful"
    else:
        return False, f"Alignment failed: {result.stderr}"


def parse_textgrid_for_sentences(textgrid_path: str) -> List[Tuple[float, float, str]]:
    """
    Parse a Praat TextGrid file to extract sentence-level timings.
    
    Args:
        textgrid_path: Path to the TextGrid file
    
    Returns:
        List of (start_time, end_time, text) tuples for each sentence
    """
    try:
        from pympi.Praat import TextGrid
        tg = TextGrid(textgrid_path)
        
        # Get tier names - pympi uses get_tier_name_num() which returns [(num, name), ...]
        tier_info = tg.get_tier_name_num()
        
        # Get the Sentence tier
        sentence_tier = None
        for tier_num, tier_name in tier_info:
            if 'sentence' in tier_name.lower():
                sentence_tier = tg.get_tier(tier_name)
                break
        
        if sentence_tier is None and tier_info:
            # Fall back to first tier (tier_info[0] is (num, name))
            sentence_tier = tg.get_tier(tier_info[0][1])
        
        if sentence_tier is None:
            return []
        
        sentences = []
        for interval in sentence_tier.get_intervals():
            start, end, text = interval
            if text.strip():  # Only include non-empty intervals
                sentences.append((start, end, text.strip()))
        
        return sentences
        
    except ImportError:
        # Parse TextGrid manually if pympi is not available
        return parse_textgrid_manual(textgrid_path)


def parse_textgrid_manual(textgrid_path: str) -> List[Tuple[float, float, str]]:
    """
    Parse TextGrid file manually without external dependencies.
    """
    sentences = []
    
    with open(textgrid_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find Sentence tier
    tier_match = re.search(
        r'name\s*=\s*"Sentence".*?intervals\s*:\s*size\s*=\s*\d+(.+?)(?=item\s*\[|$)',
        content, re.DOTALL | re.IGNORECASE
    )
    
    if not tier_match:
        return sentences
    
    tier_content = tier_match.group(1)
    
    # Extract intervals
    interval_pattern = re.compile(
        r'xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"',
        re.DOTALL
    )
    
    for match in interval_pattern.finditer(tier_content):
        start = float(match.group(1))
        end = float(match.group(2))
        text = match.group(3).strip()
        if text:
            sentences.append((start, end, text))
    
    return sentences


def split_audio_by_verses(
    audio_file: str,
    verse_timings: List[Tuple[int, float, float]],
    output_folder: str,
    book_code: str,
    chapter: int,
) -> int:
    """
    Split audio file into verse segments using ffmpeg.
    
    Args:
        audio_file: Path to the source audio file
        verse_timings: List of (verse_number, start_time, end_time) tuples
        output_folder: Directory to save verse audio files
        book_code: Book code for naming (e.g., '1CO')
        chapter: Chapter number
    
    Returns:
        Number of verses successfully extracted
    """
    os.makedirs(output_folder, exist_ok=True)
    success_count = 0
    
    for verse_num, start_time, end_time in verse_timings:
        output_file = os.path.join(
            output_folder, 
            f"{book_code}_{chapter:03d}_Verse_{verse_num:03d}.wav"
        )
        
        cmd = [
            'ffmpeg', '-y',
            '-i', audio_file,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-acodec', 'pcm_s16le',  # WAV format
            '-ar', '22050',  # Sample rate
            '-ac', '1',  # Mono
            '-loglevel', 'error',
            output_file
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            success_count += 1
        else:
            print(f"  Warning: Failed to extract verse {verse_num}: {result.stderr.decode()}")
    
    return success_count


def match_verses_to_sentences(
    verses: List[Tuple[int, str]],
    sentences: List[Tuple[float, float, str]],
) -> List[Tuple[int, float, float]]:
    """
    Match verse numbers to sentence timings.
    
    Since we write one verse per line, and readalongs treats each line as a sentence,
    the order should match directly.
    
    Args:
        verses: List of (verse_number, text) tuples
        sentences: List of (start_time, end_time, text) tuples from alignment
    
    Returns:
        List of (verse_number, start_time, end_time) tuples
    """
    verse_timings = []
    
    # Simple 1:1 matching by order
    for i, (verse_num, verse_text) in enumerate(verses):
        if i < len(sentences):
            start, end, _ = sentences[i]
            verse_timings.append((verse_num, start, end))
        else:
            print(f"  Warning: No alignment found for verse {verse_num}")
    
    if len(sentences) > len(verses):
        print(f"  Warning: {len(sentences) - len(verses)} extra sentences in alignment")
    
    return verse_timings


def save_verse_transcripts(
    verses: List[Tuple[int, str]],
    output_folder: str,
    book_code: str,
    chapter: int,
):
    """
    Save verse text files alongside audio files.
    """
    for verse_num, text in verses:
        output_file = os.path.join(
            output_folder,
            f"{book_code}_{chapter:03d}_Verse_{verse_num:03d}.txt"
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text + '\n')


def process_book(
    audio_folder: str,
    book_usx: str,
    output_folder: str,
    languages: List[str] = None,
    include_headings: bool = False,
    save_temps: bool = False,
    strip_numbers: bool = True,
) -> Dict[str, any]:
    """
    Process an entire Bible book: parse USX, align chapters, split into verses.
    
    Args:
        audio_folder: Folder containing chapter audio files
        book_usx: Path to USX or USFM file
        output_folder: Output folder for verse audio and text files
        languages: List of language codes for g2p (first is primary, rest are fallbacks)
        include_headings: Whether to include chapter headings (verse 0)
        save_temps: Whether to save temporary alignment files
        strip_numbers: Whether to strip numbers from alignment text (to avoid g2p issues)
    
    Returns:
        Dict with processing statistics
    """
    if languages is None:
        languages = ["und"]
    stats = {
        'book': None,
        'chapters_processed': 0,
        'chapters_failed': 0,
        'verses_extracted': 0,
        'errors': []
    }
    
    # Parse the scripture file (USX or USFM)
    print(f"\n=== Parsing {book_usx} ===")
    try:
        df = scripture_to_dataframe(book_usx, include_headings=include_headings)
    except Exception as e:
        stats['errors'].append(f"Failed to parse USX: {e}")
        print(f"Error: {e}")
        return stats
    
    if df.empty:
        stats['errors'].append("No verses found in USX file")
        print("Error: No verses found in USX file")
        return stats
    
    book_code = df['book'].iloc[0]
    stats['book'] = book_code
    chapters = sorted(df['chapter'].unique())
    
    print(f"Book: {book_code}")
    print(f"Chapters found in text: {len(chapters)}")
    print(f"Total verses: {len(df[df['verse'] > 0])}")
    
    # Find chapter audio files
    chapter_audio = get_chapter_audio_files(audio_folder)
    print(f"Audio files found: {len(chapter_audio)}")
    
    if not chapter_audio:
        stats['errors'].append("No audio files found")
        print("Error: No audio files found")
        return stats
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each chapter
    for chapter in chapters:
        if chapter not in chapter_audio:
            print(f"\n  Chapter {chapter}: No audio file found, skipping")
            continue
        
        print(f"\n  Processing {book_code} Chapter {chapter}...")
        audio_file = chapter_audio[chapter]
        
        # Create temp directory for alignment output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare verse text file (numbers stripped for alignment, but kept in verses list)
            text_file = os.path.join(temp_dir, f"chapter_{chapter}.txt")
            verses = prepare_verse_text_file(df, chapter, text_file, include_headings, strip_numbers)
            
            if not verses:
                print(f"    No verses found for chapter {chapter}")
                continue
            
            print(f"    Verses: {len(verses)}")
            
            # Run alignment
            align_output = os.path.join(temp_dir, f"aligned_{chapter}")
            success, message = run_readalongs_alignment(
                text_file, audio_file, align_output, languages, save_temps
            )
            
            if not success:
                print(f"    Alignment failed: {message}")
                stats['chapters_failed'] += 1
                stats['errors'].append(f"Chapter {chapter}: {message}")
                continue
            
            # Parse alignment results
            # Find TextGrid file in output directory
            textgrid_files = list(Path(align_output).glob("*.TextGrid"))
            if not textgrid_files:
                print(f"    No TextGrid output found")
                stats['chapters_failed'] += 1
                continue
            
            textgrid_path = str(textgrid_files[0])
            sentences = parse_textgrid_for_sentences(textgrid_path)
            
            if not sentences:
                print(f"    No sentence timings found in TextGrid")
                stats['chapters_failed'] += 1
                continue
            
            print(f"    Aligned sentences: {len(sentences)}")
            
            # Match verses to sentence timings
            verse_timings = match_verses_to_sentences(verses, sentences)
            
            # Split audio into verses
            num_extracted = split_audio_by_verses(
                audio_file, verse_timings, output_folder, book_code, chapter
            )
            
            # Save verse transcripts
            save_verse_transcripts(verses, output_folder, book_code, chapter)
            
            print(f"    Extracted {num_extracted} verse audio files")
            stats['chapters_processed'] += 1
            stats['verses_extracted'] += num_extracted
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Force align a Bible book using ReadAlongs Studio"
    )
    parser.add_argument(
        "-audio_folder", "--audio_folder", 
        required=True,
        help="Folder containing chapter audio files (e.g., 1CO_001.mp3)"
    )
    parser.add_argument(
        "-book_usx", "--book_usx",
        required=True,
        help="Path to the scripture file for the book (USX, USFM, or SFM format)"
    )
    parser.add_argument(
        "-output", "--output",
        required=True,
        help="Output folder for verse audio and text files"
    )
    parser.add_argument(
        "-language", "--language",
        default="und",
        help="Language code for g2p (default: 'und' for undetermined)"
    )
    parser.add_argument(
        "--g2p-fallback",
        default=None,
        help="Fallback language(s) for g2p when primary fails (e.g., 'eng' for numbers). "
             "Can specify multiple comma-separated (e.g., 'eng,fra')"
    )
    parser.add_argument(
        "--include-headings",
        action="store_true",
        help="Include chapter headings (verse 0) in alignment"
    )
    parser.add_argument(
        "--save-temps",
        action="store_true",
        help="Save temporary alignment files for debugging"
    )
    parser.add_argument(
        "--keep-numbers",
        action="store_true",
        help="Keep numbers in alignment text (by default, numbers are stripped to avoid g2p issues)"
    )
    
    args = parser.parse_args()
    
    # Build languages list (primary + fallbacks)
    languages = [args.language]
    if args.g2p_fallback:
        fallbacks = [lang.strip() for lang in args.g2p_fallback.split(",")]
        languages.extend(fallbacks)
    
    print("=" * 60)
    print("Force Alignment using ReadAlongs Studio")
    print("=" * 60)
    print(f"Audio folder: {args.audio_folder}")
    print(f"Book USX: {args.book_usx}")
    print(f"Output folder: {args.output}")
    print(f"Language: {args.language}")
    if args.g2p_fallback:
        print(f"G2P fallback: {args.g2p_fallback}")
    print(f"Strip numbers: {not args.keep_numbers}")
    
    # Verify paths exist
    if not os.path.exists(args.audio_folder):
        print(f"Error: Audio folder not found: {args.audio_folder}")
        return 1
    
    if not os.path.exists(args.book_usx):
        print(f"Error: Book USX file not found: {args.book_usx}")
        return 1
    
    # Process the book
    stats = process_book(
        audio_folder=args.audio_folder,
        book_usx=args.book_usx,
        output_folder=args.output,
        languages=languages,
        include_headings=args.include_headings,
        save_temps=args.save_temps,
        strip_numbers=not args.keep_numbers,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(f"Book: {stats['book']}")
    print(f"Chapters processed: {stats['chapters_processed']}")
    print(f"Chapters failed: {stats['chapters_failed']}")
    print(f"Verses extracted: {stats['verses_extracted']}")
    
    if stats['errors']:
        print("\nErrors:")
        for error in stats['errors']:
            print(f"  - {error}")
    
    return 0 if stats['chapters_failed'] == 0 else 1


if __name__ == "__main__":
    exit(main())
