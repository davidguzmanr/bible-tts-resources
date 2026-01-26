"""
Utilities for analyzing MP3 audio files.

Provides functions to scan directories for MP3 files, extract their durations,
and compute total audio length statistics.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from tqdm import tqdm
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
import math

def get_audio_duration(file_path: Path, audio_format: str) -> float:
    """Get audio duration in seconds. Returns NaN if unreadable."""
    try:
        if audio_format == "mp3":
            audio = MP3(file_path)
        elif audio_format == "wav":
            audio = WAVE(file_path)
        else:
            return float("nan")
        return round(audio.info.length, 2)
    except Exception:
        return float("nan")


def get_all_audio_files(
    audios_dir: str = "data/audios",
    alignment_filter: str = "exclude"
) -> pd.DataFrame:
    """
    Get all audio files in the audios directory.
    
    Args:
        audios_dir: Path to the audios directory containing language folders.
        alignment_filter: How to handle Alignment directories:
            - "exclude": Skip files in Alignment directories (default)
            - "only": Only include files in Alignment directories
            - "all": Include all files regardless of Alignment directories
        
    Returns:
        DataFrame with columns: language, testament, book, file_name, file_path, 
                               format, file_size_mb, duration_seconds
    """
    if alignment_filter not in ("exclude", "only", "all"):
        raise ValueError(f"alignment_filter must be 'exclude', 'only', or 'all', got '{alignment_filter}'")
    
    audios_path = Path(audios_dir)
    
    # Find all audio files (mp3 and wav)
    audio_extensions = ["*.mp3", "*.wav"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audios_path.glob(f"**/{ext}"))
    
    results = []
    for audio_file in tqdm(audio_files, desc="Collecting audio files"):
        # Extract info from path
        # Structure: data/audios/{language}/{Testament - format}/{Book}/{file}
        relative_path = audio_file.relative_to(audios_path)
        parts = relative_path.parts
        
        # Filter based on Alignment directories
        is_in_alignment = "Alignment" in parts
        if alignment_filter == "exclude" and is_in_alignment:
            continue
        if alignment_filter == "only" and not is_in_alignment:
            continue
        
        if len(parts) >= 4:
            language = parts[0]
            testament_format = parts[1]  # e.g., "New Testament - mp3"
            book = parts[2]
            file_name = parts[3]
        else:
            # Handle unexpected structure
            language = parts[0] if len(parts) > 0 else "unknown"
            testament_format = parts[1] if len(parts) > 1 else "unknown"
            book = parts[2] if len(parts) > 2 else "unknown"
            file_name = audio_file.name
        
        # Get file size in MB
        file_size_mb = audio_file.stat().st_size / (1024 * 1024)
        
        # Get audio format from extension
        audio_format = audio_file.suffix.lower().lstrip(".")
        
        # Get duration (mutagen reads metadata without decoding the full file)
        duration_seconds = get_audio_duration(audio_file, audio_format)
        
        results.append({
            "language": language,
            "testament_format": testament_format,
            "book": book,
            "file_name": file_name,
            "file_path": str(audio_file),
            "format": audio_format,
            "file_size_mb": round(file_size_mb, 2),
            "duration_seconds": duration_seconds
        })
    
    return pd.DataFrame(results)


