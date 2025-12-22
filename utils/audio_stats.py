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
import math

def mp3_durations_dataframe(download_dir: str) -> pd.DataFrame:
    """
    Scan a directory for MP3 files and return a DataFrame with their durations.

    Args:
        download_dir: Path to the directory to scan for MP3 files (recursive).

    Returns:
        A DataFrame with columns 'path' (absolute file path) and 
        'duration_seconds' (duration in seconds, NaN if unreadable).
    """
    base = Path(download_dir)
    mp3_files = sorted(base.rglob("*.mp3"))

    rows = []
    for p in tqdm(mp3_files, desc="Reading MP3 durations", unit="file"):
        try:
            audio = MP3(p)
            duration = float(audio.info.length)  # seconds
        except Exception:
            duration = float("nan")  # couldn't read duration
        rows.append({"path": str(p.resolve()), "duration_seconds": duration})

    return pd.DataFrame(rows)


def print_total_duration(df: pd.DataFrame) -> None:
    """
    Print the total duration of all audio files in human-readable format.

    Args:
        df: A DataFrame containing a 'duration_seconds' column with durations.
    """
    # Drop NaNs just in case
    total_seconds = df["duration_seconds"].dropna().sum()

    total_seconds = int(round(total_seconds))

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    print(f"Total duration: {hours}h {minutes}m {seconds}s")

