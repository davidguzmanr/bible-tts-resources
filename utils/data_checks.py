# Data quality checks for audio-text pairs
#
# Takes a DataFrame with columns (audio_file, text),
# and adds a 'label' column with one of:
#   - BEST: good quality data
#   - UNREADABLE: audio file cannot be read
#   - TOO_LONG: audio clip over max length
#   - TOO_SHORT_TRANS: transcript under min length
#   - OFFENDING_DATA: more text than audio (bad for CTC)
#   - NON_NORMAL: ratio is outlier (more than N std devs from mean)

import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def get_abspath(df: pd.DataFrame, base_dir: Optional[str] = None) -> pd.DataFrame:
    """Resolve absolute paths for audio files.
    
    Args:
        df: DataFrame with 'audio_file' column
        base_dir: Optional base directory to resolve relative paths
        
    Returns:
        DataFrame with added 'abspath' column
    """
    def find_abspath(audio_path, base_dir):
        if os.path.isfile(os.path.abspath(audio_path)):
            return os.path.abspath(audio_path)
        elif base_dir and os.path.isfile(os.path.abspath(os.path.join(base_dir, audio_path))):
            return os.path.abspath(os.path.join(base_dir, audio_path))
        else:
            return None

    df = df.copy()
    df["abspath"] = df["audio_file"].apply(lambda x: find_abspath(x, base_dir))
    return df


def is_audio_readable(audio_path: str) -> bool:
    """Check if an audio file can be read.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        True if readable, False otherwise
    """
    try:
        librosa.load(audio_path, sr=None, duration=0.1)
        return True
    except Exception:
        return False


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Duration in seconds, or -1 if file cannot be read
    """
    try:
        return librosa.get_duration(path=audio_path)
    except Exception:
        return -1


def calculate_num_feat_vectors(seconds: float) -> int:
    """Calculate number of feature vectors from audio duration.
    
    Assumes 20 millisecond feature window step.
    
    Args:
        seconds: Audio duration in seconds
        
    Returns:
        Number of feature vectors
    """
    return int(seconds * 1000 / 20)


def check_data_quality(
    df: pd.DataFrame,
    base_dir: Optional[str] = None,
    num_std_devs: float = 3.0,
    max_audio_len: float = 30.0,
    min_transcript_len: int = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """Check data quality and add labels to DataFrame.
    
    Args:
        df: DataFrame with 'audio_file' and 'text' columns
        base_dir: Optional base directory to resolve relative paths
        num_std_devs: Number of standard deviations for outlier detection
        max_audio_len: Maximum audio length in seconds
        min_transcript_len: Minimum transcript length in characters
        verbose: Whether to print progress messages
        
    Returns:
        DataFrame with added columns including 'label'
    """
    df = df.copy()
    
    # Validate required columns
    if "text" not in df.columns or "audio_file" not in df.columns:
        raise ValueError("DataFrame must have 'text' and 'audio_file' columns")
    
    if verbose:
        print(f"👀 ─ Found {len(df)} <text, audio> pairs")
    
    # Initialize label column
    df["label"] = "BEST"
    
    # Resolve absolute paths
    df = get_abspath(df, base_dir)
    
    # Check for unresolved paths
    unresolved_mask = df["abspath"].isna()
    if unresolved_mask.any():
        df.loc[unresolved_mask, "label"] = "UNREADABLE"
        if verbose:
            print(f"🚨 ─ Could not resolve paths for {unresolved_mask.sum()} files")
    
    # Check if audio is readable
    if verbose:
        print(" · Checking if audio is readable...")
    
    readable_mask = df["label"] == "BEST"
    df.loc[readable_mask, "is_readable"] = df.loc[readable_mask, "abspath"].apply(is_audio_readable)
    unreadable_mask = (df["label"] == "BEST") & (df["is_readable"] == False)
    df.loc[unreadable_mask, "label"] = "UNREADABLE"
    
    if verbose:
        unreadable_count = unreadable_mask.sum()
        if unreadable_count:
            print(f"👀 ─ Found {unreadable_count} unreadable audio files")
        else:
            print("😊 Found no unreadable audio files")
    
    # Get audio duration for readable files
    if verbose:
        print(" · Reading audio duration...")
    
    valid_mask = df["label"] == "BEST"
    df.loc[valid_mask, "audio_len"] = df.loc[valid_mask, "abspath"].apply(get_audio_duration)
    
    if verbose:
        total_hours = df.loc[valid_mask, "audio_len"].sum() / 3600
        print(f"👀 ─ Found a total of {total_hours:.2f} hours of readable data")
    
    # Get transcript length
    if verbose:
        print(" · Get transcript length...")
    df["transcript_len"] = df["text"].apply(lambda x: len(str(x)))
    
    # Get number of feature vectors
    if verbose:
        print(" · Get num feature vectors...")
    df["num_feat_vectors"] = df["audio_len"].apply(
        lambda x: calculate_num_feat_vectors(x) if pd.notna(x) and x > 0 else 0
    )
    
    # Check audio length
    too_long_mask = (df["label"] == "BEST") & (df["audio_len"] > max_audio_len)
    df.loc[too_long_mask, "label"] = "TOO_LONG"
    
    if verbose:
        too_long_count = too_long_mask.sum()
        if too_long_count:
            total_hours = df.loc[too_long_mask, "audio_len"].sum() / 3600
            print(f"👀 ┬ Found {too_long_count} audio clips over {max_audio_len} seconds long")
            print(f"   └ Marking {total_hours:.2f} hours of data as TOO_LONG")
        else:
            print(f"😊 Found no audio clips over {max_audio_len} seconds in length")
    
    # Check transcript length
    too_short_mask = (df["label"] == "BEST") & (df["transcript_len"] < min_transcript_len)
    df.loc[too_short_mask, "label"] = "TOO_SHORT_TRANS"
    
    if verbose:
        too_short_count = too_short_mask.sum()
        if too_short_count:
            total_hours = df.loc[too_short_mask, "audio_len"].sum() / 3600
            print(f"👀 ┬ Found {too_short_count} transcripts under {min_transcript_len} characters long")
            print(f"   └ Marking {total_hours:.2f} hours of data as TOO_SHORT_TRANS")
        else:
            print(f"😊 Found no transcripts under {min_transcript_len} characters in length")
    
    # Check input/output ratio (CTC requirement)
    if verbose:
        print(" · Get ratio (num_feats / transcript_len)...")
    
    valid_mask = df["label"] == "BEST"
    df.loc[valid_mask, "input_output_ratio"] = df.loc[valid_mask].apply(
        lambda x: float(x.num_feat_vectors) / float(x.transcript_len) if x.transcript_len > 0 else 0,
        axis=1
    )
    
    offending_mask = (df["label"] == "BEST") & (df["input_output_ratio"] <= 1.0)
    df.loc[offending_mask, "label"] = "OFFENDING_DATA"
    
    if verbose:
        offending_count = offending_mask.sum()
        if offending_count:
            total_hours = df.loc[offending_mask, "audio_len"].sum() / 3600
            print(f"👀 ┬ Found {offending_count} <text, audio> pairs with more text than audio (bad for CTC)")
            print(f"   └ Marking {total_hours:.2f} hours of data as OFFENDING_DATA")
        else:
            print("😊 Found no offending <text, audio> pairs")
    
    # Check for outliers based on lens ratio
    if verbose:
        print(" · Calculating ratio (audio_len : transcript_len)...")
    
    valid_mask = df["label"] == "BEST"
    df.loc[valid_mask, "lens_ratio"] = df.loc[valid_mask].apply(
        lambda x: float(x.audio_len) / float(x.transcript_len) if x.transcript_len > 0 else 0,
        axis=1
    )
    
    # Calculate mean and std only on valid (BEST) samples
    valid_ratios = df.loc[df["label"] == "BEST", "lens_ratio"]
    if len(valid_ratios) > 0:
        mean = valid_ratios.mean()
        std = valid_ratios.std()
        
        df.loc[valid_mask, "lens_ratio_deviation"] = df.loc[valid_mask, "lens_ratio"].apply(
            lambda x: abs(x - mean) - (num_std_devs * std)
        )
        
        outlier_mask = (df["label"] == "BEST") & (df["lens_ratio_deviation"] > 0)
        df.loc[outlier_mask, "label"] = "NON_NORMAL"
        
        if verbose:
            outlier_count = outlier_mask.sum()
            if outlier_count:
                total_hours = df.loc[outlier_mask, "audio_len"].sum() / 3600
                print(f"👀 ┬ Found {outlier_count} <text, audio> pairs more than {num_std_devs} standard deviations from the mean")
                print(f"   └ Marking {total_hours:.2f} hours of data as NON_NORMAL")
            else:
                print(f"😊 Found no <text, audio> pairs more than {num_std_devs} standard deviations from the mean")
    
    # Final summary
    if verbose:
        best_mask = df["label"] == "BEST"
        best_hours = df.loc[best_mask, "audio_len"].sum() / 3600
        best_count = best_mask.sum()
        
        print(f"🎉 ┬ {best_count} samples ({best_hours:.2f} hours) labeled as BEST")
        print(f"   └ Label distribution:")
        for label, count in df["label"].value_counts().items():
            print(f"      - {label}: {count}")
    
    return df


def remove_outliers(
    df: pd.DataFrame,
    num_std_devs: float = 3.0,
    base_dir: Optional[str] = None,
    max_audio_len: float = 30.0,
    min_transcript_len: int = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """Check data quality and return DataFrame with all labels.
    
    Args:
        df: DataFrame with 'audio_file' and 'text' columns
        num_std_devs: Number of standard deviations for outlier detection
        base_dir: Optional base directory to resolve relative paths
        max_audio_len: Maximum audio length in seconds
        min_transcript_len: Minimum transcript length in characters
        verbose: Whether to print progress messages
        
    Returns:
        DataFrame with added 'label' column
    """
    return check_data_quality(
        df,
        base_dir=base_dir,
        num_std_devs=num_std_devs,
        max_audio_len=max_audio_len,
        min_transcript_len=min_transcript_len,
        verbose=verbose
    )


def get_labeled_data(
    df: pd.DataFrame,
    num_std_devs: float = 3.0,
    base_dir: Optional[str] = None,
    max_audio_len: float = 30.0,
    min_transcript_len: int = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """Check data quality and return DataFrame with all labels.
    
    This is an alias for check_data_quality for a more intuitive API.
    
    Args:
        df: DataFrame with 'audio_file' and 'text' columns
        num_std_devs: Number of standard deviations for outlier detection
        base_dir: Optional base directory to resolve relative paths
        max_audio_len: Maximum audio length in seconds
        min_transcript_len: Minimum transcript length in characters
        verbose: Whether to print progress messages
        
    Returns:
        DataFrame with added 'label' column
    """
    return check_data_quality(
        df,
        base_dir=base_dir,
        num_std_devs=num_std_devs,
        max_audio_len=max_audio_len,
        min_transcript_len=min_transcript_len,
        verbose=verbose
    )
