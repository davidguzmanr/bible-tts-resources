# Bible TTS Resources

A toolkit for creating Text-to-Speech (TTS) datasets from Bible audio recordings. This project downloads, aligns, and processes Bible audio/text pairs from [Open.Bible](https://open.bible/) into verse-level segments suitable for TTS training.

The full list of available languages is in [Bible audio resources clean](https://docs.google.com/spreadsheets/d/1P4xk-MgjP7nxWTuo8-pTmVTeICUw1dQOT7GXmjgjsSc/edit?gid=830487007#gid=830487007).

## Features

- **Download** Bible audio files and text transcripts from Open.Bible
- **Align** audio with text using two methods:
  - **Timing files**: When timing files are available from Biblica
  - **Forced alignment**: Using [ReadAlongs Studio](https://github.com/ReadAlongs/Studio) when timing files are not available
- **Split** chapter-level audio into individual verse segments
- **Analyze** audio statistics and data quality

## Installation

### Using Conda (Recommended)

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate bible-tts-resources
```

### Using pip

```bash
pip install -r requirements.txt
```

### Additional Dependencies

Some alignment methods require `ffmpeg` and `sox`:

```bash
# Using Conda (recommended)
conda install -c conda-forge ffmpeg sox

# Ubuntu/Debian
sudo apt-get install ffmpeg sox libsox-fmt-mp3
```

## Usage

### 1. Download Data

#### Download Audio Files

Place HTML files containing artifact links in `html_files/audio/`, then use the notebook or run:

```python
from utils import download_audios

# Extract links from HTML file
links = download_audios.extract_artifact_links("html_files/audio/Yoruba.html")

# Download and extract all files
download_audios.download_and_unzip_all(links, "data/audios/Yoruba")
```

#### Download Text Files

```python
from utils import download_texts

# Extract links from HTML file
links = download_texts.extract_artifact_links("html_files/text/Yoruba.html")

# Download and extract all files
download_texts.download_and_unzip_all(links, "data/texts/Yoruba")
```

### 2. Align Audio with Text

#### Method A: Using Timing Files (Preferred)

For languages with timing files from Biblica, alignment is more accurate:

```bash
# Process a single book
python utils/split_verse_with_timing.py \
    -wav_folder "data/audios/Yoruba/New Testament - mp3/Matthew" \
    -timing_folder "data/audios/Yoruba/Timing Files/Timing Files Bundle" \
    -book_sfm "data/texts/Yoruba/Paratext (USFM)/release/USX_1/MAT.usfm" \
    -output "data/audios/Yoruba/Alignment/Matthew"

# Process all books for a language
python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Yoruba" \
    --timing_folder "data/audios/Yoruba/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Yoruba/Paratext (USFM)/release/USX_1" \
    --workers 8
```

#### Method B: Using Forced Alignment (ReadAlongs)

For languages without timing files, use zero-shot forced alignment:

```bash
# Process a single book
python utils/force_align_book.py \
    -audio_folder "data/audios/Swahili/New Testament - mp3/Matthew" \
    -book_usx "data/texts/Swahili/USX/release/USX_1/MAT.usx" \
    -output "data/audios/Swahili/Alignment/Matthew" \
    -language "und"

# Process all books for a language
python utils/process_all_books_force_align.py \
    --base_path "data/audios/Swahili" \
    --usfm_folder "data/texts/Swahili/USX/release/USX_1" \
    --language "und" \
    --workers 4 \
    --chapter-intro "chapter introduction"
```

**Options:**
- `-language`: Language code for g2p (use `"und"` for undetermined)
- `--chapter-intro`: Placeholder text to absorb speaker's chapter announcements
- `--dry-run`: Preview what would be processed without running

### 3. Analyze Audio Statistics

```python
from utils import audio_stats

# Get all audio files with durations
df = audio_stats.get_all_audio_files(
    audios_dir="data/audios",
    alignment_filter="only"  # "exclude", "only", or "all"
)

# View statistics
print(f"Total duration: {df['duration_seconds'].sum() / 3600:.2f} hours")
print(df.groupby('language')['duration_seconds'].sum())
```

### 4. Check Data Quality

Use the data checks module to identify outliers and validate TTS datasets:

```python
from utils import data_checks

# Remove outliers from alignment DataFrame
# Returns DataFrame with a 'label' column classifying each sample
alignment_df = data_checks.remove_outliers(alignment_df, num_std_devs=2.0)

# View label distribution
print(alignment_df.label.value_counts())
```

The checker validates and labels samples as:
- `BEST`: Clean samples suitable for TTS training
- `TOO_LONG`: Audio clips over 30 seconds
- `TOO_SHORT_TRANS`: Transcripts under 10 characters
- `OFFENDING_DATA`: Pairs with more text than audio (bad for CTC)
- `NON_NORMAL`: Pairs outside the specified standard deviations from mean ratio

## Output Format

After alignment, each verse is saved as:
- `{BOOK}_{CHAPTER:03d}_Verse_{VERSE:03d}.wav` - Audio segment (22050 Hz, mono, 16-bit)
- `{BOOK}_{CHAPTER:03d}_Verse_{VERSE:03d}.txt` - Text transcript

Example:
```
MAT_001_Verse_001.wav
MAT_001_Verse_001.txt
MAT_001_Verse_002.wav
MAT_001_Verse_002.txt
...
```

## Supported Languages

Languages with **timing files** (higher accuracy alignment):
Assamese, Bengali, Central Kurdish, Chhattisgarhi, Dholuo, Ewe, Gujarati, Haryanvi, Hausa, Hiligaynon, Hindi, Igbo, Kannada, Lingala, Malayalam, Marathi, Ndebele, Oromo, Punjabi, Tamil, Telugu, Toma, Twi (Akuapem), Twi (Asante), Ukrainian, Urdu, Vietnamese, Yoruba

Languages using **forced alignment**:
Apali, Arabic Standard, Chichewa, Dawro, Gamo, Gofa, Haitian Creole, Kikuyu, Luganda, Shona, Swahili, Turkish

## Notes

- **Chhattisgarhi**: Multiple audio formats were available; kept the one following standard naming convention.
- **Toma**: Transcripts follow a non-standard format; may require additional preprocessing.

## References

- [coqui-ai/open-bible-scripts](https://github.com/coqui-ai/open-bible-scripts) - Template for MFA-based alignment
- [ReadAlongs Studio](https://github.com/ReadAlongs/Studio) - Zero-shot text-speech alignment
- [coqui-ai/data-checker](https://github.com/coqui-ai/data-checker) - Data quality validation for TTS
- [Open.Bible](https://open.bible/) - Source of Bible audio recordings

## License

See [LICENSE](LICENSE) for details.
