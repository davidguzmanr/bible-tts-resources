# Imports 

import os, re
import json
import argparse
import time

from collections import defaultdict

import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run verse split pipeline")
    parser.add_argument("-wav_folder", "--path_to_wavs", required=True)
    parser.add_argument("-timing_folder", "--path_to_timings", required=True)
    parser.add_argument("-book_sfm", "--path_to_book_sfm", required=True)
    parser.add_argument("-output", "--output", required=True)

    args = parser.parse_args()
    
    path_to_wavs = args.path_to_wavs
    path_to_timings = args.path_to_timings
    path_to_book_sfm = args.path_to_book_sfm
    output = args.output
    
    print(f"\n=== Input Arguments ===")
    print(f"WAV folder: {path_to_wavs}")
    print(f"Timing folder: {path_to_timings}")
    print(f"Book SFM: {path_to_book_sfm}")
    print(f"Output: {output}")
    
    # Check if paths exist
    print(f"\n=== Path Validation ===")
    print(f"WAV folder exists: {os.path.exists(path_to_wavs)}")
    print(f"Timing folder exists: {os.path.exists(path_to_timings)}")
    print(f"Book SFM exists: {os.path.exists(path_to_book_sfm)}")
    
    if not os.path.exists(f"{output}"):
        os.makedirs(f"{output}")
    
    # Use dict of dicts to handle non-sequential verse numbers (e.g., merged or skipped verses)
    dict_chap_verse = defaultdict(dict)
    current_chap = None
    current_verse = None
    # Open file for read
    with open(f'{path_to_book_sfm}', 'r') as f: 
        for textline in f:
            current_txt = textline.split()
            if len(current_txt) == 0:
                continue
            if current_txt[0] =='\\c':
                current_chap = current_txt[1]
                current_verse = None
                continue
            
            if current_txt[0] =='\\v':
                # Handle verse ranges like "1-2" by taking the first number
                verse_str = current_txt[1].split('-')[0]
                current_verse = int(verse_str)
                # Extract verse text, preserving Unicode characters (for non-Latin scripts)
                # Remove USFM markers (\word), footnotes, and normalize whitespace
                raw_content = textline[len(current_txt[0]+current_txt[1])+2:]
                content = re.sub(r'\\[a-z]+\*?', '', raw_content)  # Remove USFM markers
                content = re.sub(r'\s+', ' ', content).strip()     # Normalize whitespace
                dict_chap_verse[current_chap][current_verse] = content
            elif len(current_txt) == 1:
                continue 
            elif current_chap and current_verse:
                # Extract continuation text, preserving Unicode characters
                raw_content = textline[len(current_txt[0])+1:] if current_txt[0].startswith('\\') else textline
                content = re.sub(r'\\[a-z]+\*?', '', raw_content)  # Remove USFM markers
                content = re.sub(r'\s+', ' ', content).strip()     # Normalize whitespace
                # Safely append to existing verse content
                if content:
                    if current_verse in dict_chap_verse[current_chap]:
                        dict_chap_verse[current_chap][current_verse] += " " + content
                    else:
                        dict_chap_verse[current_chap][current_verse] = content
    
    print(f"\n=== SFM Parsing Results ===")
    print(f"Total chapters found: {len(dict_chap_verse)}")
    for chap, verses in sorted(dict_chap_verse.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        print(f"  Chapter {chap}: {len(verses)} verses (verse numbers: {sorted(verses.keys())[:5]}...)" if len(verses) > 5 else f"  Chapter {chap}: {len(verses)} verses")
    
    audio_files = [f for f in os.listdir(path_to_wavs) if f.endswith('.wav') or f.endswith('.mp3')]
    print(f"\n=== Audio Files Found ===")
    print(f"Total audio files: {len(audio_files)}")
    if audio_files:
        print(f"First few files: {audio_files[:5]}")
    for file in tqdm(audio_files, desc="Processing audio chapters"):
        book_chap, ext = file.split('.')
        book, chap = book_chap.split('_')
        
        # Global dictionary to keep verse, [time_start, time_end]
        dict_verse_time = defaultdict(lambda : [])
        timing_file_path = os.path.join(path_to_timings, f'{book_chap}.txt')
        print(f"\n  Looking for timing file: {timing_file_path}")
        print(f"  Timing file exists: {os.path.exists(timing_file_path)}")
        # open the and read file on in the first repository             
        with open(timing_file_path, 'r') as f:  # Open file for read
            for textline in f:
                verse_time = textline.split("\t")
                # This handles the file version case
                if len(verse_time) == 1 or len(verse_time[0].split()) == 1:
                    continue
                else:
                    # This skips the Chapter Title and Headings
                    
                    verse, number = verse_time[0].split()
                    if verse != "Verse":
                        continue 
                    else:
                        time = verse_time[1]
                        dict_verse_time[f'{verse}_{number.zfill(3)}'].append(time)
                        if int(number)-1==0:
                            pass
                        else:
                            dict_verse_time[f'{verse}_{str(int(number)-1).zfill(3)}'].append(time)
        
        print(f"  Verses with timing data: {len(dict_verse_time)}")
        if dict_verse_time:
            first_key = list(dict_verse_time.keys())[0]
            print(f"  Sample timing entry: {first_key} -> {dict_verse_time[first_key]}")
                  
        for verse_key in tqdm(dict_verse_time, desc=f"Splitting verses for {book_chap}", leave=False):
            audio = os.path.join(path_to_wavs, file)
            output_file = os.path.join(output, f"{book_chap}_{verse_key}.wav")
            
            # Fix timing format: replace commas with periods for ffmpeg compatibility
            start_time = dict_verse_time[verse_key][0].replace(',', '.')
            
            if len(dict_verse_time[verse_key])==2:
                end_time = dict_verse_time[verse_key][1].replace(',', '.')
                os.system(f'ffmpeg -y -i "{audio}" -ss {start_time} -to {end_time} -loglevel error "{output_file}"')
            else:
                os.system(f'ffmpeg -y -i "{audio}" -ss {start_time} -loglevel error "{output_file}"')
            
            # Get verse number and look up text in dict (handles non-sequential verse numbers)
            verse_num = int(verse_key.split('_')[1])
            chap_num = str(int(chap))
            verse_text = dict_chap_verse.get(chap_num, {}).get(verse_num, "")
            
            if not verse_text:
                print(f"  Warning: No text found for chapter {chap_num}, verse {verse_num}")
            
            with open(os.path.join(output, f'{book_chap}_{verse_key}.txt'), "w", encoding="utf-8") as text_file:
                text_file.write(verse_text)
                text_file.write("\n")  
        
        print(f"  Completed processing {book_chap}")
    
    print(f"\n=== Processing Complete ===")
    print(f"Output written to: {output}")
    
    
    
    
    
