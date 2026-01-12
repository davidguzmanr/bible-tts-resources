"""
Utilities for parsing USX (Unified Scripture XML) files into pandas DataFrames.

USX is an XML format for Bible scripture texts. This module provides functions
to extract book, chapter, verse, and text content from USX files.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd


# Styles we treat as "opening-of-chapter heading" BEFORE verse 1
HEADING_STYLES = {"h", "mt1", "mt2", "mt3", "s1", "s2", "s3", "ms1", "ms2", "toc1", "toc2", "toc3"}

# Regex handles various verse ID formats:
# - Simple: "1CO 1:5"
# - Ranges: "1CO 1:5-6"
# - Comma-separated: "MAT 18:10,11"
_USX_ID_RE = re.compile(r"^\s*([A-Z0-9]+)\s+(\d+):(\d+)(?:[-,]\d+)*\s*$")
_CHAP_ID_RE = re.compile(r"^\s*([A-Z0-9]+)\s+(\d+)\s*$")


def _parse_usx_id(usx_id: str) -> Tuple[str, int, int]:
    """Parse a USX verse ID like '1CO 1:1', '1CO 1:5-6', or 'MAT 18:10,11' into (book, chapter, verse).
    
    For verse ranges or comma-separated verses, returns the first verse number.
    """
    m = _USX_ID_RE.match(usx_id)
    if not m:
        raise ValueError(f"Unexpected USX id format: {usx_id!r}")
    return m.group(1), int(m.group(2)), int(m.group(3))


def _parse_chapter_id(ch_id: str) -> Tuple[str, int]:
    """Parse a USX chapter ID like '1CO 1' into (book, chapter)."""
    m = _CHAP_ID_RE.match(ch_id)
    if not m:
        raise ValueError(f"Unexpected chapter id format: {ch_id!r}")
    return m.group(1), int(m.group(2))


def usx_to_dataframe(
    file_path: Union[str, Path],
    include_headings: bool = True,
) -> pd.DataFrame:
    """
    Parse a USX file and return a DataFrame with book, chapter, verse, and text.

    Args:
        file_path: Path to the USX file to parse.
        include_headings: If True (default), include verse 0 rows containing
            chapter/section headings. If False, only include actual verses.

    Returns:
        A DataFrame with columns:
        - 'book': The book code (e.g., '1CO', 'GEN')
        - 'chapter': The chapter number (int)
        - 'verse': The verse number (int, 0 for chapter headings if include_headings=True)
        - 'text': The verse text content

    Notes:
        - Verse 0 contains opening-of-chapter headings (if include_headings=True)
        - Notes/footnotes are excluded from the text
        - Verse text is aggregated between sid/eid milestones
    """
    file_path = Path(file_path)
    tree = ET.parse(file_path)
    root = tree.getroot()

    rows: List[Dict[str, object]] = []

    # Track open verses by milestone id 'BOOK C:V' -> list of text fragments
    open_verses: Dict[str, List[str]] = {}
    verse_meta: Dict[str, Tuple[str, int, int]] = {}

    current_book: str = None
    current_chapter: int = None

    # For capturing opening-of-chapter headings before the first verse of that chapter
    chapter_heading_buf: Dict[Tuple[str, int], List[str]] = {}
    chapter_heading_emitted: Dict[Tuple[str, int], bool] = {}

    def add_text_to_open_verses(fragment: str):
        if not fragment:
            return
        frag = fragment.strip()
        if not frag:
            return
        for vid in open_verses:
            open_verses[vid].append(frag)

    def close_verse(vid: str):
        acc = open_verses.pop(vid, None)
        meta = verse_meta.pop(vid, None)
        if meta is None:
            return
        text = " ".join(acc or [])
        text = re.sub(r"\s+", " ", text).strip()
        b, c, v = meta
        rows.append({"book": b, "chapter": c, "verse": v, "text": text})

    def emit_heading_if_needed(book: str, chapter: int):
        key = (book, chapter)
        if chapter_heading_emitted.get(key):
            return
        if include_headings:
            parts = [p.strip() for p in chapter_heading_buf.get(key, []) if p.strip()]
            if parts:
                heading_text = " ".join(parts)
                heading_text = re.sub(r"\s+", " ", heading_text).strip()
                rows.append({"book": book, "chapter": chapter, "verse": 0, "text": heading_text})
        chapter_heading_emitted[key] = True

    def collect_node_text(elem: ET.Element) -> str:
        """Get all text inside elem, ignoring <note> and <verse> nodes, but keeping their tails."""
        parts: List[str] = []
        if elem.text and elem.tag not in ("note", "verse"):
            parts.append(elem.text)
        for ch in list(elem):
            if ch.tag not in ("note", "verse"):
                if ch.text:
                    parts.append(ch.text)
            # Recurse unless skipping notes/verses
            if ch.tag not in ("note", "verse"):
                parts.append(collect_node_text(ch))
            # Always include tail (text after child)
            if ch.tail:
                parts.append(ch.tail)
        return " ".join(p for p in parts if p)

    def walk(elem: ET.Element):
        nonlocal current_book, current_chapter

        # Book code
        if elem.tag == "book":
            current_book = elem.attrib.get("code")

        # Chapter start milestone: <chapter number="1" ... sid="GEN 1"/>
        if elem.tag == "chapter" and "sid" in elem.attrib:
            b, c = _parse_chapter_id(elem.attrib["sid"])
            current_book, current_chapter = b, c
            chapter_heading_buf.setdefault((current_book, current_chapter), [])
            chapter_heading_emitted[(current_book, current_chapter)] = False

        # Opening-of-chapter headings BEFORE first verse: capture para text
        if (
            elem.tag == "para"
            and elem.attrib.get("style") in HEADING_STYLES
            and current_book is not None
            and current_chapter is not None
            and not chapter_heading_emitted.get((current_book, current_chapter), False)
            and not open_verses  # ensure it's before any verse has started
        ):
            txt = collect_node_text(elem)
            if txt:
                chapter_heading_buf[(current_book, current_chapter)].append(txt)

        # Verse start milestone
        if elem.tag == "verse" and "sid" in elem.attrib:
            book, chap, ver = _parse_usx_id(elem.attrib["sid"])
            # If this is the first verse of the chapter, emit heading row first
            emit_heading_if_needed(book, chap)
            verse_meta[elem.attrib["sid"]] = (book, chap, ver)
            open_verses.setdefault(elem.attrib["sid"], [])

        # Verse end milestone
        elif elem.tag == "verse" and "eid" in elem.attrib:
            vid = elem.attrib["eid"]
            if vid in open_verses:
                close_verse(vid)

        # Skip footnotes entirely (their tails are added below)
        elif elem.tag == "note":
            return

        # Add normal element text to any open verses
        if elem.text and elem.tag not in ("verse", "note"):
            add_text_to_open_verses(elem.text)

        # Recurse
        for child in list(elem):
            walk(child)
            if child.tail:
                add_text_to_open_verses(child.tail)

        # Chapter end milestone (optional to handle)
        if elem.tag == "chapter" and "eid" in elem.attrib:
            # If no verse ever appeared, we may still want to emit heading
            try:
                b, c = _parse_chapter_id(elem.attrib["eid"])
                emit_heading_if_needed(b, c)
            except Exception:
                pass

    walk(root)

    # Close any dangling verses (malformed USX)
    for vid in list(open_verses.keys()):
        close_verse(vid)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["book", "chapter", "verse"], kind="stable").reset_index(drop=True)
    return df


def usx_directory_to_dataframe(
    directory_path: Union[str, Path],
    include_headings: bool = True,
) -> pd.DataFrame:
    """
    Parse all USX files in a directory and return a combined DataFrame.

    Args:
        directory_path: Path to the directory containing USX files.
        include_headings: If True (default), include verse 0 rows containing
            chapter/section headings. If False, only include actual verses.

    Returns:
        A DataFrame with columns 'book', 'chapter', 'verse', and 'text',
        combining all USX files found in the directory.
    """
    directory_path = Path(directory_path)
    usx_files = sorted(directory_path.glob("*.usx"))
    
    dfs = []
    for usx_file in usx_files:
        df = usx_to_dataframe(usx_file, include_headings=include_headings)
        dfs.append(df)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["book", "chapter", "verse", "text"])
