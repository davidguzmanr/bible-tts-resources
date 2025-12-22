"""
Download Utility for Bible TTS Resources
=========================================

This module provides utilities for downloading and extracting Bible TTS audio
resources from artifact links. It handles:

1. Parsing HTML files to extract artifact download links
2. Downloading files with progress indication
3. Extracting ZIP archives with progress tracking
4. Batch processing of multiple resources
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Dict

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def extract_artifact_links(html_file_path: str) -> Dict[str, Dict[str, str]]:
    """
    Parse an HTML file and extract artifact download links with their sections.

    Searches for anchor tags containing "artifactContent" in their href
    attribute and maps the link text to the URL along with the section
    (e.g., "Old Testament - wav", "New Testament - mp3") it belongs to.

    Args:
        html_file_path: Path to the HTML file containing artifact links.

    Returns:
        A dictionary mapping artifact names (link text) to a dict containing:
            - "url": The download URL
            - "section": The section/header the artifact belongs to
    """
    with open(html_file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    result: Dict[str, Dict[str, str]] = {}

    # Find all section containers (li elements with an "opener" link)
    for section_li in soup.find_all("li"):
        opener = section_li.find("a", class_="opener")
        if not opener:
            continue

        # Extract section name from the opener link
        section_name = opener.get_text(strip=True)

        # Find the slide div containing the artifact links
        slide_div = section_li.find("div", class_="slide")
        if not slide_div:
            continue

        # Extract all artifact links within this section
        for a in slide_div.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            if "artifactContent" in href and text:
                # Skip Word documents - we only need the audio files
                if text == "Word":
                    continue
                result[text] = {"url": href, "section": section_name}

    return result


def safe_folder_name(name: str) -> str:
    """
    Sanitize a string to be safe for use as a folder name.

    Removes or replaces characters that are invalid in file system paths
    across different operating systems.

    Args:
        name: The original name to sanitize.

    Returns:
        A sanitized folder name safe for use on most file systems.
        Returns "unnamed" if the result would be empty.
    """
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    return name.rstrip(". ") or "unnamed"


def download_file_with_progress(url: str, dest_path: Path, timeout: int = 60) -> None:
    """
    Download a file from a URL with a progress bar.

    Creates parent directories if they don't exist. Shows download progress
    in bytes with automatic unit scaling (KB, MB, GB).

    Args:
        url: The URL to download from.
        dest_path: The destination path where the file will be saved.
        timeout: Request timeout in seconds. Defaults to 60.

    Raises:
        requests.HTTPError: If the download request fails.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        total = int(r.headers.get("Content-Length", 0))
        desc = f"Downloading {dest_path.name}"

        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
            leave=False,
        ) as pbar:
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


def unzip_file_with_progress(zip_path: Path, extract_to: Path) -> None:
    """
    Extract a ZIP archive with a progress bar.

    Creates the extraction directory if it doesn't exist. Shows extraction
    progress by file count.

    Args:
        zip_path: Path to the ZIP file to extract.
        extract_to: Directory where contents will be extracted.

    Raises:
        zipfile.BadZipFile: If the file is not a valid ZIP archive.
    """
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        desc = f"Unzipping {zip_path.name}"

        with tqdm(total=len(members), desc=desc, unit="file", leave=False) as pbar:
            for member in members:
                zf.extract(member, extract_to)
                pbar.update(1)


def download_and_unzip_all(
    links: Dict[str, Dict[str, str]],
    output_dir: str,
    *,
    overwrite: bool = False,
    timeout: int = 60,
) -> Dict[str, Path]:
    """
    Download and extract multiple resources from artifact links.

    Processes each link by downloading the ZIP file (if not already present
    or if overwrite is enabled) and extracting its contents. Shows overall
    progress across all resources. Files are organized by section.

    Args:
        links: Dictionary mapping resource names to dicts containing
            "url" and "section" keys.
        output_dir: Base directory where resources will be saved.
            Each resource gets its own subdirectory under its section.
        overwrite: If True, re-download files even if they exist.
            Defaults to False.
        timeout: Request timeout in seconds for each download.
            Defaults to 60.

    Returns:
        A dictionary mapping original resource names to their
        extracted directory paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Path] = {}

    with tqdm(links.items(), desc="Processing books", unit="book") as overall_bar:
        for name, info in overall_bar:
            url = info["url"]
            section = safe_folder_name(info["section"])
            folder_name = safe_folder_name(name)

            # Organize by section, then by book name
            section_dir = out / section
            book_dir = section_dir / folder_name
            zip_path = book_dir / f"{folder_name}.zip"

            overall_bar.set_postfix_str(f"{section}/{folder_name}")

            if zip_path.exists() and not overwrite:
                pass
            else:
                download_file_with_progress(url, zip_path, timeout=timeout)

            unzip_file_with_progress(zip_path, book_dir)
            results[name] = book_dir

    return results
