# src/data_loader.py
"""Data loading utilities for fetching and caching text datasets."""
import requests
from pathlib import Path
from typing import Optional
from config import config

def download_dataset(url: str = config.data_url,
                     save_path: Path = config.data_path,
                     force_download: bool = False) -> str:
    """Download text dataset from URL and cache locally."""
    save_path = Path(save_path)
    
    if save_path.exists() and not force_download:
        print(f"Loading cached dataset from {save_path}")
        return save_path.read_text(encoding="utf-8")
        
    print(f"Downloading dataset from {url}...")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    text = response.text
    
    save_path.write_text(text, encoding="utf-8")
    print(f"Dataset saved to {save_path} ({len(text):,} characters)")
    return text

def load_text(path: Optional[Path] = None) -> str:
    """Load text from a local file."""
    path = path or config.data_path
    return Path(path).read_text(encoding="utf-8")
