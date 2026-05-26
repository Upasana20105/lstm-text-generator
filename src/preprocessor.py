# src/preprocessor.py
"""Text preprocessing and sequence generation for LSTM training."""
import re
import numpy as np
from typing import Tuple, Dict, Optional
from config import config

class TextPreprocessor:
    """Handles text cleaning, tokenization, and sequence generation."""
    
    def __init__(self):
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0
        self._text: str = ""

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for processing."""
        # Convert to lowercase
        text = text.lower()
        # Keep only lowercase letters, basic punctuation, and whitespace
        text = re.sub(r"[^a-z\s.,!?;:\'\-]", "", text)
        # Normalize whitespace (collapse multiple spaces/newlines into one space)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def build_vocabulary(self, text: str) -> None:
        """Build character-level vocabulary from text."""
        self._text = text
        unique_chars = sorted(set(text))
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)
        print(f"Vocabulary size: {self.vocab_size} unique characters")

    def create_sequences(self,
                         sequence_length: int = config.sequence_length,
                         step_size: int = config.step_size) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output sequence pairs for training using a sliding window."""
        if not self._text:
            raise ValueError("Must call build_vocabulary() before creating sequences")
            
        sequences = []
        next_chars = []
        
        for i in range(0, len(self._text) - sequence_length, step_size):
            seq = self._text[i:i + sequence_length]
            target = self._text[i + sequence_length]
            sequences.append([self.char_to_idx[c] for c in seq])
            next_chars.append(self.char_to_idx[target])
            
        X = np.array(sequences, dtype=np.int32)
        y = np.array(next_chars, dtype=np.int32)
        print(f"Created {len(X):,} sequences of length {sequence_length}")
        return X, y

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text string to integer indices."""
        return np.array([self.char_to_idx.get(c, 0) for c in text])

    def decode_indices(self, indices: np.ndarray) -> str:
        """Decode integer indices back to text."""
        return "".join([self.idx_to_char.get(idx, "") for idx in indices])
