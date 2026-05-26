# config.py
"""Centralized hyperparameters and settings for easy experimentation."""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Data settings
    data_url: str = "https://www.gutenberg.org/files/100/100-0.txt"
    data_path: Path = Path("data/shakespeare.txt")
    
    # Preprocessing
    sequence_length: int = 100  # Number of characters per input sequence
    step_size: int = 3         # Sliding window step for creating sequences
    
    # Model architecture
    embedding_dim: int = 256
    lstm_units_1: int = 256
    lstm_units_2: int = 128
    dropout_rate: float = 0.2
    
    # Training
    batch_size: int = 128
    epochs: int = 50
    validation_split: float = 0.1
    learning_rate: float = 0.001
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001
    
    # Generation
    temperature: float = 0.5  # Controls randomness (lower is more deterministic)
    generation_length: int = 500
    
    # Paths
    checkpoint_dir: Path = Path("checkpoints")
    output_dir: Path = Path("outputs")

# Instantiate a global config object for export
config = Config()
