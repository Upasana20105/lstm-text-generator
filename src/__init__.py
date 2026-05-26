# src/__init__.py
"""
LSTM Text Generation Package.
Exposes the core components for data loading, preprocessing, model building, training, and generation.
"""

from .data_loader import download_dataset, load_text
from .preprocessor import TextPreprocessor
from .model import build_lstm_model, compile_model
from .trainer import ModelTrainer
from .generator import TextGenerator

__all__ = [
    "download_dataset",
    "load_text",
    "TextPreprocessor",
    "build_lstm_model",
    "compile_model",
    "ModelTrainer",
    "TextGenerator",
)
