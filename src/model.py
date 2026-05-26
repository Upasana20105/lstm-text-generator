"""
LSTM model architecture for text generation.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from typing import Tuple, Optional

from config import config


def build_lstm_model(vocab_size: int,
                     sequence_length: int = config.sequence_length,
                     embedding_dim: int = config.embedding_dim,
                     lstm_units_1: int = config.lstm_units_1,
                     lstm_units_2: int = config.lstm_units_2,
                     dropout_rate: float = config.dropout_rate) -> Model:
    """
    Build a stacked LSTM model for character-level text generation.
    
    Architecture:
        1. Embedding layer: Maps character indices to dense vectors
        2. LSTM layer 1: First recurrent layer with return_sequences=True
        3. Dropout: Regularization to prevent overfitting
        4. LSTM layer 2: Second recurrent layer
        5. Dropout: Additional regularization
        6. Dense output: Softmax over vocabulary for next character prediction
    
    Args:
        vocab_size: Number of unique characters in vocabulary.
        sequence_length: Length of input sequences.
        embedding_dim: Dimension of character embeddings.
        lstm_units_1: Units in first LSTM layer.
        lstm_units_2: Units in second LSTM layer.
        dropout_rate: Dropout probability.
    
    Returns:
        Compiled Keras Model.
    """
    model = Sequential([
        # Embedding layer: converts indices to dense vectors
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=sequence_length,
            name="embedding"
        ),
        
        # First LSTM layer with return_sequences for stacking
        layers.LSTM(
            units=lstm_units_1,
            return_sequences=True,
            name="lstm_1"
        ),
        layers.Dropout(dropout_rate, name="dropout_1"),
        
        # Second LSTM layer
        layers.LSTM(
            units=lstm_units_2,
            return_sequences=False,
            name="lstm_2"
        ),
        layers.Dropout(dropout_rate, name="dropout_2"),
        
        # Output layer: probability distribution over vocabulary
        layers.Dense(
            units=vocab_size,
            activation="softmax",
            name="output"
        )
    ], name="lstm_text_generator")
    
    return model


def compile_model(model: Model,
                  learning_rate: float = config.learning_rate) -> Model:
    """
    Compile model with optimizer and loss function.
    
    Args:
        model: Keras model to compile.
        learning_rate: Learning rate for Adam optimizer.
    
    Returns:
        Compiled model.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def build_deeper_model(vocab_size: int,
                       sequence_length: int = config.sequence_length,
                       num_lstm_layers: int = 3,
                       lstm_units: int = 256) -> Model:
    """
    Build a deeper LSTM model for experimentation.
    
    Args:
        vocab_size: Number of unique characters.
        sequence_length: Input sequence length.
        num_lstm_layers: Number of stacked LSTM layers.
        lstm_units: Units per LSTM layer.
    
    Returns:
        Compiled Keras Model.
    """
    model = Sequential(name="deep_lstm_generator")
    
    model.add(layers.Embedding(vocab_size, config.embedding_dim, 
                               input_length=sequence_length))
    
    for i in range(num_lstm_layers):
        return_sequences = i < num_lstm_layers - 1
        model.add(layers.LSTM(lstm_units, return_sequences=return_sequences,
                              name=f"lstm_{i+1}"))
        model.add(layers.Dropout(config.dropout_rate))
    
    model.add(layers.Dense(vocab_size, activation="softmax"))
    
    return model
