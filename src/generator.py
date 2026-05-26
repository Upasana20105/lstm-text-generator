# src/generator.py
"""Text generation using trained LSTM model."""
import numpy as np
from tensorflow.keras import Model
from src.preprocessor import TextPreprocessor
from config import config

class TextGenerator:
    """Generate text using a trained LSTM model."""
    
    def __init__(self, model: Model, preprocessor: TextPreprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.sequence_length = config.sequence_length

    def _sample_with_temperature(self, predictions: np.ndarray, temperature: float) -> int:
        """Sample from predictions with temperature scaling."""
        predictions = np.asarray(predictions).astype("float64")
        
        # Apply temperature scaling
        log_preds = np.log(predictions + 1e-10) / temperature
        exp_preds = np.exp(log_preds)
        predictions = exp_preds / np.sum(exp_preds)
        
        # Sample from the distribution
        sampled_index = np.random.multinomial(1, predictions, 1)
        return np.argmax(sampled_index)

    def generate(self,
                 seed_text: str,
                 length: int = config.generation_length,
                 temperature: float = config.temperature) -> str:
        """Generate text starting from a seed sequence."""
        seed_text = self.preprocessor.clean_text(seed_text)
        if len(seed_text) < self.sequence_length:
            raise ValueError(
                f"Seed text must be at least {self.sequence_length} characters. "
                f"Got {len(seed_text)} characters."
            )
            
        # Start with the last sequence_length characters of seed
        current_sequence = seed_text[-self.sequence_length:]
        generated = ""
        
        print(f"Generating {length} characters with temperature {temperature}...")
        for _ in range(length):
            # Encode current sequence
            encoded = self.preprocessor.encode_text(current_sequence)
            encoded = encoded.reshape(1, self.sequence_length)
            
            # Predict next character probabilities
            predictions = self.model.predict(encoded, verbose=0)[0]
            
            # Sample next character
            next_idx = self._sample_with_temperature(predictions, temperature)
            next_char = self.preprocessor.idx_to_char[next_idx]
            
            # Update sequence and generated text
            generated += next_char
            current_sequence = current_sequence[1:] + next_char
            
        return generated

    def generate_multiple(self,
                          seed_text: str,
                          temperatures: list = [0.3, 0.5, 0.7, 1.0],
                          length: int = 300) -> dict:
        """Generate text at multiple temperature settings for comparison."""
        results = {}
        for temp in temperatures:
            results[temp] = self.generate(seed_text, length, temp)
        return results
