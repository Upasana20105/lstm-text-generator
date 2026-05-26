# src/trainer.py
"""Model training utilities with callbacks and checkpointing."""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from config import config

class ModelTrainer:
    """Handles model training with proper callbacks and validation."""
    
    def __init__(self, model: Model, checkpoint_dir: Path = config.checkpoint_dir):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = None

    def _create_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Create training callbacks for monitoring and checkpointing."""
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                min_delta=config.min_delta,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(self.checkpoint_dir / "best_model.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        return callbacks

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              epochs: int = config.epochs,
              batch_size: int = config.batch_size,
              validation_split: float = config.validation_split) -> Dict[str, Any]:
        """Train the model on prepared sequences."""
        print(f"\nTraining on {len(X):,} sequences...")
        print(f"Batch size: {batch_size}, Epochs: {epochs}")
        print("-" * 50)
        
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=self._create_callbacks(),
            verbose=1
        )
        return self.history.history

    def load_best_model(self) -> None:
        """Load the best saved model from checkpoint."""
        model_path = self.checkpoint_dir / "best_model.keras"
        if model_path.exists():
            self.model = tf.keras.models.load_model(model_path)
            print(f"Loaded best model from {model_path}")
        else:
            print("No checkpoint found, using current model")

    def get_training_summary(self) -> dict:
        """Get summary statistics from training."""
        if self.history is None or not hasattr(self.history, 'history'):
            return {}
        h = self.history.history
        return {
            "final_loss": h["loss"][-1],
            "final_val_loss": h["val_loss"][-1],
            "final_accuracy": h["accuracy"][-1],
            "final_val_accuracy": h["val_accuracy"][-1],
            "best_val_loss": min(h["val_loss"]),
            "epochs_trained": len(h["loss"])
        }
