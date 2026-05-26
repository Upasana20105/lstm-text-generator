# main.py
"""Main entry point for LSTM text generation project pipeline."""
import numpy as np
from pathlib import Path
from config import config
from src.data_loader import download_dataset
from src.preprocessor import TextPreprocessor
from src.model import build_lstm_model, compile_model
from src.trainer import ModelTrainer
from src.generator import TextGenerator

def main():
    """Execute the complete text generation pipeline."""
    print("=" * 60)
    print("LSTM Text Generation Pipeline")
    print("=" * 60)
    
    # ------------------------------------------------------------------------
    # Step 1: Load and preprocess data
    # ------------------------------------------------------------------------
    print("\n[1/4] Loading dataset...")
    raw_text = download_dataset()
    print(f"Raw text length: {len(raw_text):,} characters")
    
    # Initialize preprocessor and clean text
    preprocessor = TextPreprocessor()
    clean_text = preprocessor.clean_text(raw_text)
    print(f"Cleaned text length: {len(clean_text):,} characters")
    
    # Build vocabulary and create sequences
    preprocessor.build_vocabulary(clean_text)
    X, y = preprocessor.create_sequences()
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    
    # ------------------------------------------------------------------------
    # Step 2: Build model
    # ------------------------------------------------------------------------
    print("\n[2/4] Building model...")
    model = build_lstm_model(vocab_size=preprocessor.vocab_size)
    model = compile_model(model)
    model.summary()
    
    # ------------------------------------------------------------------------
    # Step 3: Train model
    # ------------------------------------------------------------------------
    print("\n[3/4] Training model...")
    trainer = ModelTrainer(model)
    history = trainer.train(X, y)
    
    # Print training summary
    summary = trainer.get_training_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f" {key}: {value:.4f}")
        else:
            print(f" {key}: {value}")
            
    # ------------------------------------------------------------------------
    # Step 4: Generate text
    # ------------------------------------------------------------------------
    print("\n[4/4] Generating text samples...")
    # Load best model from the checkpoint callback saves
    trainer.load_best_model()
    generator = TextGenerator(trainer.model, preprocessor)
    
    # Extract seed text from the dataset
    seed_text = clean_text[:config.sequence_length + 50]
    
    # Generate samples at different temperatures
    print("\n" + "=" * 60)
    print("GENERATED TEXT SAMPLES")
    print("=" * 60)
    
    temperatures = [0.3, 0.5, 0.7, 1.0]
    samples = generator.generate_multiple(seed_text=seed_text, temperatures=temperatures, length=300)
    
    # Save and display results
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "generated_samples.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Seed text:\n{seed_text[:100]}...\n\n")
        for temp, text in samples.items():
            header = f"\n{'='*60}\nTemperature: {temp}\n{'='*60}\n"
            print(header)
            print(text[:500])
            f.write(header)
            f.write(text)
            f.write("\n")
            
    print(f"\nSamples saved to {output_file}")
    print("\nPipeline complete!")

if __name__ == "__main__":
    main()
