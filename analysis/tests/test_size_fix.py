#!/usr/bin/env python3
"""
Test the size mismatch fix
"""

import sys
sys.path.append('.')

from transformer_morphological_analyzer import (
    SanskritTransformerMorphAnalyzer,
    VedicMorphologicalTrainingDataGenerator,
    SanskritTransformerTrainer
)

def test_size_fix():
    print("Testing size mismatch fix...")

    # Define features (should match the model output size)
    features = [
        'retroflex_l', 'visarga_final', 'diphthongs_ai', 'diphthongs_au',
        'subjunctive_full', 'particle_sma', 'philosophical_terms', 'long_compounds'
    ]

    print(f"Features defined: {len(features)}")

    # Create model with matching size
    model = SanskritTransformerMorphAnalyzer(num_morphological_classes=len(features))
    print(f"Model output size: {model.morphological_classifier[-1].out_features}")

    # Create data generator
    data_gen = VedicMorphologicalTrainingDataGenerator(features)
    print(f"Data generator features: {len(data_gen.morphological_features)}")

    # Test training sample creation
    sample = data_gen.analyze_text_for_training("atha brahman sma", "Early Vedic")
    print(f"Multi-hot vector size: {sample['multi_hot_labels'].shape}")

    # Create trainer
    trainer = SanskritTransformerTrainer(model)

    # Test training step
    mini_samples = [sample]
    try:
        loss = trainer.train_epoch(mini_samples)
        print(f"✅ SUCCESS! Training step completed. Loss: {loss:.4f}")
        print("🎯 Size mismatch fixed!")
    except Exception as e:
        print(f"❌ Still failing: {e}")

if __name__ == "__main__":
    test_size_fix()