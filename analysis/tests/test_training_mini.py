#!/usr/bin/env python3
"""
Quick training test with mini dataset
"""

import sys
sys.path.append('.')

from transformer_morphological_analyzer import (
    SanskritTransformerMorphAnalyzer,
    VedicMorphologicalTrainingDataGenerator,
    SanskritTransformerTrainer
)
import torch
import tempfile
import os

def test_training_pipeline():
    print("Testing MINI Training Pipeline...")

    # Create comprehensive feature set
    features = [
        'retroflex_l', 'visarga_final', 'diphthongs_ai', 'diphthongs_au',
        'subjunctive_full', 'particle_sma', 'philosophical_terms', 'long_compounds'
    ]

    # Create model and data generator
    model = SanskritTransformerMorphAnalyzer(num_morphological_classes=len(features))
    data_gen = VedicMorphologicalTrainingDataGenerator(features)

    # Create mini training data
    mini_samples = []
    test_texts = [
        "atha brahmana puṇḍarīkaṃ sma ha vai ātman",
        "ṛgvede sma particle testing diphthong ai au",
        "philosophical dharma mokṣa terms testing here",
        "some long compound word testing superlong"
    ]

    for i, text in enumerate(test_texts):
        sample = data_gen.analyze_text_for_training(text, "Early Vedic")
        sample['text_name'] = f"test_text_{i}"
        mini_samples.append(sample)

    print(f"✓ Created {len(mini_samples)} training samples")

    # Test trainer
    trainer = SanskritTransformerTrainer(model)

    # Test single training step
    print("\n🏃‍♂️ Testing single training step...")
    try:
        loss = trainer.train_epoch(mini_samples)
        print(f"✓ Training step completed! Loss: {loss:.4f}")

        # Test validation
        print("\n📊 Testing validation...")
        val_metrics = trainer.validate(mini_samples)  # Using same data for demo
        print(f"✓ Validation completed!")
        print(f"  Morphological Loss: {val_metrics['morphological_loss']:.4f}")
        print(f"  Period Loss: {val_metrics['period_loss']:.4f}")
        print(f"  Total Loss: {val_metrics['total_loss']:.4f}")

        print("\n🎯 SUCCESS! Training pipeline works correctly!")
        print("Ready for full corpus training!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training_pipeline()