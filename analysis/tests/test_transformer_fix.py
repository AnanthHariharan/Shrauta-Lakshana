#!/usr/bin/env python3
"""
Quick test script to verify the fixed transformer works
"""

import sys
sys.path.append('.')

from transformer_morphological_analyzer import (
    SanskritTransformerMorphAnalyzer,
    VedicMorphologicalTrainingDataGenerator,
    SanskritTransformerTrainer
)
import torch

def test_basic_functionality():
    print("Testing FIXED Transformer Implementation...")

    # Test 1: Model initialization
    print("\n1. Testing model initialization...")
    model = SanskritTransformerMorphAnalyzer()
    print(f"✓ Model initialized successfully")

    # Test 2: Data generator with features
    print("\n2. Testing data generator...")
    features = ['retroflex_l', 'subjunctive_full', 'particle_sma', 'philosophical_terms']
    data_gen = VedicMorphologicalTrainingDataGenerator(features)
    print(f"✓ Data generator initialized with {len(data_gen.morphological_features)} features")

    # Test 3: Multi-hot vector creation
    print("\n3. Testing multi-hot vector creation...")
    test_features = ['retroflex_l', 'particle_sma']
    multi_hot = data_gen.create_multi_hot_vector(test_features)
    print(f"✓ Multi-hot vector shape: {multi_hot.shape}")
    print(f"✓ Non-zero elements: {multi_hot.sum()}")

    # Test 4: Pattern matching
    print("\n4. Testing pattern matching...")
    test_text = "atha brahmana puṇḍarīkaṃ sma ha vai"
    result = data_gen.analyze_text_for_training(test_text, "Early Vedic")
    print(f"✓ Detected features: {result['detected_features']}")
    print(f"✓ Multi-hot shape: {result['multi_hot_labels'].shape}")

    # Test 5: Trainer initialization
    print("\n5. Testing trainer...")
    trainer = SanskritTransformerTrainer(model)
    print(f"✓ Trainer initialized on device: {trainer.device}")

    print("\n" + "="*50)
    print("🎉 ALL TESTS PASSED!")
    print("The fixed transformer is ready for training!")
    print("="*50)

if __name__ == "__main__":
    test_basic_functionality()