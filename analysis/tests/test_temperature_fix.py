#!/usr/bin/env python3
"""
Test temperature parameter fix
"""

import sys
sys.path.append('.')

from transformer_morphological_analyzer import (
    SanskritTransformerMorphAnalyzer,
    VedicMorphologicalTrainingDataGenerator,
    SanskritTransformerTrainer
)
import torch

def test_temperature():
    print("Testing temperature parameter fix...")

    # Create model with temperature
    model = SanskritTransformerMorphAnalyzer(
        num_morphological_classes=8,
        temperature=1.0
    )

    print(f"✅ Model has temperature parameter: {model.temperature}")
    print(f"✅ Temperature value: {model.temperature.item():.2f}")

    # Test forward pass
    inputs = torch.randint(0, 1000, (1, 10))
    attention_mask = torch.ones(1, 10)

    with torch.no_grad():
        outputs = model(inputs, attention_mask)

    print(f"✅ Forward pass works with temperature scaling")
    print(f"✅ Output logits shape: {outputs['morphological_logits'].shape}")

    # Test calibration
    features = ['test_feature'] * 8
    data_gen = VedicMorphologicalTrainingDataGenerator(features)
    trainer = SanskritTransformerTrainer(model)

    # Create mini validation data
    sample = data_gen.analyze_text_for_training("test text", "Early Vedic")
    val_data = [sample]

    try:
        temp = trainer.calibrate_temperature(val_data)
        print(f"✅ Temperature calibration works! Optimal temp: {temp:.2f}")
        print(f"✅ Model temperature updated: {model.temperature.item():.2f}")
        print("🎯 Temperature parameter fix successful!")
    except Exception as e:
        print(f"❌ Calibration still failing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_temperature()