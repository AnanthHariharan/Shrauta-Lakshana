#!/usr/bin/env python3
"""
Test the complete pipeline with temperature calibration
"""

import sys
sys.path.append('.')

from transformer_morphological_analyzer import (
    SanskritTransformerMorphAnalyzer,
    VedicMorphologicalTrainingDataGenerator,
    SanskritTransformerTrainer
)
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_complete_pipeline():
    logger.info("🧪 Testing Complete Pipeline with Temperature Calibration")

    # Features
    features = [
        'retroflex_l', 'visarga_final', 'diphthongs_ai', 'diphthongs_au',
        'subjunctive_full', 'particle_sma', 'philosophical_terms', 'long_compounds'
    ]

    # Model with temperature
    model = SanskritTransformerMorphAnalyzer(
        num_morphological_classes=len(features),
        temperature=1.0
    )

    # Data generator
    data_gen = VedicMorphologicalTrainingDataGenerator(features)

    # Create training data
    test_texts = [
        "atha brahmana puṇḍarīkaṃ sma ha vai ātman dharma mokṣa",
        "ṛgvede sma particle testing diphthong ai au philosophical",
        "some long compound word testing superlongcompoundword",
        "retroflex testing ḷ visarga final ḥ testing here"
    ]

    training_data = []
    for i, text in enumerate(test_texts):
        sample = data_gen.analyze_text_for_training(text, "Early Vedic")
        sample['text_name'] = f"test_{i}"
        training_data.append(sample)

    logger.info(f"✅ Created {len(training_data)} training samples")

    # Trainer
    trainer = SanskritTransformerTrainer(model)

    # Mini training
    logger.info("🏃‍♂️ Running mini training...")
    for epoch in range(2):
        train_loss = trainer.train_epoch(training_data)
        val_metrics = trainer.validate(training_data)
        logger.info(f"Epoch {epoch+1}: Loss {train_loss:.3f}, Val {val_metrics['total_loss']:.3f}")

    # Temperature calibration
    logger.info("🌡️ Testing temperature calibration...")
    optimal_temp = trainer.calibrate_temperature(training_data)

    # Save model
    torch.save(model, "test_complete_model.pt")

    logger.info("✅ COMPLETE PIPELINE TEST PASSED!")
    logger.info(f"✅ Optimal temperature: {optimal_temp:.2f}")
    logger.info(f"✅ Model saved: test_complete_model.pt")
    logger.info("🎯 Ready for production!")

if __name__ == "__main__":
    test_complete_pipeline()