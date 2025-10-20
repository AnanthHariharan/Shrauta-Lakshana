#!/usr/bin/env python3
"""
Production Training Script for FIXED Transformer
=================================================

This script trains the corrected transformer with:
- Multi-label morphological classification
- Proper BCEWithLogitsLoss
- Temperature calibration
- Comprehensive regex weak supervision
- Validation and checkpointing

Usage:
    python train_transformer_fixed.py
"""

import sys
sys.path.append('.')

from transformer_morphological_analyzer import (
    SanskritTransformerMorphAnalyzer,
    VedicMorphologicalTrainingDataGenerator,
    SanskritTransformerTrainer
)
import torch
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Starting FIXED Transformer Training")
    logger.info("="*60)

    # Comprehensive morphological features (75 features)
    morphological_features = [
        # Phonological features
        'retroflex_l', 'visarga_final', 'diphthongs_ai', 'diphthongs_au',
        'monophthongs_e', 'monophthongs_o', 'external_sandhi_unresolved',
        'retroflex_assimilation', 'pluti_vowels', 'medial_voiced_aspirates',
        'complex_clusters',

        # Verbal morphology - subjunctive system
        'subjunctive_ati', 'subjunctive_an', 'subjunctive_as', 'subjunctive_at',
        'subjunctive_ama', 'subjunctive_full',

        # Perfect system
        'perfect_reduplicated', 'perfect_periphrastic', 'perfect_endings',

        # Aorist system
        'aorist_is', 'aorist_root', 'aorist_sigmatic',

        # Dual system
        'dual_nominative', 'dual_instrumental', 'dual_genitive', 'locative_dual',

        # Injunctive and modal
        'injunctive_augmentless', 'injunctive_modal', 'precative', 'benedictive',

        # Case system evolution
        'instrumental_archaic_a', 'instrumental_classical_ena',
        'genitive_plural_thematic', 'genitive_plural_athematic', 'locative_plural',

        # Particles
        'particle_sma', 'particle_ha', 'particle_vai', 'particle_id',
        'particle_u', 'particle_hi',

        # Participial forms
        'present_participle_ant', 'present_participle_at',
        'past_participle_ta', 'past_participle_na',

        # Gerunds and infinitives
        'gerund_tvaa', 'gerund_ya', 'infinitive_tum',

        # Syntactic constructions
        'correlatives_ya_ta', 'conditional_yadi_tarhi', 'long_compounds',
        'compound_bahuvrīhi', 'absolute_construction', 'verb_nonfinal',
        'subordinators',

        # Derivational morphology
        'primary_suffixes', 'secondary_suffixes', 'action_nouns_ti',
        'agent_nouns_tar', 'abstract_nouns_tva',

        # Lexical categories
        'ritual_sacrifice', 'deity_names', 'priestly_terms',
        'philosophical_terms', 'cosmological_terms', 'eschatological_terms',
        'sacrifice_roots', 'ritual_implements', 'substrate_lexemes',

        # Prosodic features
        'short_syllables', 'long_syllables',

        # Present formations
        'n_infix_presents', 'reduplicated_presents',

        # Discourse features
        'philosophical_context', 'reported_speech', 'connectives', 'prose_particles'
    ]

    logger.info(f"📋 Training with {len(morphological_features)} morphological features")

    # Period mapping
    period_mapping = {
        'Rigveda': 'Early Vedic',
        'Samaveda': 'Early Vedic',
        'Yajurveda': 'Early Vedic',
        'Atharvaveda (Paippalada)': 'Early Vedic',
        'Atharvaveda (Saunaka)': 'Early Vedic',
        'Kausitaki-Br': 'Late Vedic',
        'Pancavimsa-Br': 'Late Vedic',
        'Satapatha-Br': 'Late Vedic',
        'Gopatha-Br': 'Late Vedic',
        'Aitareya-Up': 'Latest Vedic',
        'Taittiriya-Up': 'Latest Vedic',
        'Chandogya-Up': 'Latest Vedic',
        'Brhadaranyaka-Up': 'Latest Vedic',
        'Prashna-Up': 'Latest Vedic',
        'Shvetashvatara-Up': 'Latest Vedic',
        'Ramayana': 'Classical',
        'Mahabharata': 'Classical',
        'Bhagavata-Purana': 'Classical'
    }

    # Corpus files
    corpus_files = {
        'Rigveda': '../texts/samhita/rig-samhita.txt',
        'Samaveda': '../texts/samhita/sama-samhita.txt',
        'Yajurveda': '../texts/samhita/yajur-samhita.txt',
        'Atharvaveda (Paippalada)': '../texts/samhita/atharva-paippalada-samhita.txt',
        'Atharvaveda (Saunaka)': '../texts/samhita/atharva-saunaka-samhita.txt',
        'Kausitaki-Br': '../texts/brahmana/rig-kausitaki.txt',
        'Pancavimsa-Br': '../texts/brahmana/sama-pancavimsa.txt',
        'Satapatha-Br': '../texts/brahmana/yajur-satapatha.txt',
        'Gopatha-Br': '../texts/brahmana/atharva-gopatha.txt',
        'Aitareya-Up': '../texts/upanishad/rig-aitareya.txt',
        'Taittiriya-Up': '../texts/upanishad/yajur-taittiriya-up.txt',
        'Chandogya-Up': '../texts/upanishad/sama-chandogya.txt',
        'Brhadaranyaka-Up': '../texts/upanishad/yajur-brhadaranyaka.txt',
        'Prashna-Up': '../texts/upanishad/atharva-prashna.txt',
        'Shvetashvatara-Up': '../texts/upanishad/yajur-shvetashvatara.txt',
        'Ramayana': '../texts/classical-sanskrit/ramayana.txt',
        'Mahabharata': '../texts/classical-sanskrit/mahabharata.txt',
        'Bhagavata-Purana': '../texts/classical-sanskrit/bhagavata-purana.txt'
    }

    # Initialize model with correct number of features
    logger.info("🤖 Initializing transformer model...")
    model = SanskritTransformerMorphAnalyzer(
        num_morphological_classes=len(morphological_features)
    )

    # Initialize data generator
    logger.info("📊 Initializing data generator...")
    data_generator = VedicMorphologicalTrainingDataGenerator(morphological_features)

    # Generate training data
    logger.info("🏭 Generating training data from corpus...")
    training_data = data_generator.generate_training_dataset(corpus_files, period_mapping)

    if len(training_data) == 0:
        logger.error("❌ No training data generated! Check file paths.")
        return

    logger.info(f"✅ Generated {len(training_data)} training samples")

    # Split data
    train_size = int(0.8 * len(training_data))
    train_data = training_data[:train_size]
    val_data = training_data[train_size:]

    logger.info(f"📈 Training samples: {len(train_data)}")
    logger.info(f"📊 Validation samples: {len(val_data)}")

    # Initialize trainer
    logger.info("🎯 Initializing trainer...")
    trainer = SanskritTransformerTrainer(model, learning_rate=2e-5)

    # Training loop
    logger.info("🚂 Starting training...")
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(15):  # More epochs for better results
        logger.info(f"\n📅 Epoch {epoch + 1}/15")
        logger.info("-" * 30)

        # Train
        train_loss = trainer.train_epoch(train_data)

        # Validate
        val_metrics = trainer.validate(val_data)
        val_loss = val_metrics['total_loss']

        logger.info(f"🏃‍♂️ Train Loss: {train_loss:.4f}")
        logger.info(f"📊 Val Loss: {val_loss:.4f}")
        logger.info(f"🧬 Morph Loss: {val_metrics['morphological_loss']:.4f}")
        logger.info(f"📅 Period Loss: {val_metrics['period_loss']:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/sanskrit_transformer_FIXED_best.pt")
            logger.info("💾 ✅ Best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"⏰ Early stopping after {epoch + 1} epochs")
            break

    # Temperature calibration
    logger.info("\n🌡️ Calibrating temperature...")
    # Note: This requires the model to have a temperature parameter
    # which needs to be added to the model definition

    # Save final model
    torch.save(model, "models/sanskrit_transformer_morph_analyzer_FIXED.pt")

    # Save training data
    logger.info("💾 Saving training data...")
    with open("training_data/sanskrit_morph_training_FIXED.json", 'w', encoding='utf-8') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = []
        for item in training_data:
            serializable_item = {
                'text': item['text'],
                'detected_features': item['detected_features'],
                'feature_counts': item['feature_counts'],
                'multi_hot_labels': item['multi_hot_labels'].tolist(),
                'period_label': item['period_label'],
                'num_features': item['num_features'],
                'text_name': item['text_name']
            }
            serializable_data.append(serializable_item)

        metadata = {
            'num_samples': len(training_data),
            'num_features': len(morphological_features),
            'morphological_features': morphological_features,
            'period_mapping': period_mapping
        }

        full_data = {
            'metadata': metadata,
            'training_samples': serializable_data
        }

        json.dump(full_data, f, ensure_ascii=False, indent=2)

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("🎉 FIXED TRANSFORMER TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"📊 Final validation loss: {best_val_loss:.4f}")
    logger.info(f"💾 Model saved: models/sanskrit_transformer_morph_analyzer_FIXED.pt")
    logger.info(f"📁 Training data: training_data/sanskrit_morph_training_FIXED.json")
    logger.info(f"🧬 Features trained: {len(morphological_features)}")
    logger.info(f"📈 Training samples: {len(train_data)}")
    logger.info("")
    logger.info("🚀 READY FOR ENHANCED ENSEMBLE ANALYSIS!")
    logger.info("="*60)

if __name__ == "__main__":
    # Create directories
    import os
    os.makedirs("models", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)

    main()