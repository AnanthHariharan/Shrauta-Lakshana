#!/usr/bin/env python3
"""
Efficient Transformer Training Script
====================================

Optimized version with:
- Limited training samples per text for speed
- Early stopping for efficient training
- Progress tracking
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
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    start_time = time.time()
    logger.info("🚀 EFFICIENT Transformer Training")
    logger.info("="*50)

    # Comprehensive morphological features (78 features)
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

    logger.info(f"Training with {len(morphological_features)} morphological features")

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

    # Initialize model
    logger.info("🤖 Initializing model...")
    model = SanskritTransformerMorphAnalyzer(
        num_morphological_classes=len(morphological_features)
    )

    # Initialize data generator with limited samples
    logger.info("📊 Generating training data...")

    class EfficientDataGenerator(VedicMorphologicalTrainingDataGenerator):
        def generate_training_dataset(self, corpus_files, period_mapping, max_samples_per_text=200):
            """Generate training dataset with limited samples per text"""
            training_data = []

            for text_name, filepath in corpus_files.items():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()

                    period = period_mapping.get(text_name, 'Unknown')

                    # Split into sentences
                    sentences = text.split('।')  # Sanskrit sentence separator
                    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]

                    # Limit samples per text for efficiency
                    sample_count = 0
                    for sentence in sentences:
                        if sample_count >= max_samples_per_text:
                            break

                        training_sample = self.analyze_text_for_training(sentence, period)
                        if training_sample['num_features'] > 0:  # Only include samples with features
                            training_sample['text_name'] = text_name
                            training_data.append(training_sample)
                            sample_count += 1

                    logger.info(f"Generated {sample_count} samples from {text_name}")

                except FileNotFoundError:
                    logger.warning(f"File not found: {filepath}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing {text_name}: {e}")
                    continue

            logger.info(f"Total training samples: {len(training_data)}")
            return training_data

    data_generator = EfficientDataGenerator(morphological_features)
    training_data = data_generator.generate_training_dataset(corpus_files, period_mapping, max_samples_per_text=100)

    if len(training_data) == 0:
        logger.error("❌ No training data generated!")
        return

    # Split data
    train_size = int(0.8 * len(training_data))
    train_data = training_data[:train_size]
    val_data = training_data[train_size:]

    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")

    # Initialize trainer
    trainer = SanskritTransformerTrainer(model, learning_rate=3e-5)

    # Quick training loop (5 epochs)
    logger.info("🏃‍♂️ Starting efficient training...")
    best_val_loss = float('inf')

    for epoch in range(5):
        epoch_start = time.time()

        # Train
        train_loss = trainer.train_epoch(train_data)

        # Validate
        val_metrics = trainer.validate(val_data)
        val_loss = val_metrics['total_loss']

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch + 1}/5 ({epoch_time:.1f}s)")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Morph Loss: {val_metrics['morphological_loss']:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "sanskrit_transformer_efficient_best.pt")
            logger.info("  ✅ Best model saved!")

    # Save final model
    torch.save(model, "sanskrit_transformer_efficient_final.pt")

    total_time = time.time() - start_time
    logger.info("\n" + "="*50)
    logger.info("🎉 EFFICIENT TRAINING COMPLETE!")
    logger.info(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    logger.info(f"📊 Final validation loss: {best_val_loss:.4f}")
    logger.info(f"💾 Models saved:")
    logger.info(f"   - sanskrit_transformer_efficient_best.pt")
    logger.info(f"   - sanskrit_transformer_efficient_final.pt")
    logger.info(f"🧬 Features: {len(morphological_features)}")
    logger.info(f"📈 Training samples: {len(train_data)}")
    logger.info("🚀 READY FOR ENSEMBLE!")
    logger.info("="*50)

if __name__ == "__main__":
    main()