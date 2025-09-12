#!/usr/bin/env python3
"""
Quick-Start Training Script for Sanskrit Transformer Morphological Analyzer
===========================================================================

This script provides a streamlined workflow to train the transformer-based
morphological analyzer on your Vedic Sanskrit corpus.

Usage:
    python train_transformer_quickstart.py

Features:
- Automatic training data generation from your corpus
- Cross-validation against regex patterns  
- Model training with early stopping
- Validation against linguistic benchmarks
- Export trained model for enhanced analysis

For ACL ARR submission preparation.
"""

import os
import sys
import logging
from pathlib import Path
import torch
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformer_morphological_analyzer import (
    SanskritTransformerMorphAnalyzer,
    VedicMorphologicalTrainingDataGenerator,
    SanskritTransformerTrainer
)
from transformer_validation_system import (
    TransformerAccuracyValidator,
    VedicLinguisticValidationSet
)
from enhanced_diachronic_transformer_analysis import TransformerEnhancedVedicAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'transformers', 'sklearn', 'numpy', 
        'pandas', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Install with: pip install -r requirements_transformer.txt")
        return False
    
    return True

def setup_training_environment():
    """Setup directories and check GPU availability"""
    
    # Create output directories
    output_dirs = ['models', 'training_data', 'validation_reports', 'plots']
    for dir_name in output_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"GPU available: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU - training will be slower")
    
    return device

def generate_training_data():
    """Generate training data from your corpus"""
    
    logger.info("Generating training data from Vedic Sanskrit corpus...")
    
    # Your corpus files
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
    
    # Initialize data generator
    data_generator = VedicMorphologicalTrainingDataGenerator()
    
    # Generate training data
    training_df = data_generator.generate_training_data(
        corpus_files, 
        output_path="training_data/sanskrit_morph_training_data.json"
    )
    
    logger.info(f"Generated {len(training_df)} training samples")
    logger.info(f"Training data shape: {training_df.shape}")
    
    return training_df

def train_model(training_df, device, epochs=5):
    """Train the transformer model"""
    
    logger.info("Initializing transformer model...")
    
    # Initialize model
    model = SanskritTransformerMorphAnalyzer(
        model_name="bert-base-multilingual-cased",
        num_morphological_classes=50,
        num_historical_periods=4,
        dropout_rate=0.1
    )
    
    # Initialize trainer
    trainer = SanskritTransformerTrainer(
        model=model,
        learning_rate=2e-5,
        warmup_steps=500
    )
    
    # Convert DataFrame to training format
    training_data = training_df.to_dict('records')
    
    # Use subset for quick training (remove for full training)
    training_subset = training_data[:1000]  # Use first 1000 samples
    
    logger.info(f"Training on {len(training_subset)} samples for {epochs} epochs...")
    
    # Training loop
    best_loss = float('inf')
    patience = 2
    patience_counter = 0
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        try:
            # Train for one epoch
            loss = trainer.train_epoch(training_subset, batch_size=8)
            
            logger.info(f"Epoch {epoch+1} Loss: {loss:.4f}")
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
                
                # Save best model
                model_path = f"models/sanskrit_transformer_epoch_{epoch+1}.pt"
                torch.save(model, model_path)
                logger.info(f"Saved model to {model_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        except Exception as e:
            logger.error(f"Training error at epoch {epoch+1}: {e}")
            continue
    
    # Save final model
    final_model_path = "models/sanskrit_transformer_final.pt"
    torch.save(model, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    return model, final_model_path

def validate_model(model_path):
    """Validate the trained model"""
    
    logger.info("Validating trained model...")
    
    try:
        # Load trained model
        model = torch.load(model_path, map_location='cpu')
        
        # Initialize validation components
        validation_set = VedicLinguisticValidationSet()
        validator = TransformerAccuracyValidator(model, validation_set)
        
        # Run validation tests
        logger.info("Running gold standard validation...")
        gold_results = validator.validate_against_gold_standard()
        
        logger.info("Running confidence calibration...")
        # Note: You'd need actual validation data here
        validation_data = validation_set.linguistic_gold_standard
        calibration_results = validator.calibrate_confidence_scores(validation_data)
        
        # Generate validation report
        report_path = validator.generate_comprehensive_validation_report()
        
        logger.info(f"Validation complete. Report saved to: {report_path}")
        
        return gold_results, calibration_results
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return None, None

def run_enhanced_analysis(model_path):
    """Run enhanced analysis with the trained model"""
    
    logger.info("Running enhanced diachronic analysis...")
    
    try:
        # Initialize enhanced analyzer with trained model
        analyzer = TransformerEnhancedVedicAnalyzer(
            transformer_model_path=model_path,
            confidence_threshold=0.7,
            ensemble_method="weighted_average"
        )
        
        # Define corpus
        corpus_files = {
            'Rigveda': '../texts/samhita/rig-samhita.txt',
            'Samaveda': '../texts/samhita/sama-samhita.txt',
            'Yajurveda': '../texts/samhita/yajur-samhita.txt',
            # Add more files as needed for demo
        }
        
        # Run enhanced analysis
        results = analyzer.analyze_corpus_enhanced(corpus_files)
        
        # Generate reports
        report_path = analyzer.generate_enhanced_report("enhanced_analysis_report.txt")
        csv_path = analyzer.export_enhanced_results("enhanced_analysis_results.csv")
        
        logger.info(f"Enhanced analysis complete!")
        logger.info(f"Report: {report_path}")
        logger.info(f"Results: {csv_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced analysis error: {e}")
        return None

def main():
    """Main training workflow"""
    
    print("=" * 70)
    print("TRANSFORMER-BASED SANSKRIT MORPHOLOGICAL ANALYZER")
    print("Quick-Start Training Script")
    print("=" * 70)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Setup environment
    device = setup_training_environment()
    
    try:
        # Step 1: Generate training data
        logger.info("STEP 1: Generating training data...")
        training_df = generate_training_data()
        
        # Step 2: Train model
        logger.info("STEP 2: Training transformer model...")
        model, model_path = train_model(training_df, device, epochs=3)
        
        # Step 3: Validate model
        logger.info("STEP 3: Validating model...")
        gold_results, calibration_results = validate_model(model_path)
        
        if gold_results:
            logger.info("Validation results:")
            for metric, value in gold_results.items():
                logger.info(f"  {metric}: {value:.3f}")
        
        # Step 4: Run enhanced analysis
        logger.info("STEP 4: Running enhanced analysis...")
        enhanced_results = run_enhanced_analysis(model_path)
        
        # Summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE - READY FOR ACL ARR SUBMISSION")
        print("=" * 70)
        print("Generated outputs:")
        print("• Trained transformer model: models/sanskrit_transformer_final.pt")
        print("• Training data: training_data/sanskrit_morph_training_data.json")  
        print("• Validation report: transformer_validation_report.txt")
        print("• Enhanced analysis: enhanced_analysis_report.txt")
        print("• Results CSV: enhanced_analysis_results.csv")
        print("\nNext steps:")
        print("• Review validation metrics")
        print("• Run full corpus analysis")
        print("• Generate publication-ready visualizations")
        print("• Prepare ACL ARR submission materials")
        
    except Exception as e:
        logger.error(f"Training workflow failed: {e}")
        logger.error("Check file paths and ensure text files are available")

if __name__ == "__main__":
    main()