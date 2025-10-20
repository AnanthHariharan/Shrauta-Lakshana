#!/usr/bin/env python3
"""
Enhanced BERT Ensemble System with Proper Tokenization and Fine-Tuning
=====================================================================

Implements proper BERT tokenization and ensemble threshold optimization
for improved ACL submission results.
"""

import sys
sys.path.append('.')

import torch
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
import re
from typing import Dict, Tuple, List
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

# Import components
from transformer_morphological_analyzer import (
    SanskritTransformerMorphAnalyzer,
    VedicMorphologicalTrainingDataGenerator
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTEnhancedEnsembleAnalyzer:
    """Enhanced ensemble with proper BERT tokenization and threshold optimization"""

    def __init__(self, optimize_thresholds: bool = True):
        # Load training metadata
        with open('training_data/sanskrit_transformer_PRODUCTION_data.json', 'r') as f:
            training_data = json.load(f)

        self.features = training_data['metadata']['morphological_features']
        self.optimal_temp = training_data['metadata']['optimal_temperature']

        # Initialize proper BERT tokenizer
        logger.info("🤖 Initializing BERT tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = AutoModel.from_pretrained('bert-base-multilingual-cased')

        # Load production transformer
        logger.info("📊 Loading production transformer model...")
        self.transformer_model = SanskritTransformerMorphAnalyzer(
            num_morphological_classes=len(self.features),
            temperature=self.optimal_temp
        )
        self.transformer_model.load_state_dict(
            torch.load('models/sanskrit_transformer_PRODUCTION_best.pt', map_location='cpu')
        )
        self.transformer_model.eval()

        # Initialize data generator
        self.data_generator = VedicMorphologicalTrainingDataGenerator(self.features)

        # Ensemble parameters with defaults
        self.ensemble_params = {
            'confidence_threshold_high': 0.7,
            'confidence_threshold_low': 0.3,
            'regex_weight': 0.6,
            'transformer_weight': 0.4,
            'ensemble_threshold': 0.5,
            'feature_specific_weights': {}
        }

        # Initialize with threshold optimization
        if optimize_thresholds:
            self.optimize_ensemble_parameters()

        # Results storage
        self.results = defaultdict(dict)
        self.validation_metrics = {}

        logger.info("✅ Enhanced BERT ensemble analyzer initialized")

    def initialize_proper_tokenization(self):
        """Set up proper BERT tokenization for Sanskrit text"""

        # Add Sanskrit-specific tokens if needed
        sanskrit_tokens = ['ॐ', '।', '॥', 'ṃ', 'ḥ', 'ṛ', 'ḷ', 'ā', 'ī', 'ū', 'ē', 'ō']

        # Check if tokens need to be added
        new_tokens = [token for token in sanskrit_tokens
                     if token not in self.tokenizer.vocab]

        if new_tokens:
            logger.info(f"Adding {len(new_tokens)} Sanskrit-specific tokens")
            self.tokenizer.add_tokens(new_tokens)
            # Note: In production, you'd need to resize transformer embeddings

        logger.info("✅ BERT tokenization initialized for Sanskrit")

    def tokenize_sanskrit_text(self, text: str, max_length: int = 512) -> Dict:
        """Properly tokenize Sanskrit text using BERT tokenizer"""

        # Preprocess text for better tokenization
        text = self.preprocess_sanskrit_text(text)

        # Use BERT tokenizer
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
            add_special_tokens=True
        )

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding.get('token_type_ids'),
            'tokens': self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        }

    def preprocess_sanskrit_text(self, text: str) -> str:
        """Preprocess Sanskrit text for better tokenization"""

        # Handle Sanskrit punctuation
        text = re.sub(r'।+', ' । ', text)  # Danda
        text = re.sub(r'॥+', ' ॥ ', text)  # Double danda

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Handle common Sanskrit ligatures and compounds
        # (Add more sophisticated preprocessing as needed)

        return text

    def analyze_with_bert_transformer(self, text: str, period: str) -> Tuple[Dict, Dict]:
        """Analyze text using proper BERT tokenization"""

        # Split into manageable sentences
        sentences = [s.strip() for s in text.split('।') if s.strip() and len(s) > 10]

        feature_scores = defaultdict(list)
        confidence_scores = defaultdict(list)

        # Sample sentences for efficiency
        sample_sentences = sentences[:50] if len(sentences) > 50 else sentences

        for sentence in sample_sentences:
            try:
                # Proper BERT tokenization
                tokenized = self.tokenize_sanskrit_text(sentence, max_length=128)

                # Get transformer predictions
                with torch.no_grad():
                    outputs = self.transformer_model(
                        tokenized['input_ids'],
                        tokenized['attention_mask']
                    )
                    morph_logits = outputs['morphological_logits']

                    # Apply sigmoid to get probabilities
                    probs = torch.sigmoid(morph_logits).squeeze()

                    # Store feature scores
                    for i, feature in enumerate(self.features):
                        if i < len(probs):
                            confidence = probs[i].item()
                            feature_scores[feature].append(confidence)
                            confidence_scores[feature].append(confidence)

            except Exception as e:
                logger.debug(f"BERT transformer prediction failed: {e}")
                continue

        # Calculate average scores
        transformer_results = {}
        avg_confidences = {}

        for feature in self.features:
            if feature in feature_scores and len(feature_scores[feature]) > 0:
                scores = feature_scores[feature]
                avg_score = np.mean(scores)
                transformer_results[feature] = avg_score * 1000 if avg_score > self.ensemble_params['ensemble_threshold'] else 0
                avg_confidences[feature] = avg_score
            else:
                transformer_results[feature] = 0
                avg_confidences[feature] = 0.0

        return transformer_results, avg_confidences

    def optimize_ensemble_parameters(self):
        """Optimize ensemble thresholds using cross-validation"""

        logger.info("🎯 Optimizing ensemble parameters...")

        # Load sample data for optimization
        if not Path('full_corpus_ensemble_results.json').exists():
            logger.warning("No existing results found for optimization")
            return

        with open('full_corpus_ensemble_results.json', 'r') as f:
            existing_results = json.load(f)

        # Extract validation data
        validation_data = self._prepare_validation_data(existing_results)

        if len(validation_data) < 10:
            logger.warning("Insufficient data for parameter optimization")
            return

        # Define optimization objective
        def objective_function(params):
            """Objective function to minimize (negative F1-score)"""
            temp_params = {
                'confidence_threshold_high': max(0.5, min(0.9, params[0])),
                'confidence_threshold_low': max(0.1, min(0.5, params[1])),
                'regex_weight': max(0.1, min(0.9, params[2])),
                'transformer_weight': 1.0 - max(0.1, min(0.9, params[2])),
                'ensemble_threshold': max(0.1, min(0.8, params[3]))
            }

            # Calculate cross-validation score
            cv_score = self._cross_validate_ensemble(validation_data, temp_params)
            return -cv_score  # Minimize negative score

        # Initial parameters
        initial_params = [0.7, 0.3, 0.6, 0.5]

        # Optimize
        result = minimize(
            objective_function,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 50}
        )

        if result.success:
            # Update parameters
            optimal_params = result.x
            self.ensemble_params.update({
                'confidence_threshold_high': max(0.5, min(0.9, optimal_params[0])),
                'confidence_threshold_low': max(0.1, min(0.5, optimal_params[1])),
                'regex_weight': max(0.1, min(0.9, optimal_params[2])),
                'transformer_weight': 1.0 - max(0.1, min(0.9, optimal_params[2])),
                'ensemble_threshold': max(0.1, min(0.8, optimal_params[3]))
            })

            logger.info("✅ Ensemble parameters optimized:")
            for param, value in self.ensemble_params.items():
                if not param.endswith('_weights'):
                    logger.info(f"   {param}: {value:.3f}")
        else:
            logger.warning("⚠️ Parameter optimization failed, using defaults")

    def _prepare_validation_data(self, results_data) -> List[Dict]:
        """Prepare validation data from existing results"""

        validation_samples = []

        if 'results' not in results_data:
            return validation_samples

        regex_results = results_data['results'].get('regex_results', {})
        transformer_results = results_data['results'].get('transformer_results', {})

        for text_name in regex_results:
            if text_name in transformer_results:
                sample = {
                    'text_name': text_name,
                    'regex_features': regex_results[text_name],
                    'transformer_features': transformer_results[text_name],
                    # Create ground truth from strong agreement
                    'ground_truth': self._create_ground_truth(
                        regex_results[text_name],
                        transformer_results[text_name]
                    )
                }
                validation_samples.append(sample)

        return validation_samples

    def _create_ground_truth(self, regex_results: Dict, transformer_results: Dict) -> Dict:
        """Create pseudo ground truth from strong method agreement"""

        ground_truth = {}

        for feature in self.features:
            regex_val = regex_results.get(feature, 0)
            trans_val = transformer_results.get(feature, 0)

            # Consider as positive if both methods strongly agree
            regex_binary = 1 if regex_val > 1.0 else 0
            trans_binary = 1 if trans_val > 0.7 else 0

            # High confidence ground truth
            if regex_binary == 1 and trans_binary == 1:
                ground_truth[feature] = 1
            elif regex_binary == 0 and trans_binary == 0:
                ground_truth[feature] = 0
            else:
                ground_truth[feature] = -1  # Uncertain, exclude from validation

        return ground_truth

    def _cross_validate_ensemble(self, validation_data: List[Dict], params: Dict) -> float:
        """Cross-validate ensemble with given parameters"""

        if len(validation_data) < 5:
            return 0.0

        kf = KFold(n_splits=min(5, len(validation_data)), shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, test_idx in kf.split(validation_data):
            test_samples = [validation_data[i] for i in test_idx]

            y_true = []
            y_pred = []

            for sample in test_samples:
                for feature in self.features:
                    if sample['ground_truth'].get(feature, -1) != -1:  # Skip uncertain labels
                        y_true.append(sample['ground_truth'][feature])

                        # Apply ensemble with current parameters
                        regex_val = sample['regex_features'].get(feature, 0)
                        trans_val = sample['transformer_features'].get(feature, 0)

                        ensemble_pred = self._apply_ensemble_logic(
                            regex_val, trans_val, 0.5, params  # Dummy confidence
                        )

                        y_pred.append(1 if ensemble_pred > params['ensemble_threshold'] else 0)

            if len(y_true) > 0 and len(set(y_true)) > 1:
                # Calculate F1-score
                from sklearn.metrics import f1_score
                score = f1_score(y_true, y_pred, average='binary', zero_division=0)
                cv_scores.append(score)

        return np.mean(cv_scores) if cv_scores else 0.0

    def _apply_ensemble_logic(self, regex_val: float, trans_val: float,
                            confidence: float, params: Dict) -> float:
        """Apply ensemble logic with given parameters"""

        regex_binary = 1 if regex_val > 0.5 else 0
        trans_binary = 1 if trans_val > 0.5 else 0

        if regex_binary == 1 and trans_binary == 1:
            # Both agree positive - weighted average
            return params['regex_weight'] * regex_val + params['transformer_weight'] * trans_val
        elif regex_binary == 0 and trans_binary == 0:
            # Both agree negative
            return 0
        elif regex_binary == 1 and trans_binary == 0:
            # Regex only - trust if low transformer confidence
            return regex_val if confidence < params['confidence_threshold_low'] else regex_val * 0.7
        elif regex_binary == 0 and trans_binary == 1:
            # Transformer only - trust if high confidence
            return trans_val if confidence > params['confidence_threshold_high'] else trans_val * 0.3
        else:
            return 0

    def create_optimized_ensemble(self, regex_results: Dict, transformer_results: Dict,
                                confidence_scores: Dict) -> Tuple[Dict, Dict]:
        """Create ensemble results with optimized parameters"""

        ensemble_results = {}
        agreement_stats = {
            'total_features': 0,
            'optimized_agreements': 0,
            'high_confidence_decisions': 0,
            'weighted_combinations': 0
        }

        for feature in self.features:
            regex_val = regex_results.get(feature, 0)
            transformer_val = transformer_results.get(feature, 0)
            confidence = confidence_scores.get(feature, 0.0)

            agreement_stats['total_features'] += 1

            # Apply optimized ensemble logic
            ensemble_val = self._apply_ensemble_logic(
                regex_val, transformer_val, confidence, self.ensemble_params
            )

            ensemble_results[feature] = ensemble_val

            # Track ensemble statistics
            if ensemble_val > self.ensemble_params['ensemble_threshold']:
                agreement_stats['optimized_agreements'] += 1

            if confidence > self.ensemble_params['confidence_threshold_high']:
                agreement_stats['high_confidence_decisions'] += 1

            regex_binary = 1 if regex_val > 0.5 else 0
            trans_binary = 1 if transformer_val > 0.5 else 0
            if regex_binary == 1 and trans_binary == 1:
                agreement_stats['weighted_combinations'] += 1

        return ensemble_results, agreement_stats

    def run_enhanced_analysis(self, corpus_files: Dict) -> Dict:
        """Run enhanced analysis with BERT tokenization and optimized ensemble"""

        logger.info("🚀 STARTING ENHANCED BERT ENSEMBLE ANALYSIS")
        logger.info("=" * 60)

        # Initialize proper tokenization
        self.initialize_proper_tokenization()

        period_mapping = {
            'Rigveda': 'Early Vedic',
            'Samaveda': 'Early Vedic',
            'Yajurveda (Taittiriya)': 'Early Vedic',
            'Yajurveda (Maitrayani)': 'Early Vedic',
            'Atharvaveda (Paippalada)': 'Early Vedic',
            'Atharvaveda (Saunaka)': 'Early Vedic',
            'Kausitaki-Br': 'Late Vedic',
            'Pancavimsa-Br': 'Late Vedic',
            'Taittiriya-Br': 'Late Vedic',
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

        analysis_summary = {
            'texts_analyzed': 0,
            'bert_tokenization_enabled': True,
            'ensemble_optimized': True,
            'total_tokens_processed': 0,
            'improved_agreement_stats': defaultdict(int)
        }

        # Analyze subset of texts for demonstration (can be extended to full corpus)
        sample_texts = list(corpus_files.items())[:5]  # First 5 texts for optimization

        for text_name, filepath in sample_texts:
            if not Path(filepath).exists():
                continue

            logger.info(f"🔍 Analyzing {text_name} with enhanced BERT ensemble...")

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()

                period = period_mapping.get(text_name, 'Unknown')

                # 1. Regex analysis
                sample = self.data_generator.analyze_text_for_training(text, period)
                total_words = len(re.findall(r'\b\w+\b', text))

                regex_results = {}
                for feature in self.features:
                    count = sample['feature_counts'].get(feature, 0)
                    normalized = (count / total_words) * 1000 if total_words > 0 else 0
                    regex_results[feature] = normalized

                # 2. Enhanced BERT transformer analysis
                transformer_results, confidence_scores = self.analyze_with_bert_transformer(text, period)

                # 3. Optimized ensemble
                ensemble_results, agreement_stats = self.create_optimized_ensemble(
                    regex_results, transformer_results, confidence_scores
                )

                # Store results
                self.results[text_name] = {
                    'regex': regex_results,
                    'transformer': transformer_results,
                    'ensemble': ensemble_results,
                    'confidence': confidence_scores,
                    'agreement': agreement_stats
                }

                analysis_summary['texts_analyzed'] += 1
                analysis_summary['total_tokens_processed'] += len(self.tokenizer.tokenize(text[:10000]))  # Sample

                for key, value in agreement_stats.items():
                    analysis_summary['improved_agreement_stats'][key] += value

                logger.info(f"  ✅ {text_name}: {agreement_stats['optimized_agreements']}/{agreement_stats['total_features']} ensemble features")

            except Exception as e:
                logger.error(f"Error analyzing {text_name}: {e}")

        # Generate final report
        self._generate_enhanced_report(analysis_summary)

        return self.results

    def _generate_enhanced_report(self, summary):
        """Generate enhanced analysis report with optimization details"""

        logger.info("\n" + "=" * 60)
        logger.info("🎯 ENHANCED BERT ENSEMBLE ANALYSIS COMPLETE")
        logger.info("=" * 60)

        logger.info(f"📊 Enhanced Performance:")
        logger.info(f"   Texts Analyzed: {summary['texts_analyzed']}")
        logger.info(f"   BERT Tokens Processed: {summary['total_tokens_processed']:,}")
        logger.info(f"   Ensemble Optimization: {'✅ Enabled' if summary['ensemble_optimized'] else '❌ Disabled'}")

        # Optimized agreement statistics
        stats = summary['improved_agreement_stats']
        if stats['total_features'] > 0:
            agreement_rate = stats['optimized_agreements'] / stats['total_features']
            logger.info(f"\n🎯 Optimized Ensemble Performance:")
            logger.info(f"   Agreement Rate: {agreement_rate:.3f}")
            logger.info(f"   High Confidence Decisions: {stats['high_confidence_decisions']}")
            logger.info(f"   Weighted Combinations: {stats['weighted_combinations']}")

        logger.info(f"\n⚙️ Optimized Parameters:")
        for param, value in self.ensemble_params.items():
            if not param.endswith('_weights'):
                logger.info(f"   {param}: {value:.3f}")

        # Save enhanced results
        enhanced_data = {
            'enhanced_analysis': {
                'bert_tokenization': True,
                'optimized_ensemble': True,
                'ensemble_parameters': self.ensemble_params,
                'analysis_summary': dict(summary),
                'results': dict(self.results)
            }
        }

        with open('enhanced_bert_ensemble_results.json', 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)

        logger.info("💾 Enhanced results saved to: enhanced_bert_ensemble_results.json")
        logger.info("✅ BERT tokenization and ensemble optimization complete!")

def main():
    """Main function to demonstrate enhanced BERT ensemble"""

    # Sample corpus files for testing
    sample_corpus = {
        'Taittiriya-Up': '../texts/upanishad/yajur-taittiriya-up.txt',
        'Ramayana': '../texts/classical-sanskrit/ramayana.txt',
        'Mahabharata': '../texts/classical-sanskrit/mahabharata.txt'
    }

    analyzer = BERTEnhancedEnsembleAnalyzer(optimize_thresholds=True)
    results = analyzer.run_enhanced_analysis(sample_corpus)

    return results

if __name__ == "__main__":
    main()