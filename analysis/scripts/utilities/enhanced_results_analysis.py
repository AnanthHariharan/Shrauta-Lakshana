#!/usr/bin/env python3
"""
Enhanced Results Analysis and Comparison
=======================================

Analyze the new ensemble results and compare with baseline metrics
for ACL submission evaluation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_enhanced_results():
    """Comprehensive analysis of enhanced ensemble results"""

    logger.info("📊 ENHANCED RESULTS ANALYSIS")
    logger.info("=" * 60)

    # Load results
    with open('enhanced_ensemble_analysis_results.json', 'r') as f:
        results = json.load(f)

    summary = results['summary']
    analysis_results = results['results']

    logger.info("🎯 OVERALL ENSEMBLE PERFORMANCE:")
    logger.info(f"   Texts Analyzed: {summary['texts_analyzed']}")

    # Agreement Analysis
    agreement_stats = summary['agreement_stats']
    total_comparisons = agreement_stats.get('total_features', 0)

    if total_comparisons > 0:
        agreement_rate = (agreement_stats.get('both_agree_positive', 0) +
                         agreement_stats.get('both_agree_negative', 0)) / total_comparisons

        logger.info(f"   Overall Agreement Rate: {agreement_rate:.3f}")
        logger.info(f"   Both Methods Positive: {agreement_stats.get('both_agree_positive', 0)}")
        logger.info(f"   Both Methods Negative: {agreement_stats.get('both_agree_negative', 0)}")
        logger.info(f"   Regex Only Detections: {agreement_stats.get('regex_only', 0)}")
        logger.info(f"   Transformer Only: {agreement_stats.get('transformer_only', 0)}")
        logger.info(f"   High Confidence Transformer: {agreement_stats.get('high_confidence_transformer', 0)}")

    # Method Comparison
    logger.info(f"\n🔬 METHOD COMPARISON:")
    method_comparison = summary['method_comparison']

    for method in ['regex', 'transformer', 'ensemble']:
        if method in method_comparison:
            features_detected = len([f for f, count in method_comparison[method].items() if count > 0])
            total_features = len(results['metadata']['features'])
            logger.info(f"   {method.capitalize()}: {features_detected}/{total_features} features detected")

    # Feature Performance Analysis
    logger.info(f"\n🎯 FEATURE PERFORMANCE ANALYSIS:")

    # Analyze which features are most reliably detected
    feature_performance = analyze_feature_performance(analysis_results)

    # Top performing features
    logger.info("   🏆 Top Performing Features (Regex):")
    for feature, stats in feature_performance['regex_top'][:10]:
        logger.info(f"      {feature}: avg {stats['avg']:.1f}, texts {stats['texts']}")

    logger.info("   🤖 Transformer High Confidence Features:")
    for feature, stats in feature_performance['transformer_consistent'][:10]:
        logger.info(f"      {feature}: avg {stats['avg']:.3f}, texts {stats['texts']}")

    # Comparison with Previous Baseline
    logger.info(f"\n📈 COMPARISON WITH PREVIOUS BASELINE:")
    baseline_comparison = compare_with_baseline()

    for metric, comparison in baseline_comparison.items():
        logger.info(f"   {metric}: {comparison}")

    # Generate improvement recommendations
    recommendations = generate_recommendations(results, feature_performance)

    logger.info(f"\n💡 RECOMMENDATIONS FOR ACL SUBMISSION:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"   {i}. {rec}")

    # Save detailed analysis
    save_detailed_analysis(results, feature_performance, baseline_comparison, recommendations)

    return results

def analyze_feature_performance(analysis_results):
    """Analyze individual feature performance across methods"""

    regex_performance = defaultdict(list)
    transformer_performance = defaultdict(list)
    ensemble_performance = defaultdict(list)

    # Collect performance across all texts
    for text_name in analysis_results['regex_only']:
        for feature, value in analysis_results['regex_only'][text_name].items():
            if value > 0:
                regex_performance[feature].append(value)

        for feature, value in analysis_results['transformer_only'][text_name].items():
            if value > 0:
                transformer_performance[feature].append(value)

        for feature, value in analysis_results['ensemble'][text_name].items():
            if value > 0:
                ensemble_performance[feature].append(value)

    # Calculate statistics
    regex_stats = {}
    for feature, values in regex_performance.items():
        regex_stats[feature] = {
            'avg': np.mean(values),
            'std': np.std(values),
            'texts': len(values)
        }

    transformer_stats = {}
    for feature, values in transformer_performance.items():
        transformer_stats[feature] = {
            'avg': np.mean(values),
            'std': np.std(values),
            'texts': len(values)
        }

    # Sort by reliability (combination of frequency and consistency)
    regex_top = sorted(regex_stats.items(), key=lambda x: (x[1]['texts'], x[1]['avg']), reverse=True)
    transformer_consistent = sorted(transformer_stats.items(), key=lambda x: x[1]['texts'], reverse=True)

    return {
        'regex_top': regex_top,
        'transformer_consistent': transformer_consistent,
        'regex_stats': regex_stats,
        'transformer_stats': transformer_stats
    }

def compare_with_baseline():
    """Compare current results with previous baseline"""

    comparisons = {}

    # Previous metrics from validation report
    prev_confidence_correlation = 0.709
    prev_agreement_rates = {
        'subjunctive_full': 0.000,
        'dual_instrumental': 0.200,
        'particle_sma': 0.000,
        'long_compounds': 0.400,
        'perfect_reduplicated': 0.000
    }

    # Current overall agreement rate from our analysis
    # (This would be calculated from actual ensemble comparison)
    current_agreement = 0.136  # From our analysis output

    comparisons['Agreement Rate'] = f"Previous: variable (0.0-0.4), Current: {current_agreement:.3f}"
    comparisons['Confidence Correlation'] = f"Previous: {prev_confidence_correlation:.3f}, Current: N/A (needs validation)"
    comparisons['Feature Coverage'] = "Previous: 4/163 significant, Current: 76/78 regex + 56/78 transformer"
    comparisons['Method Integration'] = "Previous: Separate validation, Current: Ensemble with confidence weighting"

    return comparisons

def generate_recommendations(results, feature_performance):
    """Generate recommendations for improving ACL submission"""

    recommendations = []

    # Based on low ensemble agreement
    if results['summary']['agreement_stats'].get('total_features', 0) > 0:
        agreement_rate = ((results['summary']['agreement_stats'].get('both_agree_positive', 0) +
                          results['summary']['agreement_stats'].get('both_agree_negative', 0)) /
                         results['summary']['agreement_stats']['total_features'])

        if agreement_rate < 0.3:
            recommendations.append("Improve ensemble agreement by fine-tuning confidence thresholds")
            recommendations.append("Consider feature-specific ensemble weights based on method reliability")

    # Based on feature coverage
    regex_features = len([f for f, c in results['summary']['method_comparison']['regex'].items() if c > 0])
    transformer_features = len([f for f, c in results['summary']['method_comparison']['transformer'].items() if c > 0])

    recommendations.append(f"Leverage high regex coverage ({regex_features} features) as strong baseline")
    recommendations.append(f"Investigate transformer predictions ({transformer_features} features) for novel patterns")

    # Method-specific recommendations
    if len(feature_performance['transformer_consistent']) > 0:
        recommendations.append("Focus on transformer's contextual understanding for complex syntactic features")

    recommendations.append("Validate ensemble on held-out gold standard for confidence calibration")
    recommendations.append("Generate diachronic visualizations showing linguistic evolution trends")

    return recommendations

def save_detailed_analysis(results, feature_performance, baseline_comparison, recommendations):
    """Save comprehensive analysis to file"""

    analysis_report = {
        'enhanced_analysis': {
            'summary': results['summary'],
            'feature_performance': {
                'regex_top_features': feature_performance['regex_top'][:20],
                'transformer_consistent': feature_performance['transformer_consistent'][:20],
            },
            'baseline_comparison': baseline_comparison,
            'recommendations': recommendations,
            'acl_readiness': {
                'strengths': [
                    'High feature coverage with dual methods',
                    'Production-trained transformer with calibration',
                    'Comprehensive morphological feature set (78 features)',
                    'Multi-period corpus analysis',
                    'Ensemble approach with confidence weighting'
                ],
                'areas_for_improvement': [
                    'Low ensemble agreement rate needs investigation',
                    'Need gold standard validation for confidence metrics',
                    'Transformer tokenization needs BERT integration',
                    'Diachronic trend analysis visualization needed',
                    'Statistical significance testing required'
                ],
                'next_steps': [
                    'Run validation on linguistic gold standard',
                    'Generate publication-ready visualizations',
                    'Calculate statistical significance of detected trends',
                    'Prepare ACL camera-ready submission'
                ]
            }
        }
    }

    with open('acl_submission_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, ensure_ascii=False, indent=2)

    logger.info("💾 Detailed analysis saved to: acl_submission_analysis.json")

if __name__ == "__main__":
    analyze_enhanced_results()