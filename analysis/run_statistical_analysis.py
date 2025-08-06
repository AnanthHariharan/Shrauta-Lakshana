#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis Script for Vedic Sanskrit Diachronic Studies

This script demonstrates the complete statistical modeling workflow:
1. Load diachronic analysis results
2. Apply text metadata classification
3. Build predictive models using text characteristics
4. Quantify feature importance
5. Generate comprehensive visualizations and reports

Usage:
    python run_statistical_analysis.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diachronic_analysis import EnhancedVedicAnalyzer
from statistical_modeling import VedicTextMetadata, VedicStatisticalModeler, VedicStatisticalVisualizer
import pandas as pd
import numpy as np

def main():
    """Run comprehensive statistical analysis"""
    print("VEDIC SANSKRIT STATISTICAL MODELING ANALYSIS")
    print("=" * 70)
    print("Integrating diachronic linguistics with statistical prediction")
    print()
    
    # Step 1: Load linguistic analysis data
    print("Step 1: Loading Linguistic Analysis Data")
    print("-" * 50)
    
    analyzer = EnhancedVedicAnalyzer()
    
    corpus_files = {
        # Samhitas (Early to Middle Vedic)
        'Rigveda': '../texts/samhita/rig-samhita.txt',
        'Yajurveda': '../texts/samhita/yajur-samhita.txt',
        'Samaveda': '../texts/samhita/sama-samhita.txt',
        'Atharvaveda': '../texts/samhita/atharva-samhita.txt',

        # Brahmanas (Late Vedic)
        'Kausitaki-Br': '../texts/brahmana/rig-kausitaki.txt',
        'Pancavimsa-Br': '../texts/brahmana/sama-pancavimsa.txt',
        'Satapatha-Br': '../texts/brahmana/yajur-satapatha.txt',
        'Gopatha-Br': '../texts/brahmana/atharva-gopatha.txt',

        # Upanishads (Latest Vedic)
        'Brhadaranyaka-Up': '../texts/upanishad/yajur-brhadaranyaka.txt',
        'Chandogya-Up': '../texts/upanishad/sama-chandogya.txt',
        'Aitareya-Up': '../texts/upanishad/rig-aitareya.txt',
        'Mandukya-Up': '../texts/upanishad/atharva-mandukya.txt'
    }
    
    print("Analyzing corpus files...")
    try:
        analyzer.analyze_corpus(corpus_files)
        print(f"✓ Successfully analyzed {len(corpus_files)} texts")
        print(f"✓ Extracted {len(analyzer.features)} linguistic features")
    except Exception as e:
        print(f"⚠ Warning: Some files may not be available: {e}")
        print("Proceeding with available data...")
    
    # Step 2: Initialize statistical modeling components
    print("\nStep 2: Initializing Statistical Modeling")
    print("-" * 50)
    
    metadata_handler = VedicTextMetadata()
    modeler = VedicStatisticalModeler(analyzer, metadata_handler)
    visualizer = VedicStatisticalVisualizer(modeler)
    
    print("✓ Text metadata classification system loaded")
    print("✓ Statistical modeling pipeline initialized")
    print("✓ Visualization system ready")
    
    # Step 3: Build predictive models
    print("\nStep 3: Building Predictive Models")
    print("-" * 50)
    
    # Key linguistic features for modeling
    target_features = [
        'subjunctive_full',      # Archaic morphology
        'retroflex_l',           # Phonological archaism
        'dual_nominative',       # Morphological archaism
        'long_compounds',        # Syntactic innovation
        'philosophical_terms',   # Lexical development
        'correlatives_ya_ta',    # Syntactic innovation
        'perfect_periphrastic',  # Morphological innovation
        'particle_sma'           # Archaic particle
    ]
    
    model_results = modeler.build_predictive_models(target_features)
    
    # Step 4: Analyze feature importance
    print("\nStep 4: Analyzing Feature Importance")
    print("-" * 50)
    
    importance_results = modeler.analyze_feature_importance()
    
    # Step 5: Model validation
    print("\nStep 5: Model Validation")
    print("-" * 50)
    
    validation_results = modeler.validate_models()
    
    # Step 6: Detailed regression analysis for key features
    print("\nStep 6: Detailed Regression Analysis")
    print("-" * 50)
    
    regression_analyses = {}
    key_features_for_regression = ['subjunctive_full', 'long_compounds', 'philosophical_terms']
    
    for feature in key_features_for_regression:
        if feature in target_features:
            print(f"\nAnalyzing: {feature}")
            regression_results = modeler.perform_regression_analysis(feature)
            if regression_results:
                regression_analyses[feature] = regression_results
    
    # Step 7: Multivariate analysis
    print("\nStep 7: Multivariate Analysis")
    print("-" * 50)
    
    # Principal Component Analysis
    print("Performing Principal Component Analysis...")
    pca_results = modeler.principal_component_analysis()
    
    # Hierarchical Clustering
    print("Performing Hierarchical Clustering...")
    clustering_results = modeler.cluster_texts_by_features()
    
    # Step 8: Generate comprehensive visualizations
    print("\nStep 8: Generating Visualizations")
    print("-" * 50)
    
    print("Creating model performance plots...")
    visualizer.plot_model_performance()
    
    print("Creating feature importance heatmap...")
    visualizer.plot_feature_importance_heatmap()
    
    print("Creating PCA analysis plots...")
    visualizer.plot_pca_analysis(pca_results)
    
    print("Creating clustering dendrogram...")
    visualizer.plot_clustering_dendrogram(clustering_results)
    
    # Generate regression diagnostics for key features
    for feature, results in regression_analyses.items():
        print(f"Creating regression diagnostics for {feature}...")
        visualizer.plot_regression_diagnostics(results, feature)
    
    # Step 9: Generate comprehensive statistical report
    print("\nStep 9: Generating Statistical Report")
    print("-" * 50)
    
    generate_statistical_report(modeler, model_results, importance_results, 
                               validation_results, pca_results, clustering_results)
    
    # Step 10: Demonstrate prediction capabilities
    print("\nStep 10: Prediction Demonstrations")
    print("-" * 50)
    
    # Predict linguistic features for specific texts
    demo_texts = ['Rigveda', 'Satapatha-Br', 'Brhadaranyaka-Up']
    demo_features = ['subjunctive_full', 'long_compounds']
    
    for text in demo_texts:
        for feature in demo_features:
            if feature in model_results:
                print(f"\nPredicting {feature} for {text}:")
                explanation = modeler.explain_model_predictions(feature, text)
    
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 70)
    print("Generated outputs:")
    print("• Model performance plots")
    print("• Feature importance heatmaps") 
    print("• PCA analysis visualizations")
    print("• Hierarchical clustering dendrograms")
    print("• Regression diagnostic plots")
    print("• Comprehensive statistical report")
    print()
    print("All visualizations saved as PNG files in current directory")

def generate_statistical_report(modeler, model_results, importance_results, 
                               validation_results, pca_results, clustering_results):
    """Generate comprehensive statistical analysis report"""
    
    with open('vedic_statistical_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE STATISTICAL ANALYSIS REPORT\n")
        f.write("Vedic Sanskrit Diachronic Modeling\n")
        f.write("=" * 70 + "\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 50 + "\n")
        f.write("This report presents statistical modeling results for Vedic Sanskrit\n")
        f.write("diachronic analysis, using text characteristics (prose/poetry, period,\n")
        f.write("recension) as predictive features for linguistic phenomena.\n\n")
        
        # Model Performance Summary
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("-" * 50 + "\n")
        
        if model_results:
            best_models = {}
            for target, models in model_results.items():
                best_model = max(models.keys(), key=lambda x: models[x]['cv_r2_mean'])
                best_r2 = models[best_model]['cv_r2_mean']
                best_models[target] = (best_model, best_r2)
            
            f.write("Best performing models by target feature:\n")
            for target, (model, r2) in sorted(best_models.items(), key=lambda x: x[1][1], reverse=True):
                f.write(f"  {target:25s}: {model:20s} (R² = {r2:.3f})\n")
        
        # Feature Importance Analysis
        f.write("\n\nFEATURE IMPORTANCE ANALYSIS\n")
        f.write("-" * 50 + "\n")
        
        if importance_results:
            # Aggregate importance across models
            feature_importance_agg = {}
            for target, models in importance_results.items():
                for model_name, features in models.items():
                    for feature, importance in features.items():
                        if feature not in feature_importance_agg:
                            feature_importance_agg[feature] = []
                        feature_importance_agg[feature].append(importance)
            
            # Calculate average importance
            avg_importance = {feature: np.mean(importances) 
                            for feature, importances in feature_importance_agg.items()}
            
            f.write("Most important predictive features (averaged across models):\n")
            for feature, importance in sorted(avg_importance.items(), 
                                            key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"  {feature:25s}: {importance:.4f}\n")
        
        # Model Validation Summary
        f.write("\n\nMODEL VALIDATION SUMMARY\n")
        f.write("-" * 50 + "\n")
        
        if validation_results:
            recommended_models = []
            for target, models in validation_results.items():
                for model_name, validation in models.items():
                    if validation['recommended']:
                        recommended_models.append((target, model_name, validation['interpretation']))
            
            f.write("Recommended models (low overfitting risk, good performance):\n")
            for target, model, interpretation in recommended_models:
                f.write(f"  {target} - {model}: {interpretation}\n")
        
        # PCA Results
        f.write("\n\nPRINCIPAL COMPONENT ANALYSIS\n")
        f.write("-" * 50 + "\n")
        
        if pca_results:
            explained_var = pca_results['explained_variance_ratio']
            f.write(f"First 5 components explain {sum(explained_var[:5]):.1%} of variance:\n")
            for i, var in enumerate(explained_var[:5]):
                f.write(f"  PC{i+1}: {var:.3f} ({var:.1%})\n")
        
        # Clustering Results
        f.write("\n\nHIERARCHICAL CLUSTERING RESULTS\n")
        f.write("-" * 50 + "\n")
        
        if clustering_results:
            clustered_data = clustering_results['clustered_data']
            cluster_labels = clustering_results['cluster_labels']
            
            for i in range(1, max(cluster_labels) + 1):
                cluster_texts = clustered_data[clustered_data['cluster'] == i].index.tolist()
                f.write(f"Cluster {i}: {', '.join(cluster_texts)}\n")
                
                # Analyze cluster characteristics
                cluster_subset = clustered_data[clustered_data['cluster'] == i]
                periods = cluster_subset['period'].value_counts()
                text_types = cluster_subset['text_type'].value_counts()
                
                f.write(f"  Periods: {dict(periods)}\n")
                f.write(f"  Types: {dict(text_types)}\n\n")
        
        # Methodology
        f.write("METHODOLOGY\n")
        f.write("-" * 50 + "\n")
        f.write("• Linear regression, Ridge, Lasso, Random Forest, Gradient Boosting models\n")
        f.write("• Cross-validation for model selection and validation\n")
        f.write("• Feature importance analysis via coefficients and tree-based methods\n")
        f.write("• Principal Component Analysis for dimensionality reduction\n")
        f.write("• Hierarchical clustering for text grouping\n")
        f.write("• Comprehensive diagnostic plots for model validation\n\n")
        
        f.write("Statistical modeling demonstrates quantitative relationships between\n")
        f.write("text characteristics and linguistic feature frequencies, enabling\n")
        f.write("prediction and explanation of diachronic language change patterns.\n")

if __name__ == "__main__":
    main()