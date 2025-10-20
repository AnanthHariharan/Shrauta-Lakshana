#!/usr/bin/env python3
"""
Comprehensive Type-Token Analysis Integration Script

This script demonstrates how to run type-token analysis alongside traditional
token-only frequency analysis, providing insights into:

1. Morphological productivity (type/token ratios)
2. Vocabulary richness and diversity
3. Hapax legomena and frequency distributions
4. Zipfian patterns in lexical usage
5. Diachronic evolution of lexical diversity

Usage:
    python run_type_token_analysis.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from type_token_analysis import VedicTypeTokenAnalyzer, VedicTypeTokenVisualizer
from diachronic_analysis import EnhancedVedicAnalyzer
import pandas as pd
import numpy as np

def main():
    """Run comprehensive type-token analysis"""
    print("VEDIC SANSKRIT TYPE-TOKEN FREQUENCY ANALYSIS")
    print("=" * 70)
    print("Comprehensive analysis of type frequency vs token frequency")
    print("Revealing morphological productivity and vocabulary richness")
    print()
    
    # Step 1: Initialize analyzers
    print("Step 1: Initializing Analysis Systems")
    print("-" * 50)
    
    # Type-token analyzer (new)
    tt_analyzer = VedicTypeTokenAnalyzer()
    
    # Traditional token analyzer (existing)
    token_analyzer = EnhancedVedicAnalyzer()
    
    print("✓ Type-token analyzer initialized")
    print("✓ Traditional token analyzer initialized")
    
    # Step 2: Define corpus
    print("\nStep 2: Corpus Definition")
    print("-" * 50)
    
    corpus_files = {
        # Samhitas (Early to Middle Vedic)
        'Rigveda': '../texts/samhita/rig-samhita.txt',
        'Samaveda': '../texts/samhita/sama-samhita.txt',
        'Yajurveda': '../texts/samhita/yajur-samhita.txt',
        'Atharvaveda (Paippalada)': '../texts/samhita/atharva-paippalada-samhita.txt',
        'Atharvaveda (Saunaka)': '../texts/samhita/atharva-saunaka-samhita.txt',

        # Brahmanas (Late Vedic)
        'Kausitaki-Br': '../texts/brahmana/rig-kausitaki.txt',
        'Pancavimsa-Br': '../texts/brahmana/sama-pancavimsa.txt',
        'Satapatha-Br': '../texts/brahmana/yajur-satapatha.txt',
        'Gopatha-Br': '../texts/brahmana/atharva-gopatha.txt',

        # Upanishads (Latest Vedic)
        'Aitareya-Up': '../texts/upanishad/rig-aitareya.txt',
        'Taittiriya-Up': '../texts/upanishad/yajur-taittiriya-up.txt',
        'Chandogya-Up': '../texts/upanishad/sama-chandogya.txt',
        'Brhadaranyaka-Up': '../texts/upanishad/yajur-brhadaranyaka.txt',
        'Prashna-Up': '../texts/upanishad/atharva-prashna.txt',
        'Shvetashvatara-Up': '../texts/upanishad/yajur-shvetashvatara.txt',
        
        # Classical Sanskrit (Post-Vedic)
        'Ramayana': '../texts/classical-sanskrit/ramayana.txt',
        'Mahabharata': '../texts/classical-sanskrit/mahabharata.txt',
        'Bhagavata-Purana': '../texts/classical-sanskrit/bhagavata-purana.txt'
    }
    
    text_order = list(corpus_files.keys())
    print(f"Corpus: {len(corpus_files)} texts across 4 chronological periods (Vedic Samhitas, Brahmanas, Upanishads, Classical Sanskrit)")
    
    # Step 3: Run comprehensive analysis
    print("\nStep 3: Comprehensive Text Analysis")
    print("-" * 50)
    
    # Type-token analysis
    print("Running type-token analysis...")
    for text_name, filepath in corpus_files.items():
        try:
            tt_analyzer.analyze_text_comprehensive(filepath, text_name)
        except FileNotFoundError:
            print(f"⚠ Warning: File not found - {filepath}")
            continue
        except Exception as e:
            print(f"⚠ Warning: Error analyzing {text_name} - {e}")
            continue
    
    # Traditional token analysis for comparison
    print("\nRunning traditional token analysis...")
    try:
        token_analyzer.analyze_corpus(corpus_files)
        print("✓ Traditional analysis completed")
    except Exception as e:
        print(f"⚠ Warning: Traditional analysis issues - {e}")
    
    # Step 4: Calculate comprehensive metrics
    print("\nStep 4: Calculating Linguistic Metrics")
    print("-" * 50)
    
    # Morphological productivity
    print("Calculating morphological productivity...")
    productivity_results = tt_analyzer.calculate_morphological_productivity()
    
    # Lexical diversity measures
    print("Calculating lexical diversity measures...")
    diversity_results = tt_analyzer.calculate_lexical_diversity_measures()
    
    # Frequency distributions and Zipfian analysis
    print("Analyzing frequency distributions...")
    distribution_results = tt_analyzer.analyze_frequency_distributions()
    
    # Diachronic comparison
    print("Comparing type-token evolution...")
    evolution_results = tt_analyzer.compare_type_token_evolution(text_order)
    
    print("✓ All metrics calculated successfully")
    
    # Step 5: Comparative Analysis
    print("\nStep 5: Type vs Token Comparative Analysis")
    print("-" * 50)
    
    # Compare key findings
    analyze_morphological_productivity(productivity_results)
    analyze_lexical_diversity_trends(diversity_results, text_order)
    analyze_zipfian_patterns(distribution_results)
    
    # Step 6: Generate visualizations
    print("\nStep 6: Generating Visualizations")
    print("-" * 50)
    
    visualizer = VedicTypeTokenVisualizer(tt_analyzer)
    
    print("Creating type-token comparison plots...")
    visualizer.plot_type_token_comparison()
    
    print("Creating TTR evolution plots...")
    visualizer.plot_ttr_evolution(text_order)
    
    # Step 7: Generate comprehensive reports
    print("\nStep 7: Generating Reports")
    print("-" * 50)
    
    # Type-token specific report
    print("Generating type-token analysis report...")
    tt_analyzer.generate_comprehensive_report('vedic_type_token_analysis_report.txt')
    
    # Comparative analysis report
    print("Generating comparative analysis report...")
    generate_comparative_report(tt_analyzer, token_analyzer, text_order)
    
    # Statistical summary
    print("Generating statistical summary...")
    generate_statistical_summary(productivity_results, diversity_results, distribution_results)
    
    print("\n" + "=" * 70)
    print("TYPE-TOKEN ANALYSIS COMPLETE")
    print("=" * 70)
    print("Generated outputs:")
    print("• Type-token comparison visualizations")
    print("• TTR evolution plots")
    print("• Comprehensive analysis report")
    print("• Comparative token vs type-token report")
    print("• Statistical summary")
    print()
    print("Key insights:")
    print("• Morphological productivity patterns revealed")
    print("• Vocabulary richness evolution quantified")
    print("• Type-token ratios show linguistic complexity changes")
    print("• Zipfian distributions validate text authenticity")

def analyze_morphological_productivity(productivity_results):
    """Analyze morphological productivity patterns"""
    print("\nMORPHOLOGICAL PRODUCTIVITY ANALYSIS")
    print("-" * 40)
    
    # Find most and least productive features
    feature_productivity = {}
    
    for text_name, features in productivity_results.items():
        for feature_name, metrics in features.items():
            if feature_name not in feature_productivity:
                feature_productivity[feature_name] = []
            feature_productivity[feature_name].append(metrics['productivity_score'])
    
    # Average productivity across texts
    avg_productivity = {feature: np.mean(scores) 
                       for feature, scores in feature_productivity.items() 
                       if scores}
    
    if avg_productivity:
        print("Most productive morphological patterns:")
        for feature, score in sorted(avg_productivity.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feature.replace('_', ' ')}: {score:.4f}")
        
        print("\nLeast productive morphological patterns:")
        for feature, score in sorted(avg_productivity.items(), key=lambda x: x[1])[:5]:
            print(f"  {feature.replace('_', ' ')}: {score:.4f}")

def analyze_lexical_diversity_trends(diversity_results, text_order):
    """Analyze lexical diversity trends across periods"""
    print("\nLEXICAL DIVERSITY TRENDS")
    print("-" * 40)
    
    # Organize by periods
    periods = {
        'Early Vedic': text_order[:4],
        'Late Vedic': text_order[4:8], 
        'Latest Vedic': text_order[8:]
    }
    
    for period, texts in periods.items():
        ttrs = []
        herdan_cs = []
        
        for text in texts:
            if text in diversity_results:
                ttrs.append(diversity_results[text]['ttr'])
                herdan_cs.append(diversity_results[text]['herdan_c'])
        
        if ttrs:
            print(f"\n{period}:")
            print(f"  Average TTR: {np.mean(ttrs):.4f} (±{np.std(ttrs):.4f})")
            print(f"  Average Herdan's C: {np.mean(herdan_cs):.4f} (±{np.std(herdan_cs):.4f})")

def analyze_zipfian_patterns(distribution_results):
    """Analyze Zipfian distribution patterns"""
    print("\nZIPFIAN DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    all_zipf_coeffs = []
    all_r2_values = []
    good_fits = 0
    total_distributions = 0
    
    for text_name, features in distribution_results.items():
        for feature_name, metrics in features.items():
            total_distributions += 1
            if metrics['zipf_r2'] > 0.5:  # Good fit threshold
                good_fits += 1
                all_zipf_coeffs.append(metrics['zipf_coefficient'])
                all_r2_values.append(metrics['zipf_r2'])
    
    if all_zipf_coeffs:
        print(f"Zipfian analysis results:")
        print(f"  Good fits: {good_fits}/{total_distributions} ({good_fits/total_distributions:.1%})")
        print(f"  Average Zipf coefficient: {np.mean(all_zipf_coeffs):.3f}")
        print(f"  Average R²: {np.mean(all_r2_values):.3f}")
        
        # Classic Zipf law has α ≈ 1
        classic_zipf = [coeff for coeff in all_zipf_coeffs if 0.8 <= coeff <= 1.2]
        print(f"  Classic Zipf distributions (α ≈ 1): {len(classic_zipf)}/{len(all_zipf_coeffs)}")

def generate_comparative_report(tt_analyzer, token_analyzer, text_order):
    """Generate comparative analysis report"""
    
    with open('vedic_type_token_comparative_report.txt', 'w', encoding='utf-8') as f:
        f.write("COMPARATIVE ANALYSIS: TYPE-TOKEN vs TOKEN-ONLY ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("METHODOLOGY COMPARISON\n")
        f.write("-" * 50 + "\n")
        f.write("Traditional Token Analysis:\n")
        f.write("• Counts total occurrences of linguistic patterns\n")
        f.write("• Measures frequency and productivity\n")
        f.write("• Focuses on usage patterns\n\n")
        
        f.write("Type-Token Analysis:\n")
        f.write("• Distinguishes unique forms (types) from total occurrences (tokens)\n")
        f.write("• Measures morphological diversity and vocabulary richness\n")
        f.write("• Reveals productive vs. fossilized patterns\n\n")
        
        f.write("KEY INSIGHTS FROM TYPE-TOKEN ANALYSIS\n")
        f.write("-" * 50 + "\n")
        
        # Calculate key insights
        diversity_results = tt_analyzer.calculate_lexical_diversity_measures()
        productivity_results = tt_analyzer.calculate_morphological_productivity()
        
        # TTR evolution
        f.write("TTR Evolution Across Periods:\n")
        periods = {
            'Early Vedic': text_order[:4],
            'Late Vedic': text_order[4:8], 
            'Latest Vedic': text_order[8:]
        }
        
        for period, texts in periods.items():
            ttrs = [diversity_results[text]['ttr'] for text in texts if text in diversity_results]
            if ttrs:
                f.write(f"  {period}: {np.mean(ttrs):.4f} (range: {min(ttrs):.4f}-{max(ttrs):.4f})\n")
        
        # Morphological productivity insights
        f.write("\nMorphological Productivity Patterns:\n")
        
        # Calculate feature-wise productivity averages
        feature_productivity = {}
        for text_name, features in productivity_results.items():
            for feature_name, metrics in features.items():
                if feature_name not in feature_productivity:
                    feature_productivity[feature_name] = []
                feature_productivity[feature_name].append(metrics['ttr'])
        
        # Most diverse morphological categories
        avg_ttr = {feature: np.mean(ttrs) for feature, ttrs in feature_productivity.items() if ttrs}
        
        if avg_ttr:
            f.write("Most morphologically diverse categories:\n")
            for feature, ttr in sorted(avg_ttr.items(), key=lambda x: x[1], reverse=True)[:5]:
                f.write(f"  {feature.replace('_', ' ')}: TTR = {ttr:.3f}\n")
        
        f.write("\nIMPLICATIONS FOR VEDIC SANSKRIT STUDIES\n")
        f.write("-" * 50 + "\n")
        f.write("• Type-token ratios reveal morphological complexity changes\n")
        f.write("• High TTR indicates productive, creative language use\n")
        f.write("• Low TTR suggests formulaic or ritualized language\n")
        f.write("• Diachronic TTR changes track linguistic evolution\n")
        f.write("• Hapax legomena patterns validate text authenticity\n")

def generate_statistical_summary(productivity_results, diversity_results, distribution_results):
    """Generate statistical summary table"""
    
    # Create comprehensive dataframe
    summary_data = []
    
    for text_name in diversity_results:
        row = {'Text': text_name}
        
        # Diversity metrics
        row.update(diversity_results[text_name])
        
        # Productivity averages
        if text_name in productivity_results:
            productivity_scores = [metrics['productivity_score'] 
                                 for metrics in productivity_results[text_name].values() 
                                 if metrics['productivity_score'] > 0]
            row['avg_productivity'] = np.mean(productivity_scores) if productivity_scores else 0
        
        # Distribution metrics
        if text_name in distribution_results:
            zipf_coeffs = [metrics['zipf_coefficient'] 
                          for metrics in distribution_results[text_name].values() 
                          if metrics['zipf_r2'] > 0.3]
            row['avg_zipf_coeff'] = np.mean(zipf_coeffs) if zipf_coeffs else 0
            
            entropies = [metrics['shannon_entropy'] 
                        for metrics in distribution_results[text_name].values() 
                        if metrics['total_tokens'] > 10]
            row['avg_entropy'] = np.mean(entropies) if entropies else 0
        
        summary_data.append(row)
    
    # Save as CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('vedic_type_token_statistical_summary.csv', index=False)
    print("Statistical summary saved to CSV")

if __name__ == "__main__":
    main()