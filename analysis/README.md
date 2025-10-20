# Analysis Directory - Organized Structure

This directory contains all analysis scripts, models, visualizations, and results for the Shrauta-Lakshana project.

Last organized: 2025-10-14

---

## 📁 Directory Structure

```
analysis/
├── scripts/           # All analysis scripts organized by category
│   ├── diachronic/   # Diachronic linguistic evolution analysis
│   ├── statistical/  # Statistical modeling and type-token analysis
│   ├── ensemble/     # Transformer ensemble and validation systems
│   └── utilities/    # Helper scripts and analysis runners
├── training/         # Transformer model training
│   ├── models/       # Trained PyTorch model files (.pt)
│   ├── data/         # Training data (if separate from main data/)
│   └── logs/         # Training logs
├── tests/            # Test scripts for validation
├── data/             # JSON data files and results
├── visualizations/   # All generated plots (PNG files)
├── docs/             # Documentation, reports, and papers
├── models/           # Legacy models directory
├── logs/             # Legacy logs directory
├── plots/            # Legacy plots directory
└── training_data/    # Legacy training data directory
```

---

## 📂 Detailed Contents

### `scripts/` - Analysis Scripts

#### `scripts/diachronic/`
**Purpose**: Analyze linguistic evolution across historical periods

- **`diachronic_analysis.py`** (37 KB)
  - Main diachronic analysis with 78+ linguistic features
  - Classes: `VedicDiachronicAnalyzer`, `EnhancedVedicAnalyzer`
  - Functions: `plot_diachronic_trends()`, `plot_category_trends()`, `export_results()`
  - Generates: `vedic_comprehensive_diachronic_analysis.png`, `vedic_category_trends.png`
  - Output: `vedic_analysis.csv`, `vedic_comprehensive_diachronic_report.txt`

- **`enhanced_diachronic_transformer_analysis.py`** (24 KB)
  - Transformer-based diachronic pattern detection
  - Advanced neural network approach to linguistic change

#### `scripts/statistical/`
**Purpose**: Statistical modeling and lexical diversity analysis

- **`statistical_modeling.py`** (39 KB)
  - Machine learning models for feature prediction
  - Classes: `VedicTextMetadata`, `VedicStatisticalModeler`, `VedicStatisticalVisualizer`
  - Models: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting
  - Generates: Model performance plots, PCA analysis, regression diagnostics, clustering dendrograms
  - Output: 7 visualization files

- **`type_token_analysis.py`** (28 KB)
  - Type-Token Ratio (TTR) and lexical diversity analysis
  - Classes: `VedicTypeTokenAnalyzer`, `VedicTypeTokenVisualizer`
  - Measures: TTR, Root TTR, Herdan's C, Shannon entropy, Zipfian analysis
  - Generates: `vedic_type_token_comparison.png`, `vedic_ttr_evolution.png`
  - Output: `type_token_analysis_report.txt`

#### `scripts/ensemble/`
**Purpose**: Neural-symbolic ensemble systems combining regex and transformers

- **`full_corpus_ensemble_analysis.py`** (35 KB)
  - Complete corpus analysis with dual methods
  - Generates: 5 ensemble visualization plots
  - Methods: Regex pattern matching + Transformer predictions
  - Output: Agreement analysis, method comparison

- **`enhanced_bert_ensemble_system.py`** (23 KB)
  - BERT-based ensemble with confidence calibration
  - mBERT fine-tuning for Sanskrit morphology
  - Temperature scaling for calibrated predictions

- **`transformer_morphological_analyzer.py`** (44 KB)
  - Core transformer architecture for morphological analysis
  - Multi-label classification with BCEWithLogitsLoss
  - 50-75 morphological features

- **`transformer_validation_system.py`** (24 KB)
  - Validation framework for transformer predictions
  - Cross-validation and metrics computation

#### `scripts/utilities/`
**Purpose**: Helper scripts and analysis runners

- **`compute_validation_metrics.py`** (16 KB)
  - **IMPORTANT**: Computes validation metrics for ACL paper
  - Metrics: Period classification F1, confidence calibration, pattern detection agreement
  - Output: `validation_metrics_for_paper.json`

- **`comprehensive_results_analysis.py`** (19 KB)
  - Comprehensive analysis of ensemble results
  - Comparison with baseline metrics

- **`enhanced_results_analysis.py`** (10 KB)
  - Enhanced ensemble results comparison
  - ACL submission evaluation

- **`run_enhanced_ensemble_analysis.py`** (15 KB)
  - Runner script for ensemble analysis

- **`run_statistical_analysis.py`** (13 KB)
  - Runner script for statistical modeling

- **`run_type_token_analysis.py`** (15 KB)
  - Runner script for type-token analysis

- **`immediate_improvements_guide.py`** (11 KB)
  - Guide for improving analysis results

- **`calculate_period_word_counts.py`** (3 KB)
  - Utility to calculate word counts by period

- **`samhita_analysis.py`** (4 KB)
  - Focused analysis of Samhita texts

---

### `training/` - Model Training

#### `training/` (root)
Training scripts for transformer models:

- **`train_transformer_production.py`** (15 KB) - Production training pipeline
- **`train_transformer_efficient.py`** (10 KB) - Efficient training with optimizations
- **`train_transformer_fixed.py`** (10 KB) - Fixed training issues
- **`train_transformer_quickstart.py`** (11 KB) - Quick training for testing

#### `training/models/`
Trained PyTorch model files (`.pt` format):

- `sanskrit_transformer_morph_analyzer_final.pt` (727 MB) - Final morphological analyzer
- `sanskrit_transformer_final.pt` (723 MB) - Final transformer model
- `sanskrit_transformer_efficient_final.pt` (727 MB) - Efficient training final model
- `sanskrit_transformer_efficient_best.pt` (723 MB) - Best checkpoint from efficient training
- `test_complete_model.pt` (727 MB) - Complete test model

**Total model size**: ~3.6 GB

---

### `tests/` - Test Scripts

Validation and testing scripts:

- **`test_trained_model.py`** (5 KB) - Test trained transformer models
- **`test_improved_system.py`** (3 KB) - Test system improvements
- **`test_complete_pipeline.py`** (2 KB) - Test complete analysis pipeline
- **`test_temperature_fix.py`** (2 KB) - Test temperature scaling fix
- **`test_size_fix.py`** (2 KB) - Test size compatibility fix
- **`test_transformer_fix.py`** (2 KB) - Test transformer fixes
- **`test_training_mini.py`** (2 KB) - Minimal training test

---

### `data/` - JSON Data Files

Analysis results and training data:

- **`validation_metrics_for_paper.json`** (6 KB)
  - **IMPORTANT**: Validation metrics for ACL paper
  - Contains: Period classification F1, confidence calibration r/MAE/ECE, pattern detection agreement, diachronic trends

- **`full_corpus_ensemble_results.json`** (468 KB)
  - Complete ensemble analysis results for 20 texts

- **`enhanced_ensemble_analysis_results.json`** (94 KB)
  - Enhanced ensemble comparison results

- **`sanskrit_morph_training_data.json`** (9.8 MB)
  - Training data for morphological analyzer

- **`improvement_summary.json`** (1 KB)
  - Summary of improvements to analysis

---

### `visualizations/` - Generated Plots

All visualization outputs (PNG format, 300 DPI):

#### Diachronic Analysis (2 files)
- `vedic_comprehensive_diachronic_analysis.png` (2.8 MB) - 6-panel feature evolution
- `vedic_category_trends.png` (1.1 MB) - Category-level trends

#### Ensemble Analysis (5 files)
- `diachronic_evolution.png` (693 KB) - Evolution across periods
- `method_comparison.png` (661 KB) - Regex vs Transformer comparison
- `agreement_analysis.png` (450 KB) - Inter-method agreement
- `feature_heatmap.png` (695 KB) - 78 features × 20 texts heatmap
- `period_characteristics.png` (454 KB) - Period-distinctive features

See `../PLOTTING_GUIDE.md` for complete visualization documentation.

---

### `docs/` - Documentation

Documentation, reports, and paper materials:

- **`ACL_SUBMISSION_SUMMARY.md`** (9 KB)
  - **IMPORTANT**: Complete ACL submission summary
  - Contains: Validation metrics, recommended phrasing, strengths/limitations, next steps

- **`acl_formatted_tables.tex`** (5 KB)
  - LaTeX tables for ACL paper

- **`requirements_transformer.txt`** (1 KB)
  - Python dependencies for transformer models

- **`transformer_validation_report.txt`** (1 KB)
  - Validation report for transformer system

---

## 🚀 Quick Start Guide

### Run Diachronic Analysis
```bash
cd scripts/diachronic
python diachronic_analysis.py
```
**Output**: `../../output/vedic_comprehensive_diachronic_analysis.png`, `vedic_analysis.csv`

### Run Statistical Modeling
```bash
cd scripts/utilities
python run_statistical_analysis.py
```
**Output**: Model performance plots in `../../visualizations/`

### Run Type-Token Analysis
```bash
cd scripts/utilities
python run_type_token_analysis.py
```
**Output**: TTR plots in `../../visualizations/`

### Run Ensemble Analysis
```bash
cd scripts/utilities
python run_enhanced_ensemble_analysis.py
```
**Output**: Ensemble plots in `../../visualizations/`

### Compute Validation Metrics (ACL Paper)
```bash
cd scripts/utilities
python compute_validation_metrics.py
```
**Output**: `../../data/validation_metrics_for_paper.json`

---

## 📊 Key Output Files

### For ACL Paper Submission
1. **Metrics**: `data/validation_metrics_for_paper.json`
2. **Summary**: `docs/ACL_SUBMISSION_SUMMARY.md`
3. **Tables**: `docs/acl_formatted_tables.tex`
4. **Main Figure**: `visualizations/vedic_comprehensive_diachronic_analysis.png`
5. **Results CSV**: `../output/vedic_analysis.csv`

### For Analysis
1. **Diachronic Results**: `../output/vedic_analysis.csv` (20 texts × 78 features)
2. **Full Ensemble**: `data/full_corpus_ensemble_results.json`
3. **All Visualizations**: `visualizations/*.png` (16 plot files)

---

## 🔧 Dependencies

Install required packages:
```bash
pip install -r docs/requirements_transformer.txt
```

Main dependencies:
- Python 3.8+
- PyTorch 2.0+
- transformers (Hugging Face)
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- scipy

---

## 📝 File Size Summary

| Directory | Files | Total Size |
|-----------|-------|------------|
| `training/models/` | 5 | ~3.6 GB |
| `visualizations/` | 16 | ~7.5 MB |
| `data/` | 5 | ~10.4 MB |
| `scripts/` | 17 | ~400 KB |
| `tests/` | 7 | ~20 KB |
| `docs/` | 4 | ~16 KB |

**Total**: ~3.62 GB

---

## 🗂️ Legacy Directories

The following directories contain older files and may be cleaned up in the future:

- `models/` - Legacy model storage (duplicates in `training/models/`)
- `logs/` - Legacy training logs
- `plots/` - Legacy plot storage (now in `visualizations/`)
- `training_data/` - Legacy training data (now in `data/`)
- `validation_reports/` - Legacy validation reports (now in `docs/`)

---

## 📚 Related Documentation

- **Main README**: `../README.md`
- **Plotting Guide**: `../PLOTTING_GUIDE.md`
- **ACL Submission**: `docs/ACL_SUBMISSION_SUMMARY.md`

---

## 🤝 Contributing

When adding new scripts:
1. Place in appropriate `scripts/` subdirectory
2. Update this README
3. Add documentation to script header
4. Follow existing naming conventions

When adding new visualizations:
1. Save to `visualizations/` directory
2. Use 300 DPI, PNG format
3. Update `../PLOTTING_GUIDE.md`

When adding new data:
1. Save JSON to `data/` directory
2. Document format and purpose
3. Update this README

---

**Maintained by**: Ananth Hariharan
**Last updated**: 2025-10-14
