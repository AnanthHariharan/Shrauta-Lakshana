# Analysis Directory - Quick Reference Card

**Last organized**: 2025-10-14

---

## 📂 Directory Map

```
analysis/
├── 📜 scripts/          → All analysis code (organized by type)
├── 🎓 training/         → Model training scripts & trained models (3.6 GB)
├── 🧪 tests/            → Validation & test scripts
├── 💾 data/             → JSON results & training data
├── 📊 visualizations/   → All plots (PNG files)
└── 📚 docs/             → Documentation & ACL materials
```

---

## 🚀 Common Tasks

### Generate All Visualizations
```bash
# Diachronic plots
python scripts/diachronic/diachronic_analysis.py

# Statistical plots
python scripts/utilities/run_statistical_analysis.py

# Type-Token plots
python scripts/utilities/run_type_token_analysis.py

# Ensemble plots
python scripts/utilities/run_enhanced_ensemble_analysis.py
```

### Run ACL Paper Metrics
```bash
python scripts/utilities/compute_validation_metrics.py
# Output: data/validation_metrics_for_paper.json
```

### Train New Model
```bash
python training/train_transformer_production.py
# Output: training/models/*.pt
```

---

## 📁 Find Key Files

| What You Need | Location |
|---------------|----------|
| **ACL validation metrics** | `data/validation_metrics_for_paper.json` |
| **ACL submission summary** | `docs/ACL_SUBMISSION_SUMMARY.md` |
| **Main results CSV** | `../output/vedic_analysis.csv` |
| **LaTeX tables** | `docs/acl_formatted_tables.tex` |
| **Plotting guide** | `../PLOTTING_GUIDE.md` |
| **Main diachronic plot** | `visualizations/vedic_comprehensive_diachronic_analysis.png` |
| **Trained models** | `training/models/*.pt` (5 files, 3.6 GB) |
| **Ensemble results** | `data/full_corpus_ensemble_results.json` |

---

## 📜 Script Categories

### Diachronic Analysis (`scripts/diachronic/`)
- `diachronic_analysis.py` - Main 78-feature diachronic analysis
- `enhanced_diachronic_transformer_analysis.py` - Neural diachronic patterns

### Statistical Analysis (`scripts/statistical/`)
- `statistical_modeling.py` - ML models (5 types), PCA, clustering
- `type_token_analysis.py` - TTR, lexical diversity, Zipfian analysis

### Ensemble Methods (`scripts/ensemble/`)
- `full_corpus_ensemble_analysis.py` - Regex + Transformer ensemble
- `enhanced_bert_ensemble_system.py` - mBERT with calibration
- `transformer_morphological_analyzer.py` - Core transformer (44 KB)
- `transformer_validation_system.py` - Validation framework

### Utilities (`scripts/utilities/`)
- `compute_validation_metrics.py` - **ACL paper metrics** ⭐
- `run_enhanced_ensemble_analysis.py` - Ensemble runner
- `run_statistical_analysis.py` - Statistical runner
- `run_type_token_analysis.py` - Type-token runner
- `comprehensive_results_analysis.py` - Results comparison
- `enhanced_results_analysis.py` - Enhanced analysis
- `calculate_period_word_counts.py` - Word count utility
- `samhita_analysis.py` - Samhita-specific analysis

---

## 📊 Visualization Outputs

| Category | Files | Total Size |
|----------|-------|------------|
| Diachronic | 2 | 3.9 MB |
| Ensemble | 5 | 3.0 MB |
| Statistical | 7 | 3.4 MB |
| Type-Token | 2 | 1.4 MB |
| **Total** | **16** | **~7.7 MB** |

All visualizations are 300 DPI PNG, publication-ready.

---

## 💾 Data Files

| File | Size | Purpose |
|------|------|---------|
| `validation_metrics_for_paper.json` | 6 KB | **ACL validation metrics** ⭐ |
| `full_corpus_ensemble_results.json` | 468 KB | Complete ensemble results |
| `enhanced_ensemble_analysis_results.json` | 94 KB | Enhanced comparison |
| `sanskrit_morph_training_data.json` | 9.8 MB | Transformer training data |
| `improvement_summary.json` | 1 KB | Analysis improvements |

---

## 🎓 Model Files

**Location**: `training/models/`

| File | Size | Description |
|------|------|-------------|
| `sanskrit_transformer_morph_analyzer_final.pt` | 727 MB | **Final production model** ⭐ |
| `sanskrit_transformer_final.pt` | 723 MB | Final transformer |
| `sanskrit_transformer_efficient_final.pt` | 727 MB | Efficient training final |
| `sanskrit_transformer_efficient_best.pt` | 723 MB | Best checkpoint |
| `test_complete_model.pt` | 727 MB | Test model |

**Total**: ~3.6 GB

---

## 🧪 Test Suite

**Location**: `tests/`

- `test_trained_model.py` - Test trained models
- `test_improved_system.py` - Test improvements
- `test_complete_pipeline.py` - Full pipeline test
- `test_temperature_fix.py` - Temperature scaling test
- `test_size_fix.py` - Size compatibility test
- `test_transformer_fix.py` - Transformer fixes test
- `test_training_mini.py` - Minimal training test

---

## 📚 Documentation

**Location**: `docs/`

- `ACL_SUBMISSION_SUMMARY.md` - **Complete ACL summary** ⭐
- `acl_formatted_tables.tex` - LaTeX tables for paper
- `requirements_transformer.txt` - Python dependencies
- `transformer_validation_report.txt` - Validation report

---

## 🔗 Related Files

- **Main README**: `../README.md`
- **Plotting Guide**: `../PLOTTING_GUIDE.md`
- **This Directory README**: `README.md`
- **Results CSV**: `../output/vedic_analysis.csv`

---

## 💡 Pro Tips

1. **Before running scripts**: Check current working directory
   ```bash
   pwd  # Should be in analysis/
   ```

2. **Import paths**: Scripts expect to be run from their directory
   ```bash
   cd scripts/diachronic
   python diachronic_analysis.py
   ```

3. **Large files**: Model files are ~3.6 GB total, consider `.gitignore`

4. **Outputs**: Most scripts output to `../output/` or `../../visualizations/`

5. **Dependencies**: Install from `docs/requirements_transformer.txt`
   ```bash
   pip install -r docs/requirements_transformer.txt
   ```

---

## 🗑️ Legacy Directories

Can be cleaned up later:
- `models/` (duplicates in `training/models/`)
- `logs/` (old training logs)
- `plots/` (now in `visualizations/`)
- `training_data/` (now in `data/`)
- `validation_reports/` (now in `docs/`)

---

## 📏 Size Summary

| Component | Size |
|-----------|------|
| Models | 3.6 GB |
| Training data | 9.8 MB |
| Results data | 560 KB |
| Visualizations | 7.7 MB |
| Scripts | 400 KB |
| Total | **~3.62 GB** |

---

**Need help?** See `README.md` for detailed documentation.

**For ACL paper**: Check `docs/ACL_SUBMISSION_SUMMARY.md`

**For plotting**: See `../PLOTTING_GUIDE.md`

---

Last updated: 2025-10-14
