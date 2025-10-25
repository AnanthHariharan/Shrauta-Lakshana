# How to Generate Individual Feature Trend Plots

## Dr. Mortensen's Feedback

Instead of plotting aggregated categories (e.g., "Morphological Archaic"), plot **individual features** as separate lines (e.g., "Subjunctive Full", "Dual Nominative", "Particle Sma") to show distinct evolutionary paths.

---

## Quick Start (3 Steps)

### 1. Navigate to the script directory
```bash
cd /Users/ananthhariharan/Documents/Research/Shrauta-Lakshana/analysis/scripts/utilities
```

### 2. Run the plotting script
```bash
python3 plot_individual_feature_trends.py
```

### 3. Find your plots
The new plots will be saved in:
```
/Users/ananthhariharan/Documents/Research/Shrauta-Lakshana/output/
```

---

## What Gets Generated

The script creates **4 new plot files**:

### 1. `vedic_individual_feature_trends.png` (Main plot)
- **4 panels** showing individual features grouped by type:
  - **Panel 1**: Archaic Verbal (subjunctive, perfect, injunctive)
  - **Panel 2**: Archaic Nominal + Particles (dual, particle_sma)
  - **Panel 3**: Innovative Features (long_compounds, infinitive, subordinators)
  - **Panel 4**: Lexical Evolution (philosophical_terms, ritual_terminology, deity_names)

### 2. `vedic_phonological_individual_trends.png`
- **2 panels** for phonological features:
  - Archaic sounds (retroflex_l, pluti_vowels, etc.)
  - Vowel evolution (diphthongs vs monophthongs)

### 3. `vedic_table4_feature_trends.png` ⭐ **RECOMMENDED FOR PAPER**
- **Single plot** with all key features from Table 4
- Shows: subjunctive_full, perfect_reduplicated, dual_nominative, particle_sma, long_compounds, philosophical_terms, monophthongs_e
- **Use this one in your paper!**

### 4. `vedic_unexpected_trends.png`
- **2 panels** highlighting:
  - The unexpected subjunctive increase
  - Expected declining archaic features

---

## Customization

### Change which features are plotted

Edit the script at line ~220:
```python
features = [
    'subjunctive_full',
    'perfect_reduplicated',
    'dual_nominative',
    # Add your features here
]
```

### Change colors or markers

Edit the colors/markers lists around line ~225:
```python
colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, orange, green
markers = ['o', 's', '^']                    # Circle, square, triangle
```

### Change output directory

Edit line ~18:
```python
def load_data(csv_path='../../output/vedic_analysis.csv'):
```

Or pass it as an argument when calling the functions.

---

## Troubleshooting

### Error: "FileNotFoundError: vedic_analysis.csv"
**Solution**: Make sure `vedic_analysis.csv` exists in the `output/` directory. If not, run your main analysis first:
```bash
cd /Users/ananthhariharan/Documents/Research/Shrauta-Lakshana/analysis/scripts/diachronic
python3 diachronic_analysis.py
```

### Error: "ModuleNotFoundError: No module named 'seaborn'"
**Solution**: Install required packages:
```bash
pip install matplotlib seaborn pandas numpy
```

### Plots look cramped or labels overlap
**Solution**: The script uses high DPI (300). If viewing on screen looks weird, the saved PNG files will be crisp for publication.

### Feature names don't match your CSV
**Solution**: Check your CSV column names:
```bash
head -1 /Users/ananthhariharan/Documents/Research/Shrauta-Lakshana/output/vedic_analysis.csv
```

Then update the feature names in the script to match exactly (case-sensitive).

---

## Differences from Old Plot

### OLD (vedic_category_trends.png):
- ❌ Averaged features into broad categories
- ❌ Only 2 lines per panel (Archaic vs Innovative)
- ❌ Hides individual feature patterns

### NEW (vedic_individual_feature_trends.png):
- ✅ Each line = one specific feature
- ✅ Multiple features per panel for comparison
- ✅ Shows distinct evolutionary paths
- ✅ Period shading for context
- ✅ Directly visualizes unexpected subjunctive trend

---

## For Your Paper

**Recommended usage**:
1. Use `vedic_table4_feature_trends.png` as your **main figure**
2. Use `vedic_unexpected_trends.png` to highlight the subjunctive anomaly
3. Move `vedic_individual_feature_trends.png` to supplementary materials (shows more detail)

**Figure caption example**:
> **Figure X: Diachronic trends of key morphological features.** Each line represents the frequency (per 1,000 words) of a single linguistic feature across 19 texts arranged chronologically. Background shading indicates historical periods (Early Vedic, Late Vedic, Latest Vedic, Classical). Note the unexpected increase in subjunctive forms (red line) alongside the expected decline of archaic features like particle *sma* (blue line) and dual nominative (green line).

---

## Quick Reference

| Task | Command |
|------|---------|
| Generate all plots | `python3 plot_individual_feature_trends.py` |
| View output | `open ../../output/vedic_table4_feature_trends.png` |
| Re-run analysis | `cd ../diachronic && python3 diachronic_analysis.py` |
| Check CSV columns | `head -1 ../../output/vedic_analysis.csv` |

---

**Questions?** Check the inline comments in `plot_individual_feature_trends.py` or email your advisor.
