# ✅ New Individual Feature Trend Plots Generated

## Summary

Successfully created **4 new plot files** per Dr. Mortensen's feedback, showing **individual features as separate lines** instead of aggregated categories.

---

## How to Run (Quick Reference)

```bash
cd /Users/ananthhariharan/Documents/Research/Shrauta-Lakshana/analysis/scripts/utilities
python3 plot_individual_feature_trends.py
```

**That's it!** The script will:
1. Auto-detect the location of `vedic_analysis.csv`
2. Generate all 4 plots
3. Save them to the `output/` directory

---

## Generated Files

All files are in: `/Users/ananthhariharan/Documents/Research/Shrauta-Lakshana/output/`

### 1. **vedic_individual_feature_trends.png** (931 KB)
**4-panel plot showing individual morphological features**
- Panel 1: Archaic Verbal (subjunctive_full, perfect_reduplicated, injunctive_modal)
- Panel 2: Archaic Nominal + Particles (dual_nominative, dual_instrumental, particle_sma)
- Panel 3: Innovative Features (long_compounds, infinitive_tum, subordinators)
- Panel 4: Lexical Evolution (philosophical_terms, ritual_terminology, deity_names)

**Use case**: Comprehensive view of morphological evolution

### 2. **vedic_phonological_individual_trends.png** (504 KB)
**2-panel plot for phonological features**
- Panel 1: Archaic sounds (retroflex_l, pluti_vowels, medial_voiced_aspirates)
- Panel 2: Vowel evolution (diphthongs_ai, diphthongs_au, monophthongs_e, monophthongs_o)

**Use case**: Phonological evolution analysis

### 3. **vedic_table4_feature_trends.png** (481 KB) ⭐ **RECOMMENDED**
**Single comprehensive plot with 7 key features from your Table 4**
- subjunctive_full
- perfect_reduplicated
- dual_nominative
- particle_sma
- long_compounds
- philosophical_terms
- monophthongs_e

**Use case**: **Main figure for your paper** - shows all critical features discussed in Table 4

### 4. **vedic_unexpected_trends.png** (594 KB)
**2-panel plot highlighting unexpected findings**
- Panel 1: Subjunctive increase (all subjunctive subcategories)
- Panel 2: Expected declining features (particle_sma, retroflex_l, dual_nominative)

**Use case**: Emphasize the unexpected subjunctive trend you discuss in the paper

---

## Key Improvements vs Old Plot

| Old (`vedic_category_trends.png`) | New (Individual Feature Plots) |
|-----------------------------------|--------------------------------|
| ❌ Aggregated features into categories | ✅ Each line = one specific feature |
| ❌ Only 2 lines per panel | ✅ 3-7 features per panel for comparison |
| ❌ Hides individual patterns | ✅ Shows distinct evolutionary paths |
| ❌ Can't see subjunctive anomaly | ✅ Clearly visualizes unexpected trends |
| No period context | ✅ Background shading for historical periods |

---

## Recommendation for Your Paper

### Main Text Figure:
Use **`vedic_table4_feature_trends.png`**
- Shows the 7 key features you discuss
- Clean, publication-ready
- Directly supports your narrative

### Supporting Figure (Optional):
Use **`vedic_unexpected_trends.png`**
- Highlights the subjunctive anomaly
- Contrasts expected vs unexpected trends

### Supplementary Materials:
Include **`vedic_individual_feature_trends.png`** and **`vedic_phonological_individual_trends.png`**
- More comprehensive views
- Readers interested in details

---

## Sample Figure Caption

> **Figure X: Diachronic evolution of key morphological features.** Each line represents the frequency (per 1,000 words) of a single linguistic feature across 19 Sanskrit texts arranged chronologically (Early Vedic to Classical Sanskrit, c. 1500 BCE–800 CE). Background shading indicates historical periods: Early Vedic (light blue), Late Vedic (light yellow), Latest Vedic (light purple), Classical (light red). Notable trends include the unexpected increase in subjunctive forms (red line) alongside the expected decline of archaic dual nominative (green) and particle *sma* (blue), and the marked rise in philosophical terminology (brown) and long compounds (purple) in later periods.

---

## What Changed from Old to New

### Dr. Mortensen's Critique:
> "The aggregation is too broad. I want to see individual features, not averaged categories."

### Your Response (What We Did):
1. ✅ **Separated** all features - no more averaging
2. ✅ **Plotted** 3-7 individual features per panel
3. ✅ **Grouped** related features together for comparison
4. ✅ **Added** period shading for context
5. ✅ **Highlighted** unexpected trends (subjunctive increase)
6. ✅ **Created** a focused "Table 4" plot with your 7 key features

---

## Customization (If Needed)

### Change which features are shown:
Edit line ~274 in `plot_individual_feature_trends.py`:
```python
features = [
    'subjunctive_full',
    'your_feature_here',
    # Add more
]
```

### Change colors:
Edit the `colors` lists (e.g., line ~117):
```python
colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, Orange, Green
```

### Change figure size:
Edit `figsize` parameters (e.g., line ~101):
```python
fig, axes = plt.subplots(2, 2, figsize=(16, 10))  # Width x Height in inches
```

---

## Next Steps

1. ✅ **Review** the 4 new PNG files in the `output/` directory
2. ✅ **Select** which plots to use in your paper (recommend `vedic_table4_feature_trends.png`)
3. ✅ **Replace** old `vedic_category_trends.png` references in your LaTeX
4. ✅ **Write** figure captions explaining individual features
5. ✅ **Address** Dr. Mortensen's feedback in your response

---

## Technical Details

- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG (can convert to PDF/EPS if needed)
- **Data source**: `vedic_analysis.csv` (19 texts × 83 features)
- **Period boundaries**: Based on your chronological ordering
- **Feature normalization**: Per 1,000 words (already in your CSV)

---

## Troubleshooting

**Q: Plots look different from expected?**
A: Check your CSV column names match the script's feature names (case-sensitive)

**Q: Want different features?**
A: Edit the `features` lists in the script functions

**Q: Need higher resolution?**
A: Change `dpi=300` to `dpi=600` (but files will be larger)

**Q: Want to revert to old plots?**
A: Old `vedic_category_trends.png` is still in `output/` directory

---

**Status**: ✅ Complete - Ready for paper submission
**Files**: 4 new plots generated
**Next**: Review plots and update paper figures
