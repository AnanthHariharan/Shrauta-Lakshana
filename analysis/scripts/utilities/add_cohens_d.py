#!/usr/bin/env python3
"""
Compute Cohen's d for all features and add to statistics table
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load existing statistical summary
csv_path = Path(__file__).parent / '../../../output/feature_statistical_summary.csv'
df = pd.read_csv(csv_path)

# Load features over time to compute Cohen's d
features_csv = Path(__file__).parent / '../../../output/features_over_time.csv'
features_df = pd.read_csv(features_csv, index_col=0).T

# Function to compute Cohen's d
def cohens_d(group1, group2):
    """Compute Cohen's d effect size between two groups"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group2) - np.mean(group1)) / pooled_std

# Compute Cohen's d for each feature (first 5 vs last 5 texts)
cohens_d_values = []

for feature in df['feature']:
    if feature in features_df.columns:
        values = features_df[feature].values
        early = values[:5]  # First 5 texts (Early Vedic)
        late = values[-5:]  # Last 5 texts (Classical)
        d = cohens_d(early, late)
        cohens_d_values.append(d)
    else:
        cohens_d_values.append(np.nan)

# Add Cohen's d to dataframe
df['Cohen\'s d'] = cohens_d_values

# Save updated CSV
df.to_csv(csv_path, index=False)
print(f"✓ Updated {csv_path} with Cohen's d values")

# Show summary
print("\n" + "="*80)
print("COHEN'S d EFFECT SIZE SUMMARY")
print("="*80)
print(f"Mean |Cohen's d|: {df['Cohen\'s d'].abs().mean():.3f}")
print(f"Median |Cohen's d|: {df['Cohen\'s d'].abs().median():.3f}")
print(f"Features with large effect (|d| > 0.8): {(df['Cohen\'s d'].abs() > 0.8).sum()}")
print(f"Features with medium effect (0.5 < |d| < 0.8): {((df['Cohen\'s d'].abs() > 0.5) & (df['Cohen\'s d'].abs() <= 0.8)).sum()}")
print(f"Features with small effect (0.2 < |d| < 0.5): {((df['Cohen\'s d'].abs() > 0.2) & (df['Cohen\'s d'].abs() <= 0.5)).sum()}")

# Now regenerate the LaTeX table with Cohen's d
key_features = [
    'subjunctive_full',
    'perfect_reduplicated',
    'dual_nominative',
    'particle_sma',
    'long_compounds',
    'philosophical_terms',
    'monophthongs_e',
    'retroflex_l',
    'deity_names',
    'gerund_tvaa',
    'prose_particles',
    'reported_speech',
    'connectives',
    'diphthongs_ai',
    'injunctive_augmentless'
]

key_df = df[df['feature'].isin(key_features)].copy()
key_df = key_df.sort_values('R²', ascending=False)

feature_names = {
    'subjunctive_full': 'Subjunctive (full paradigm)',
    'perfect_reduplicated': 'Perfect reduplicated',
    'dual_nominative': 'Dual nominative',
    'particle_sma': 'Particle \\textit{sma}',
    'long_compounds': 'Long compounds',
    'philosophical_terms': 'Philosophical terminology',
    'monophthongs_e': 'Monophthong /e/',
    'retroflex_l': 'Retroflex /ḷ/',
    'deity_names': 'Deity names',
    'gerund_tvaa': 'Gerund \\textit{-tvā}',
    'prose_particles': 'Prose particles',
    'reported_speech': 'Reported speech',
    'connectives': 'Connectives',
    'diphthongs_ai': 'Diphthong /ai/',
    'injunctive_augmentless': 'Injunctive (augmentless)'
}

# Generate updated LaTeX table
latex = r"""\begin{table}[ht]
\centering
\caption{Statistical summary of diachronic trends for key linguistic features. $R^2$ = proportion of variance explained by linear trend. $\beta$ = rate of change per text position. Spearman $\rho$ = rank correlation. Cohen's $d$ = effect size between Early Vedic (first 5 texts) and Classical (last 5 texts); $|d| > 0.8$ indicates large effect, $0.5 < |d| < 0.8$ medium, $0.2 < |d| < 0.5$ small. Significance: $^{***}p < 0.001$, $^{**}p < 0.01$, $^{*}p < 0.05$.}
\label{tab:top-features}
\small
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Feature} & \textbf{$R^2$} & \textbf{$\beta$} & \textbf{$p$} & \textbf{$\rho$} & \textbf{Cohen's $d$} & \textbf{Effect} \\
\midrule
"""

for idx, row in key_df.iterrows():
    feature = feature_names.get(row['feature'], row['feature'])
    r2 = row['R²']
    beta = row['β (slope)']
    p = row['p-value']
    rho = row['Spearman ρ']
    d = row['Cohen\'s d']

    # Add significance stars
    if p < 0.001:
        sig = r'$^{***}$'
    elif p < 0.01:
        sig = r'$^{**}$'
    elif p < 0.05:
        sig = r'$^{*}$'
    else:
        sig = ''

    # Effect size label
    abs_d = abs(d)
    if abs_d > 0.8:
        effect = 'Large'
    elif abs_d > 0.5:
        effect = 'Medium'
    elif abs_d > 0.2:
        effect = 'Small'
    else:
        effect = 'Negligible'

    latex += f"{feature} & {r2:.3f} & {beta:.4f} & {p:.4f}{sig} & {rho:.3f} & {d:.3f} & {effect} \\\\\n"

latex += r"""\bottomrule
\end{tabular}
\end{table}"""

# Save updated table
output_path = Path(__file__).parent / '../../../output/top_features_table.tex'
with open(output_path, 'w') as f:
    f.write(latex)

print("\n" + "="*80)
print("UPDATED TABLE WITH COHEN'S d")
print("="*80)
print(latex)
print(f"\n✓ Updated LaTeX table saved to: {output_path}")

# Print top 10 by |Cohen's d|
print("\n" + "="*80)
print("TOP 10 FEATURES BY |COHEN'S d| EFFECT SIZE")
print("="*80)
df_sorted_d = df.copy()
df_sorted_d['abs_d'] = df_sorted_d['Cohen\'s d'].abs()
df_sorted_d = df_sorted_d.sort_values('abs_d', ascending=False)

print(f"{'Feature':<30} {'Cohen\'s d':>12} {'Effect Size':<12} {'Direction'}")
print("-"*80)
for idx, row in df_sorted_d.head(10).iterrows():
    d = row['Cohen\'s d']
    abs_d = abs(d)
    if abs_d > 0.8:
        effect = 'Large'
    elif abs_d > 0.5:
        effect = 'Medium'
    elif abs_d > 0.2:
        effect = 'Small'
    else:
        effect = 'Negligible'

    direction = 'Increase' if d > 0 else 'Decrease'
    print(f"{row['feature']:<30} {d:>12.3f} {effect:<12} {direction}")
