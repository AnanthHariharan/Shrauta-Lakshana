#!/usr/bin/env python3
"""
Generate LaTeX table with statistical summaries for key features
"""

import pandas as pd
from pathlib import Path

# Load the computed statistics
csv_path = Path(__file__).parent / '../../../output/feature_statistical_summary.csv'
df = pd.read_csv(csv_path)

# Key features to include in the table
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

# Filter to key features
key_df = df[df['feature'].isin(key_features)].copy()

# Sort by R² descending
key_df = key_df.sort_values('R²', ascending=False)

# Create feature name mapping for better display
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

# Generate LaTeX table
latex = r"""\begin{table}[ht]
\centering
\caption{Statistical summary of diachronic trends for key linguistic features. Linear regression models test for monotonic change across the 20-text chronological sequence. Features are sorted by $R^2$ (proportion of variance explained by chronological position). $\beta$ indicates the rate of change per text position. Spearman $\rho$ measures rank correlation, robust to non-linear trends. Significance: $^{***}p < 0.001$, $^{**}p < 0.01$, $^{*}p < 0.05$.}
\label{tab:feature-statistics}
\begin{tabular}{lrrrrl}
\toprule
\textbf{Feature} & \textbf{$R^2$} & \textbf{$\beta$} & \textbf{$p$-value} & \textbf{Spearman $\rho$} & \textbf{Trend} \\
\midrule
"""

for idx, row in key_df.iterrows():
    feature = feature_names.get(row['feature'], row['feature'])
    r2 = row['R²']
    beta = row['β (slope)']
    p = row['p-value']
    rho = row['Spearman ρ']
    trend = row['MK trend']

    # Add significance stars
    if p < 0.001:
        sig = r'$^{***}$'
    elif p < 0.01:
        sig = r'$^{**}$'
    elif p < 0.05:
        sig = r'$^{*}$'
    else:
        sig = ''

    # Format trend direction with arrows
    if 'increas' in trend.lower():
        trend_symbol = r'$\uparrow$'
    elif 'decreas' in trend.lower():
        trend_symbol = r'$\downarrow$'
    else:
        trend_symbol = r'$\rightarrow$'

    # Format numbers
    latex += f"{feature} & {r2:.3f} & {beta:.4f} & {p:.4f}{sig} & {rho:.3f} & {trend_symbol} \\\\\n"

latex += r"""\bottomrule
\end{tabular}
\end{table}"""

# Save to file
output_path = Path(__file__).parent / '../../../output/feature_statistics_table.tex'
with open(output_path, 'w') as f:
    f.write(latex)

print(latex)
print(f"\n✓ LaTeX table saved to: {output_path}")

# Also create a more compact version with just top 10
print("\n" + "="*80)
print("COMPACT VERSION (TOP 10 BY R²)")
print("="*80)

latex_compact = r"""\begin{table}[ht]
\centering
\caption{Top 10 linguistic features by diachronic trend strength ($R^2$).}
\label{tab:top-features}
\begin{tabular}{lrrrr}
\toprule
\textbf{Feature} & \textbf{$R^2$} & \textbf{$\beta$} & \textbf{$p$} & \textbf{$\rho$} \\
\midrule
"""

for idx, row in key_df.head(10).iterrows():
    feature = feature_names.get(row['feature'], row['feature'])
    r2 = row['R²']
    beta = row['β (slope)']
    p = row['p-value']
    rho = row['Spearman ρ']

    # Add significance stars
    if p < 0.001:
        sig = r'$^{***}$'
    elif p < 0.01:
        sig = r'$^{**}$'
    elif p < 0.05:
        sig = r'$^{*}$'
    else:
        sig = ''

    latex_compact += f"{feature} & {r2:.3f} & {beta:.4f} & {p:.4f}{sig} & {rho:.3f} \\\\\n"

latex_compact += r"""\bottomrule
\end{tabular}
\end{table}"""

output_path_compact = Path(__file__).parent / '../../../output/top_features_table.tex'
with open(output_path_compact, 'w') as f:
    f.write(latex_compact)

print(latex_compact)
print(f"\n✓ Compact LaTeX table saved to: {output_path_compact}")
