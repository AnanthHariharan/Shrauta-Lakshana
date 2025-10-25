#!/usr/bin/env python3
"""
Generate LaTeX table for period-level composite indices
"""

from pathlib import Path

# Data from vedic_comprehensive_diachronic_report.txt
period_data = {
    'Samhita (Early-Middle Vedic)': {
        'archaism': 23.05,
        'innovation': 20.62,
        'conservation': 1.13,
        'morph_innov': 1.84,
        'syntactic': 2.80,
        'phon_arch': 74.29,
        'textual': 5.10,
        'substrate': 0.02,
        'overall': 3.24
    },
    'Brahmana (Late Vedic)': {
        'archaism': 27.19,
        'innovation': 24.47,
        'conservation': 1.13,
        'morph_innov': 2.99,
        'syntactic': 6.66,
        'phon_arch': 79.36,
        'textual': 13.09,
        'substrate': 0.02,
        'overall': 7.58
    },
    'Upanishad (Latest Vedic)': {
        'archaism': 28.26,
        'innovation': 27.41,
        'conservation': 1.05,
        'morph_innov': 2.68,
        'syntactic': 8.64,
        'phon_arch': 76.96,
        'textual': 14.72,
        'substrate': 0.01,
        'overall': 8.68
    },
    'Classical Sanskrit': {
        'archaism': 28.40,
        'innovation': 21.62,
        'conservation': 1.33,
        'morph_innov': 2.24,
        'syntactic': 2.93,
        'phon_arch': 87.93,
        'textual': 9.11,
        'substrate': 0.05,
        'overall': 4.76
    }
}

# Table 1: Basic indices
latex1 = r"""\begin{table}[ht]
\centering
\caption{Period-level linguistic indices. Archaism Index aggregates frequency of archaic morphological features (subjunctive, dual, retroflex /ḷ/, particle \textit{sma}). Innovation Index aggregates innovative features (long compounds, philosophical terms, gerunds, subordinators). Conservation Ratio = Archaism / Innovation, with values $>1$ indicating retention of archaic forms.}
\label{tab:period-detection}
\begin{tabular}{lrrr}
\toprule
\textbf{Period} & \textbf{Archaism Index} & \textbf{Innovation Index} & \textbf{Conservation Ratio} \\
\midrule
"""

for period, data in period_data.items():
    latex1 += f"{period} & {data['archaism']:.2f} & {data['innovation']:.2f} & {data['conservation']:.2f} \\\\\n"

latex1 += r"""\bottomrule
\end{tabular}
\end{table}"""

# Table 2: Advanced composite indices
latex2 = r"""\begin{table}[ht]
\centering
\caption{Advanced composite indices per period. Morphological Innovation Density = mean frequency of periphrastic constructions and innovative case forms. Syntactic Complexity = mean of subordinators, correlatives, and non-finite verb forms. Phonological Archaism = retention of archaic sounds (retroflex /ḷ/, pluti vowels, medial aspirates). Textual Sophistication = prose particles, reported speech, and connectives. Overall Innovation Score = weighted average of morphological and syntactic innovation.}
\label{tab:period-indices}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Period} & \textbf{Morph.} & \textbf{Syntactic} & \textbf{Phon.} & \textbf{Textual} & \textbf{Overall} \\
 & \textbf{Innov.} & \textbf{Complex.} & \textbf{Arch.} & \textbf{Soph.} & \textbf{Innov.} \\
\midrule
"""

for period, data in period_data.items():
    latex2 += f"{period} & {data['morph_innov']:.2f} & {data['syntactic']:.2f} & {data['phon_arch']:.2f} & {data['textual']:.2f} & {data['overall']:.2f} \\\\\n"

latex2 += r"""\bottomrule
\end{tabular}
\end{table}"""

# Combined table (more compact)
latex_combined = r"""\begin{table}[ht]
\centering
\caption{Composite linguistic indices per chronological period. Values represent aggregate feature frequencies (per 1,000 words) across multiple linguistic dimensions. Archaism/Innovation indices measure retention vs. replacement of morphological features. Conservation Ratio $>1$ indicates archaic retention dominates innovation. Advanced indices quantify syntactic complexity, phonological conservation, and discourse sophistication.}
\label{tab:period-detection}
\small
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Period} & \textbf{Arch.} & \textbf{Innov.} & \textbf{Cons.} & \textbf{Synt.} & \textbf{Text.} & \textbf{Overall} \\
 & \textbf{Index} & \textbf{Index} & \textbf{Ratio} & \textbf{Cplx.} & \textbf{Soph.} & \textbf{Innov.} \\
\midrule
"""

for period, data in period_data.items():
    latex_combined += f"{period} & {data['archaism']:.1f} & {data['innovation']:.1f} & {data['conservation']:.2f} & {data['syntactic']:.1f} & {data['textual']:.1f} & {data['overall']:.1f} \\\\\n"

latex_combined += r"""\bottomrule
\end{tabular}
\end{table}"""

# Save all versions
output_dir = Path(__file__).parent / '../../../output'

with open(output_dir / 'period_basic_indices.tex', 'w') as f:
    f.write(latex1)

with open(output_dir / 'period_advanced_indices.tex', 'w') as f:
    f.write(latex2)

with open(output_dir / 'period_combined_indices.tex', 'w') as f:
    f.write(latex_combined)

print("TABLE 1: BASIC INDICES (Archaism, Innovation, Conservation)")
print("="*80)
print(latex1)

print("\n\nTABLE 2: ADVANCED COMPOSITE INDICES")
print("="*80)
print(latex2)

print("\n\nTABLE 3: COMBINED (COMPACT VERSION)")
print("="*80)
print(latex_combined)

print("\n" + "="*80)
print("✓ Saved 3 versions:")
print(f"  - {output_dir / 'period_basic_indices.tex'}")
print(f"  - {output_dir / 'period_advanced_indices.tex'}")
print(f"  - {output_dir / 'period_combined_indices.tex'}")
