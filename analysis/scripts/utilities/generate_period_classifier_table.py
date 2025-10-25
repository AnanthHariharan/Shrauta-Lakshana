#!/usr/bin/env python3
"""
Generate LaTeX tables for Period Classifier performance
"""

import json
from pathlib import Path

# Load metrics
json_path = Path(__file__).parent / '../../../analysis/data/validation_metrics_for_paper.json'
with open(json_path) as f:
    data = json.load(f)

period_class = data['period_classification']

# Table 1: Per-class metrics
latex1 = r"""\begin{table}[ht]
\centering
\caption{Period classifier performance. The transformer's auxiliary period classification head predicts one of four chronological periods (Early Vedic, Late Vedic, Latest Vedic, Classical) from text representations. Metrics computed on held-out validation set. Poor performance (33\% accuracy) reflects substantial linguistic variation within periods and limited training data ($n=18$ validation samples).}
\label{tab:period-classifier}
\begin{tabular}{lrrrr}
\toprule
\textbf{Period} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Support} \\
\midrule
"""

period_order = ['Early Vedic', 'Late Vedic', 'Latest Vedic', 'Classical']
for period in period_order:
    metrics = period_class['classification_report'][period]
    latex1 += f"{period} & {metrics['precision']:.3f} & {metrics['recall']:.3f} & {metrics['f1-score']:.3f} & {int(metrics['support'])} \\\\\n"

latex1 += r"""\midrule
"""

# Add macro average
macro = period_class['classification_report']['macro avg']
latex1 += f"Macro average & {macro['precision']:.3f} & {macro['recall']:.3f} & {macro['f1-score']:.3f} & {int(macro['support'])} \\\\\n"

latex1 += r"""\midrule
\multicolumn{5}{l}{\textbf{Overall Accuracy: """ + f"{period_class['accuracy']:.3f}" + r"""}} \\
\bottomrule
\end{tabular}
\end{table}"""

# Table 2: Confusion matrix
latex2 = r"""\begin{table}[ht]
\centering
\caption{Period classifier confusion matrix. Rows represent true periods, columns represent predicted periods. The classifier struggles to distinguish Late Vedic texts (0\% recall), frequently misclassifying them as Classical. Latest Vedic texts are also often misclassified as Classical, suggesting these periods share substantial linguistic features.}
\label{tab:period-confusion}
\begin{tabular}{lrrrr}
\toprule
\textbf{True / Predicted} & \textbf{Early V.} & \textbf{Late V.} & \textbf{Latest V.} & \textbf{Classical} \\
\midrule
"""

cm = period_class['confusion_matrix']
period_labels = ['Early Vedic', 'Late Vedic', 'Latest Vedic', 'Classical']
period_abbrev = ['Early V.', 'Late V.', 'Latest V.', 'Classical']

for i, (period, abbrev) in enumerate(zip(period_labels, period_abbrev)):
    row = cm[i]
    # Bold the diagonal (correct predictions)
    formatted_row = []
    for j, val in enumerate(row):
        if i == j and val > 0:
            formatted_row.append(f"\\textbf{{{val}}}")
        else:
            formatted_row.append(str(val))
    latex2 += f"{abbrev} & {' & '.join(formatted_row)} \\\\\n"

latex2 += r"""\bottomrule
\end{tabular}
\end{table}"""

# Combined compact version
latex_combined = r"""\begin{table}[ht]
\centering
\caption{Period classification performance (validation set, $n=18$). Overall accuracy 33.3\%. Confusion matrix shows true period (rows) vs. predicted (columns). Bold indicates correct predictions. The classifier fails to identify Late Vedic texts (0\% recall), often confusing them with Classical Sanskrit.}
\label{tab:period-classifier}
\small
\begin{tabular}{lrrrr|rrr}
\toprule
\multicolumn{5}{c}{\textbf{Confusion Matrix}} & \multicolumn{3}{c}{\textbf{Performance}} \\
\cmidrule(lr){1-5} \cmidrule(lr){6-8}
\textbf{True / Pred.} & \textbf{EV} & \textbf{LV} & \textbf{LatV} & \textbf{Cl} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} \\
\midrule
"""

for i, (period, abbrev) in enumerate(zip(period_labels, period_abbrev)):
    row = cm[i]
    formatted_row = []
    for j, val in enumerate(row):
        if i == j and val > 0:
            formatted_row.append(f"\\textbf{{{val}}}")
        else:
            formatted_row.append(str(val))

    metrics = period_class['classification_report'][period]
    latex_combined += f"{abbrev} & {' & '.join(formatted_row)} & {metrics['precision']:.2f} & {metrics['recall']:.2f} & {metrics['f1-score']:.2f} \\\\\n"

latex_combined += r"""\midrule
\multicolumn{5}{l}{Overall Accuracy: 0.333} & \multicolumn{3}{r}{Macro F1: 0.304} \\
\bottomrule
\end{tabular}
\end{table}"""

# Save all versions
output_dir = Path(__file__).parent / '../../../output'

with open(output_dir / 'period_classifier_metrics.tex', 'w') as f:
    f.write(latex1)

with open(output_dir / 'period_confusion_matrix.tex', 'w') as f:
    f.write(latex2)

with open(output_dir / 'period_classifier_combined.tex', 'w') as f:
    f.write(latex_combined)

print("TABLE 1: PERIOD CLASSIFIER METRICS")
print("="*80)
print(latex1)

print("\n\nTABLE 2: CONFUSION MATRIX")
print("="*80)
print(latex2)

print("\n\nTABLE 3: COMBINED (COMPACT)")
print("="*80)
print(latex_combined)

print("\n" + "="*80)
print("✓ Saved 3 versions:")
print(f"  - {output_dir / 'period_classifier_metrics.tex'}")
print(f"  - {output_dir / 'period_confusion_matrix.tex'}")
print(f"  - {output_dir / 'period_classifier_combined.tex'}")
print("\nKey findings:")
print(f"  • Overall accuracy: {period_class['accuracy']:.1%}")
print(f"  • Late Vedic completely failed (F1=0.00)")
print(f"  • Latest Vedic best precision (100%) but low recall (33%)")
print(f"  • Classical has perfect recall (100%) but low precision (27%)")
print(f"  • Classifier tends to over-predict Classical period")
