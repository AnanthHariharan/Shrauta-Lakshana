#!/usr/bin/env python3
"""
Add regression line to agreement vs detection correlation plot
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from scipy import stats
from pathlib import Path

# Load the existing results
results_path = Path(__file__).parent / '../../data/full_corpus_ensemble_results.json'
with open(results_path, 'r') as f:
    results = json.load(f)

# Extract agreement data
agreement_data = []
for text_name in results['results'].get('agreement_analysis', {}):
    stats_data = results['results']['agreement_analysis'][text_name]
    if stats_data['total_features'] > 0:
        agreement_rate = (stats_data['both_agree_positive'] + stats_data['both_agree_negative']) / stats_data['total_features']
        ensemble_rate = stats_data['ensemble_positive'] / stats_data['total_features']
        agreement_data.append({'agreement': agreement_rate, 'ensemble': ensemble_rate})

df = pd.DataFrame(agreement_data)

# Create plot with regression line
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['agreement'], df['ensemble'], alpha=0.6, c='green', s=50)

# Add regression line
x = df['agreement'].values
y = df['ensemble'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.8, label=f'R²={r_value**2:.3f}, p={p_value:.4f}')

ax.set_xlabel("Agreement Rate", fontsize=12)
ax.set_ylabel("Ensemble Detection Rate", fontsize=12)
ax.set_title("Agreement vs Detection Correlation", fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()

output_path = Path(__file__).parent / '../../../output/agreement_correlation_with_regression.png'
output_path.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")

plt.close()
