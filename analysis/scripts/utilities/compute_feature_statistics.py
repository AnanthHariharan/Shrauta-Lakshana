#!/usr/bin/env python3
"""
Compute comprehensive statistical summaries for diachronic features
Outputs: R², β, p-values, Spearman ρ for top changing features
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Load the features over time data
csv_path = Path(__file__).parent / '../../../output/features_over_time.csv'
df = pd.read_csv(csv_path, index_col=0)

# Transpose so texts are rows, features are columns
df = df.T

# Chronological position (1-20)
chronology = np.arange(1, len(df) + 1)

# Compute statistics for each feature
results = []

for feature in df.columns:
    y = df[feature].values

    # Skip if all zeros or constant
    if np.std(y) == 0:
        continue

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(chronology, y)
    r_squared = r_value ** 2

    # Spearman correlation
    spearman_rho, spearman_p = stats.spearmanr(chronology, y)

    # Mann-Kendall trend (simplified - just check if increasing/decreasing)
    # Count concordant vs discordant pairs
    n = len(y)
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(y[j] - y[i])

    # Determine trend direction
    if s > 0:
        mk_trend = "increasing"
    elif s < 0:
        mk_trend = "decreasing"
    else:
        mk_trend = "no trend"

    # Percent change from first to last
    if y[0] != 0:
        pct_change = ((y[-1] - y[0]) / y[0]) * 100
    else:
        pct_change = np.inf if y[-1] > 0 else 0

    results.append({
        'feature': feature,
        'R²': r_squared,
        'β (slope)': slope,
        'p-value': p_value,
        'Spearman ρ': spearman_rho,
        'Spearman p': spearman_p,
        'MK trend': mk_trend,
        '% change': pct_change,
        'first_val': y[0],
        'last_val': y[-1]
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Sort by absolute percent change
results_df['abs_pct_change'] = results_df['% change'].abs()
results_df = results_df.sort_values('abs_pct_change', ascending=False)

# Print top 20 changing features
print("=" * 120)
print("TOP 20 FEATURES BY MAGNITUDE OF DIACHRONIC CHANGE")
print("=" * 120)
print(f"{'Feature':<30} {'R²':>8} {'β':>10} {'p-value':>10} {'ρ':>8} {'MK Trend':<12} {'% Change':>10}")
print("-" * 120)

for idx, row in results_df.head(20).iterrows():
    print(f"{row['feature']:<30} {row['R²']:>8.3f} {row['β (slope)']:>10.4f} {row['p-value']:>10.4f} "
          f"{row['Spearman ρ']:>8.3f} {row['MK trend']:<12} {row['% change']:>10.1f}%")

# Print summary statistics
print("\n" + "=" * 120)
print("STATISTICAL SUMMARY")
print("=" * 120)

sig_features = results_df[results_df['p-value'] < 0.05]
print(f"Features with p < 0.05: {len(sig_features)} / {len(results_df)}")

high_r2 = results_df[results_df['R²'] > 0.5]
print(f"Features with R² > 0.5: {len(high_r2)} / {len(results_df)}")

strong_spearman = results_df[results_df['Spearman ρ'].abs() > 0.7]
print(f"Features with |ρ| > 0.7: {len(strong_spearman)} / {len(results_df)}")

# Triangulated evidence (all three tests agree)
triangulated = results_df[
    (results_df['p-value'] < 0.05) &
    (results_df['Spearman p'] < 0.05) &
    (results_df['R²'] > 0.3)
]
print(f"Features with triangulated evidence (p<0.05, Spearman p<0.05, R²>0.3): {len(triangulated)}")

# Save full results
output_path = Path(__file__).parent / '../../../output/feature_statistical_summary.csv'
results_df.to_csv(output_path, index=False)
print(f"\n✓ Full results saved to: {output_path}")

# Print key features mentioned in paper
print("\n" + "=" * 120)
print("KEY FEATURES FROM PAPER")
print("=" * 120)
key_features = [
    'subjunctive_full', 'perfect_reduplicated', 'dual_nominative',
    'particle_sma', 'long_compounds', 'philosophical_terms',
    'monophthongs_e', 'retroflex_l', 'deity_names'
]

print(f"{'Feature':<30} {'R²':>8} {'β':>10} {'p-value':>10} {'ρ':>8} {'MK Trend':<12} {'% Change':>10}")
print("-" * 120)

for feature in key_features:
    row = results_df[results_df['feature'] == feature].iloc[0]
    print(f"{row['feature']:<30} {row['R²']:>8.3f} {row['β (slope)']:>10.4f} {row['p-value']:>10.4f} "
          f"{row['Spearman ρ']:>8.3f} {row['MK trend']:<12} {row['% change']:>10.1f}%")

print("\n" + "=" * 120)
