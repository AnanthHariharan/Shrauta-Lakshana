#!/usr/bin/env python3
"""
Plot Individual Feature Trends (Per Dr. Mortensen's Feedback)
==============================================================

Instead of aggregating features into categories, this script plots individual
linguistic features as separate lines to show their distinct evolutionary paths.

Author: Generated for Shrauta-Lakshana project
Usage: python plot_individual_feature_trends.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 9

# Text order (chronological) - UPDATED with all texts
TEXT_ORDER = [
    "Rigveda",
    "Samaveda",
    "Yajurveda (Taittiriya)",
    "Yajurveda (Maitrayani)",
    "Atharvaveda (Paippalada)",
    "Atharvaveda (Saunaka)",
    "Kausitaki-Br",
    "Pancavimsa-Br",
    "Taittiriya-Br",
    "Satapatha-Br",
    "Gopatha-Br",
    "Aitareya-Up",
    "Taittiriya-Up",
    "Chandogya-Up",
    "Brhadaranyaka-Up",
    "Prashna-Up",
    "Shvetashvatara-Up",
    "Ramayana",
    "Mahabharata",
    "Bhagavata-Purana",
]

# Shortened labels for x-axis
TEXT_LABELS = [
    "RV",
    "SV",
    "YV-T",
    "YV-M",
    "AV-P",
    "AV-S",
    "Kaus-Br",
    "Panc-Br",
    "Tait-Br",
    "Sata-Br",
    "Gop-Br",
    "Ait-Up",
    "Tait-Up",
    "Chan-Up",
    "Brh-Up",
    "Pra-Up",
    "Shv-Up",
    "Ram",
    "Mbh",
    "BhP",
]


def load_data(csv_path=None):
    """Load the analysis results"""
    if csv_path is None:
        # Auto-detect path relative to script location
        script_dir = Path(__file__).parent
        csv_path = script_dir / "../../.." / "output" / "vedic_analysis.csv"

    csv_path = Path(csv_path).resolve()
    print(f"   Loading from: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            "Please run the diachronic analysis first to generate vedic_analysis.csv"
        )

    df = pd.read_csv(csv_path, index_col=0)
    # Reorder by chronology
    df = df.reindex(TEXT_ORDER)
    return df


# def add_period_shading(ax):
#     """Add background shading for historical periods"""
#     for period, (start, end) in PERIOD_BOUNDARIES.items():
#         ax.axvspan(
#             start,
#             end,
#             alpha=0.15,
#             color=PERIOD_COLORS[period],
#             label=period if start == 0 else "",
#         )


def plot_morphological_features(df, output_dir=None):
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "../../.." / "output"
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Individual Morphological Feature Trends",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Panel 1: Archaic Verbal Features
    ax = axes[0, 0]
    features = ["subjunctive_full", "perfect_reduplicated", "injunctive_modal"]
    colors = ["#d62728", "#ff7f0e", "#8c564b"]
    markers = ["o", "s", "^"]

    for feat, color, marker in zip(features, colors, markers):
        if feat in df.columns:
            values = df[feat].values
            ax.plot(
                range(len(TEXT_ORDER)),
                values,
                label=feat.replace("_", " ").title(),
                color=color,
                linewidth=2,
                marker=marker,
                markersize=5,
                alpha=0.8,
            )

    # add_period_shading(ax)
    ax.set_title("Archaic Verbal Morphology", fontweight="bold")
    ax.set_ylabel("Frequency (per 1,000 words)", fontsize=11)
    ax.set_xticks(range(len(TEXT_ORDER)))
    ax.set_xticklabels(TEXT_LABELS, rotation=45, ha="right")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Panel 2: Archaic Nominal Features
    ax = axes[0, 1]
    features = ["dual_nominative", "dual_instrumental", "particle_sma"]
    colors = ["#2ca02c", "#1f77b4", "#9467bd"]
    markers = ["D", "v", "p"]

    for feat, color, marker in zip(features, colors, markers):
        if feat in df.columns:
            values = df[feat].values
            ax.plot(
                range(len(TEXT_ORDER)),
                values,
                label=feat.replace("_", " ").title(),
                color=color,
                linewidth=2,
                marker=marker,
                markersize=5,
                alpha=0.8,
            )

    # add_period_shading(ax)
    ax.set_title("Archaic Nominal Features & Particles", fontweight="bold")
    ax.set_ylabel("Frequency (per 1,000 words)", fontsize=11)
    ax.set_xticks(range(len(TEXT_ORDER)))
    ax.set_xticklabels(TEXT_LABELS, rotation=45, ha="right")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Panel 3: Innovative Features
    ax = axes[1, 0]
    features = ["long_compounds", "infinitive_tum", "subordinators"]
    colors = ["#e377c2", "#bcbd22", "#17becf"]
    markers = ["o", "s", "^"]

    for feat, color, marker in zip(features, colors, markers):
        if feat in df.columns:
            values = df[feat].values
            ax.plot(
                range(len(TEXT_ORDER)),
                values,
                label=feat.replace("_", " ").title(),
                color=color,
                linewidth=2,
                marker=marker,
                markersize=5,
                alpha=0.8,
            )

    # add_period_shading(ax)
    ax.set_title("Innovative Morphological & Syntactic Features", fontweight="bold")
    ax.set_ylabel("Frequency (per 1,000 words)", fontsize=11)
    ax.set_xticks(range(len(TEXT_ORDER)))
    ax.set_xticklabels(TEXT_LABELS, rotation=45, ha="right")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Panel 4: Lexical Features
    ax = axes[1, 1]
    features = ["philosophical_terms", "ritual_terminology", "deity_names"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    markers = ["D", "v", "p"]

    for feat, color, marker in zip(features, colors, markers):
        if feat in df.columns:
            values = df[feat].values
            ax.plot(
                range(len(TEXT_ORDER)),
                values,
                label=feat.replace("_", " ").title(),
                color=color,
                linewidth=2,
                marker=marker,
                markersize=5,
                alpha=0.8,
            )

    # add_period_shading(ax)
    ax.set_title("Lexical Evolution", fontweight="bold")
    ax.set_ylabel("Frequency (per 1,000 words)", fontsize=11)
    ax.set_xticks(range(len(TEXT_ORDER)))
    ax.set_xticklabels(TEXT_LABELS, rotation=45, ha="right")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    output_path = Path(output_dir) / "vedic_individual_feature_trends.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_phonological_features(df, output_dir=None):
    """
    Plot phonological features individually
    """
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "../../.." / "output"
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Individual Phonological Feature Trends", fontsize=16, fontweight="bold", y=1.00
    )

    # Panel 1: Archaic sounds
    ax = axes[0]
    features = ["retroflex_l", "pluti_vowels", "medial_voiced_aspirates"]
    colors = ["#d62728", "#ff7f0e", "#8c564b"]
    markers = ["o", "s", "^"]

    for feat, color, marker in zip(features, colors, markers):
        if feat in df.columns:
            values = df[feat].values
            ax.plot(
                range(len(TEXT_ORDER)),
                values,
                label=feat.replace("_", " ").title(),
                color=color,
                linewidth=2,
                marker=marker,
                markersize=5,
                alpha=0.8,
            )

    # add_period_shading(ax)
    ax.set_title("Archaic Phonological Features", fontweight="bold")
    ax.set_ylabel("Frequency (per 1,000 words)", fontsize=11)
    ax.set_xticks(range(len(TEXT_ORDER)))
    ax.set_xticklabels(TEXT_LABELS, rotation=45, ha="right")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Panel 2: Vowel evolution
    ax = axes[1]
    features = ["diphthongs_ai", "diphthongs_au", "monophthongs_e", "monophthongs_o"]
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#e377c2"]
    markers = ["o", "s", "^", "D"]

    for feat, color, marker in zip(features, colors, markers):
        if feat in df.columns:
            values = df[feat].values
            ax.plot(
                range(len(TEXT_ORDER)),
                values,
                label=feat.replace("_", " ").title(),
                color=color,
                linewidth=2,
                marker=marker,
                markersize=5,
                alpha=0.8,
            )

    # add_period_shading(ax)
    ax.set_title("Vowel System Evolution", fontweight="bold")
    ax.set_ylabel("Frequency (per 1,000 words)", fontsize=11)
    ax.set_xticks(range(len(TEXT_ORDER)))
    ax.set_xticklabels(TEXT_LABELS, rotation=45, ha="right")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    output_path = Path(output_dir) / "vedic_phonological_individual_trends.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_key_table4_features(df, output_dir=None):
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "../../.." / "output"
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    features = [
        "subjunctive_full",
        "perfect_reduplicated",
        "dual_nominative",
        "particle_sma",
        "long_compounds",
        "philosophical_terms",
        "monophthongs_e",
    ]

    colors = [
        "#d62728",
        "#ff7f0e",
        "#2ca02c",
        "#1f77b4",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]
    markers = ["o", "s", "^", "D", "v", "p", "h"]

    for feat, color, marker in zip(features, colors, markers):
        if feat in df.columns:
            values = df[feat].values
            label = feat.replace("_", " ").title()
            ax.plot(
                range(len(TEXT_ORDER)),
                values,
                label=label,
                color=color,
                linewidth=2.5,
                marker=marker,
                markersize=6,
                alpha=0.85,
            )

    # add_period_shading(ax)
    # ax.set_title('Key Diachronic Features (Table 4)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Text (Chronological Order)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency (per 1,000 words)", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(TEXT_ORDER)))
    ax.set_xticklabels(TEXT_LABELS, rotation=45, ha="right")
    ax.legend(
        loc="upper left", ncol=2, framealpha=0.95, edgecolor="black", fancybox=True
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add period labels at top
    # for period, (start, end) in PERIOD_BOUNDARIES.items():
    #     mid = (start + end) / 2
    #     ax.text(
    #         mid,
    #         ax.get_ylim()[1] * 1.02,
    #         period,
    #         ha="center",
    #         va="bottom",
    #         fontsize=9,
    #         fontweight="bold",
    #         alpha=0.7,
    #     )

    plt.tight_layout()
    output_path = Path(output_dir) / "vedic_table4_feature_trends.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_unexpected_trends(df, output_dir=None):
    """
    Highlight the unexpected subjunctive increase mentioned in your paper
    """
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "../../.." / "output"
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Subjunctive detailed breakdown
    ax = axes[0]
    subjunctive_features = [col for col in df.columns if "subjunctive" in col.lower()]

    # --- rainbow-like colormap (HSV cycles through the color wheel) ---
    colors = plt.cm.tab20(np.linspace(0, 1, len(subjunctive_features)))

    # Expanded marker set for better contrast
    markers = ["o", "s", "^", "D", "v", "<", ">", "P", "X"]

    for i, feat in enumerate(subjunctive_features):
        values = df[feat].values
        label = feat.replace("_", " ").title()
        marker = markers[i % len(markers)]
        ax.plot(
            range(len(TEXT_ORDER)),
            values,
            label=label,
            color=colors[i],
            linewidth=2.5,
            marker=marker,
            markersize=7,
            alpha=0.95,
        )

    # add_period_shading(ax)
    ax.set_title("Unexpected Subjunctive Increase", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency (per 1,000 words)", fontsize=11)
    ax.set_xticks(range(len(TEXT_ORDER)))
    ax.set_xticklabels(TEXT_LABELS, rotation=45, ha="right")
    ax.legend(loc="upper left", framealpha=0.95, ncol=1)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Panel 2: Contrast with expected declining features
    ax = axes[1]
    declining = ["particle_sma", "retroflex_l", "dual_nominative"]

    # Keep high-contrast colors here
    colors_dec = ["#d62728", "#ff7f0e", "#2ca02c"]
    markers_dec = ["o", "s", "^"]

    for feat, color, marker in zip(declining, colors_dec, markers_dec):
        if feat in df.columns:
            values = df[feat].values
            label = feat.replace("_", " ").title()
            ax.plot(
                range(len(TEXT_ORDER)),
                values,
                label=label,
                color=color,
                linewidth=2.5,
                marker=marker,
                markersize=7,
                alpha=0.95,
            )

    # add_period_shading(ax)
    ax.set_title("Expected Declining Archaic Features", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency (per 1,000 words)", fontsize=11)
    ax.set_xticks(range(len(TEXT_ORDER)))
    ax.set_xticklabels(TEXT_LABELS, rotation=45, ha="right")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    output_path = Path(output_dir) / "vedic_unexpected_trends.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Main execution"""
    print("=" * 70)
    print("PLOTTING INDIVIDUAL FEATURE TRENDS")
    print("(Per Dr. Mortensen's Feedback)")
    print("=" * 70)

    # Load data
    print("\n📊 Loading data from vedic_analysis.csv...")
    df = load_data()
    print(f"   Loaded {len(df)} texts with {len(df.columns)} features")

    # Create output directory if needed
    script_dir = Path(__file__).parent
    output_dir = (script_dir / "../../.." / "output").resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"   Output directory: {output_dir}")

    # Generate plots
    print("\n🎨 Generating plots...")
    print("-" * 70)

    print("\n1. Main morphological features (4 panels):")
    plot_morphological_features(df, output_dir)

    print("\n2. Phonological features (2 panels):")
    plot_phonological_features(df, output_dir)

    print("\n3. Key Table 4 features (single combined plot):")
    plot_key_table4_features(df, output_dir)

    print("\n4. Unexpected trends (subjunctive increase):")
    plot_unexpected_trends(df, output_dir)

    print("\n" + "=" * 70)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nOutput files saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  • vedic_individual_feature_trends.png")
    print("  • vedic_phonological_individual_trends.png")
    print("  • vedic_table4_feature_trends.png")
    print("  • vedic_unexpected_trends.png")
    print("\nThese replace the old aggregated vedic_category_trends.png")
    print("\n💡 TIP: Use vedic_table4_feature_trends.png in your paper")
    print("   as it shows the specific features you discuss in Table 4.")


if __name__ == "__main__":
    main()
