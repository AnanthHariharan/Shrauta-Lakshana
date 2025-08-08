"""
Key metrics:
- Type-Token Ratio (TTR): Lexical diversity measure
- Hapax Legomena: Words occurring only once
- Frequency distributions: Zipfian patterns
- Morphological productivity: Type/token ratios for inflectional patterns
- Vocabulary richness: Unique forms per category
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

class VedicTypeTokenAnalyzer:
    """Enhanced analyzer implementing both type and token frequency analysis"""

    def __init__(self):
        # Enhanced features with type/token distinction
        self.feature_patterns = {
            # MORPHOLOGICAL PATTERNS (Type analysis crucial)
            'subjunctive_morphology': {
                'pattern': r'\b\w+(\u0101ti|\u0101s|\u0101t|\u0101ma|\u0101tana|\u0101n|\u0101ni)\b',
                'extract_stem': True,
                'morphological': True
            },
            'dual_morphology': {
                'pattern': r'\b\w+(au|\u012b|os|\u0101bhy\u0101m)\b',
                'extract_stem': True,
                'morphological': True
            },
            'perfect_morphology': {
                'pattern': r'\b([^aeiou\u0101\u012b\u016b\u0113\u014d])\\1\w+a\\b',
                'extract_stem': True,
                'morphological': True
            },
            'instrumental_archaic': {
                'pattern': r'\b\w+\u0101\b(?!\w)',
                'extract_stem': True,
                'morphological': True
            },
            'genitive_plural': {
                'pattern': r'\b\w+(\u0101n\u0101m|\u0101m)\b',
                'extract_stem': True,
                'morphological': True
            },

            # DERIVATIONAL MORPHOLOGY (High type significance)
            'agent_nouns': {
                'pattern': r'\b\w+(tar|t\u1e5b)\b',
                'extract_stem': True,
                'derivational': True
            },
            'action_nouns': {
                'pattern': r'\b\w+(ti|tu)\b(?!s)',
                'extract_stem': True,
                'derivational': True
            },
            'abstract_nouns': {
                'pattern': r'\b\w+(tva|t\u0101|ya)\b',
                'extract_stem': True,
                'derivational': True
            },
            'primary_derivatives': {
                'pattern': r'\b\w+(ti|tu|tar|man|van)\b',
                'extract_stem': True,
                'derivational': True
            },
            'secondary_derivatives': {
                'pattern': r'\b\w+(tva|t\u0101|ya|ika|\u012bya)\b',
                'extract_stem': True,
                'derivational': True
            },

            # COMPOUND MORPHOLOGY (Type analysis essential)
            'long_compounds': {
                'pattern': r'\b\w{15,}\b',
                'extract_stem': False,
                'compositional': True
            },
            'bahuvrihi_compounds': {
                'pattern': r'\b\w+-(b\u0101hu|hasta|p\u0101da)\w*\b',
                'extract_stem': False,
                'compositional': True
            },

            # LEXICAL CATEGORIES (Type/token both important)
            'deity_names': {
                'pattern': r'\b(indra|agni|soma|varu\u1e47a|mitra|rudra|v\u0101yu|s\u016brya)\b',
                'extract_stem': False,
                'lexical': True
            },
            'ritual_terms': {
                'pattern': r'\b(yaj\u00f1a|hot\u1e5b|brahman|\u1e5btvij|havana|hotra|vedi|y\u016bpa)\b',
                'extract_stem': False,
                'lexical': True
            },
            'philosophical_terms': {
                'pattern': r'\b(\u0101tman|brahman|mok\u1e63a|dharma|karma|sa\u1e43s\u0101ra)\b',
                'extract_stem': False,
                'lexical': True
            },

            # PHONOLOGICAL PATTERNS (Token-focused but type relevant)
            'retroflex_sounds': {
                'pattern': r'\u1e37|\u1e6d|\u1e0d|\u1e47|\u1e63',
                'extract_stem': False,
                'phonological': True
            },
            'diphthongs_ai': {
                'pattern': r'[a\u0101][i\u012b]',
                'extract_stem': False,
                'phonological': True
            },
            'diphthongs_au': {
                'pattern': r'[a\u0101][u\u016b]',
                'extract_stem': False,
                'phonological': True
            },

            # PARTICLES (Token analysis primary)
            'vedic_particles': {
                'pattern': r'\b(sma|ha|vai|id|u|hi)\b',
                'extract_stem': False,
                'syntactic': True
            }
        }

        # Results storage
        self.token_results = defaultdict(lambda: defaultdict(int))
        self.type_results = defaultdict(lambda: defaultdict(int))
        self.vocabulary_data = defaultdict(lambda: defaultdict(set))
        self.frequency_distributions = defaultdict(lambda: defaultdict(Counter))

    def analyze_text_comprehensive(self, filepath, text_name):
        """Comprehensive type-token analysis of a single text"""
        print(f"Analyzing {text_name} (type-token analysis)...")

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().lower()

        # Basic text statistics
        words = re.findall(r'\b\w+\b', text)
        total_words = len(words)
        unique_words = len(set(words))

        # Overall type-token ratio
        ttr_overall = unique_words / total_words if total_words > 0 else 0

        # Word frequency distribution
        word_freq = Counter(words)
        hapax_legomena = len([word for word, freq in word_freq.items() if freq == 1])

        # Store basic metrics
        self.token_results[text_name]['total_tokens'] = total_words
        self.type_results[text_name]['total_types'] = unique_words
        self.type_results[text_name]['ttr_overall'] = ttr_overall
        self.type_results[text_name]['hapax_legomena'] = hapax_legomena
        self.type_results[text_name]['hapax_ratio'] = hapax_legomena / unique_words if unique_words > 0 else 0

        # Analyze each linguistic feature
        for feature_name, feature_config in self.feature_patterns.items():
            pattern = feature_config['pattern']
            extract_stem = feature_config.get('extract_stem', False)

            # Find all matches
            matches = re.findall(pattern, text)

            if extract_stem and matches:
                # For morphological analysis, extract stems
                stems = set()
                all_forms = []

                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match[0] else match[1] if len(match) > 1 else ''

                    if match:
                        # Simple stemming by removing common suffixes
                        stem = self._extract_stem(match)
                        stems.add(stem)
                        all_forms.append(match)

                # Type and token counts
                token_count = len(all_forms)
                type_count = len(stems)

                # Store results
                self.token_results[text_name][feature_name] = token_count
                self.type_results[text_name][feature_name] = type_count

                # Type-token ratio for this feature
                ttr_feature = type_count / token_count if token_count > 0 else 0
                self.type_results[text_name][f'{feature_name}_ttr'] = ttr_feature

                # Store vocabulary for further analysis
                self.vocabulary_data[text_name][feature_name] = stems

                # Frequency distribution
                form_freq = Counter(all_forms)
                self.frequency_distributions[text_name][feature_name] = form_freq

            else:
                # Simple token counting
                token_count = len(matches)
                type_count = len(set(matches)) if matches else 0

                self.token_results[text_name][feature_name] = token_count
                self.type_results[text_name][feature_name] = type_count

                if token_count > 0:
                    ttr_feature = type_count / token_count
                    self.type_results[text_name][f'{feature_name}_ttr'] = ttr_feature

        return {
            'total_tokens': total_words,
            'total_types': unique_words,
            'ttr_overall': ttr_overall,
            'hapax_legomena': hapax_legomena
        }

    def _extract_stem(self, word_form):
        suffixes = [
            r'\u0101ti$', r'\u0101s$', r'\u0101t$', r'\u0101ma$', r'\u0101tana$', r'\u0101n$', r'\u0101ni$',  # Subjunctive
            r'au$', r'\u012b$', r'os$', r'\u0101bhy\u0101m$',  # Dual
            r'tar$', r't\u1e5b$', r'ti$', r'tu$',  # Agent/action nouns
            r'tva$', r't\u0101$', r'ya$',  # Abstract nouns
            r'\u0101$(?!\w)', r'ena$', r'\u0101n\u0101m$', r'\u0101m$'  # Case endings
        ]

        stem = word_form
        for suffix in suffixes:
            stem = re.sub(suffix, '', stem)
            if stem != word_form:
                break

        return stem if stem else word_form

    def calculate_morphological_productivity(self):
        productivity_results = defaultdict(lambda: defaultdict(dict))

        for text_name in self.token_results:
            for feature_name in self.feature_patterns:
                if feature_name in self.token_results[text_name]:
                    tokens = self.token_results[text_name][feature_name]
                    types = self.type_results[text_name][feature_name]

                    if tokens > 0:
                        # Productivity measures
                        ttr = types / tokens

                        # Potential vocabulary (estimated)
                        # Based on Good-Turing estimation
                        potential_vocab = self._estimate_potential_vocabulary(
                            text_name, feature_name
                        )

                        # Realized vs potential productivity
                        realized_productivity = types / potential_vocab if potential_vocab > 0 else 0

                        productivity_results[text_name][feature_name] = {
                            'tokens': tokens,
                            'types': types,
                            'ttr': ttr,
                            'potential_vocab': potential_vocab,
                            'realized_productivity': realized_productivity,
                            'productivity_score': ttr * realized_productivity
                        }

        return productivity_results

    def _estimate_potential_vocabulary(self, text_name, feature_name):
        if feature_name not in self.frequency_distributions[text_name]:
            return 0

        freq_dist = self.frequency_distributions[text_name][feature_name]
        if not freq_dist:
            return 0

        # Simple estimation based on hapax legomena
        hapax_count = len([word for word, freq in freq_dist.items() if freq == 1])
        total_types = len(freq_dist)

        # Good-Turing estimation approximation
        if hapax_count > 0 and total_types > 0:
            estimated_vocab = total_types + (hapax_count * hapax_count / total_types)
        else:
            estimated_vocab = total_types

        return max(estimated_vocab, total_types)

    def calculate_lexical_diversity_measures(self):
        diversity_results = defaultdict(dict)

        for text_name in self.token_results:
            total_tokens = self.token_results[text_name]['total_tokens']
            total_types = self.type_results[text_name]['total_types']

            if total_tokens > 0:
                # Standard Type-Token Ratio
                ttr = total_types / total_tokens

                # Root TTR (less sensitive to text length)
                rttr = total_types / np.sqrt(total_tokens)

                # Corrected TTR (Carrolls)
                cttr = total_types / np.sqrt(2 * total_tokens)

                # Bilogarithmic TTR
                bilttr = np.log(total_types) / np.log(total_tokens) if total_tokens > 1 else 0

                # Uber Index (more stable across text lengths)
                uber = (np.log(total_tokens) ** 2) / (np.log(total_tokens) - np.log(total_types)) if total_types < total_tokens else 0

                # Herdan's C
                herdan_c = np.log(total_types) / np.log(total_tokens) if total_tokens > 1 else 0

                diversity_results[text_name] = {
                    'ttr': ttr,
                    'rttr': rttr,
                    'cttr': cttr,
                    'bilttr': bilttr,
                    'uber': uber,
                    'herdan_c': herdan_c,
                    'total_tokens': total_tokens,
                    'total_types': total_types
                }

        return diversity_results

    def analyze_frequency_distributions(self):
        distribution_results = defaultdict(lambda: defaultdict(dict))

        for text_name in self.frequency_distributions:
            for feature_name, freq_dist in self.frequency_distributions[text_name].items():
                if not freq_dist:
                    continue

                # Frequency statistics
                frequencies = list(freq_dist.values())

                if frequencies:
                    # Basic statistics
                    mean_freq = np.mean(frequencies)
                    median_freq = np.median(frequencies)
                    std_freq = np.std(frequencies)

                    # Zipfian analysis
                    sorted_freqs = sorted(frequencies, reverse=True)
                    ranks = np.arange(1, len(sorted_freqs) + 1)

                    # Log-log regression for Zipf coefficient
                    if len(sorted_freqs) > 2:
                        log_ranks = np.log(ranks)
                        log_freqs = np.log(sorted_freqs)

                        # Linear regression in log space
                        slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)
                        zipf_coefficient = -slope  # Negative because of inverse relationship
                        zipf_r2 = r_value ** 2
                    else:
                        zipf_coefficient = 0
                        zipf_r2 = 0

                    # Entropy (information content)
                    total_tokens = sum(frequencies)
                    probabilities = [f / total_tokens for f in frequencies]
                    shannon_entropy = entropy(probabilities, base=2)

                    # Vocabulary richness indicators
                    hapax_count = len([f for f in frequencies if f == 1])
                    dis_legomena = len([f for f in frequencies if f == 2])

                    distribution_results[text_name][feature_name] = {
                        'mean_frequency': mean_freq,
                        'median_frequency': median_freq,
                        'std_frequency': std_freq,
                        'zipf_coefficient': zipf_coefficient,
                        'zipf_r2': zipf_r2,
                        'shannon_entropy': shannon_entropy,
                        'hapax_count': hapax_count,
                        'dis_legomena': dis_legomena,
                        'hapax_ratio': hapax_count / len(frequencies),
                        'total_types': len(frequencies),
                        'total_tokens': total_tokens
                    }

        return distribution_results

    def compare_type_token_evolution(self, texts_ordered):
        comparison_results = defaultdict(lambda: defaultdict(list))

        # For each feature, track type/token evolution
        for feature_name in self.feature_patterns:
            token_evolution = []
            type_evolution = []
            ttr_evolution = []

            for text_name in texts_ordered:
                if text_name in self.token_results:
                    tokens = self.token_results[text_name].get(feature_name, 0)
                    types = self.type_results[text_name].get(feature_name, 0)
                    ttr = types / tokens if tokens > 0 else 0

                    token_evolution.append(tokens)
                    type_evolution.append(types)
                    ttr_evolution.append(ttr)

            if any(token_evolution):  # Only store if there's data
                comparison_results[feature_name] = {
                    'token_evolution': token_evolution,
                    'type_evolution': type_evolution,
                    'ttr_evolution': ttr_evolution,
                    'texts': texts_ordered
                }

                # Calculate trends
                if len(token_evolution) > 2:
                    # Token trend
                    x = np.arange(len(token_evolution))
                    token_slope, _, token_r, _, _ = stats.linregress(x, token_evolution)
                    type_slope, _, type_r, _, _ = stats.linregress(x, type_evolution)
                    ttr_slope, _, ttr_r, _, _ = stats.linregress(x, ttr_evolution)

                    comparison_results[feature_name]['trends'] = {
                        'token_slope': token_slope,
                        'token_r': token_r,
                        'type_slope': type_slope,
                        'type_r': type_r,
                        'ttr_slope': ttr_slope,
                        'ttr_r': ttr_r
                    }

        return comparison_results

    def generate_comprehensive_report(self, output_file='type_token_analysis_report.txt'):

        # Calculate all metrics
        productivity_results = self.calculate_morphological_productivity()
        diversity_results = self.calculate_lexical_diversity_measures()
        distribution_results = self.analyze_frequency_distributions()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE TYPE-TOKEN FREQUENCY ANALYSIS\n")
            f.write("Vedic Sanskrit Diachronic Study\n")
            f.write("=" * 70 + "\n\n")

            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 50 + "\n")
            f.write("This analysis distinguishes between:\n")
            f.write("• TOKEN FREQUENCY: Total occurrences of patterns\n")
            f.write("• TYPE FREQUENCY: Unique forms exhibiting patterns\n")
            f.write("• TYPE-TOKEN RATIO: Morphological/lexical diversity\n\n")

            # Overall Lexical Diversity
            f.write("OVERALL LEXICAL DIVERSITY\n")
            f.write("-" * 50 + "\n")

            for text_name, metrics in diversity_results.items():
                f.write(f"\n{text_name}:\n")
                f.write(f"  Total Tokens: {metrics['total_tokens']:,}\n")
                f.write(f"  Total Types: {metrics['total_types']:,}\n")
                f.write(f"  TTR: {metrics['ttr']:.4f}\n")
                f.write(f"  Root TTR: {metrics['rttr']:.4f}\n")
                f.write(f"  Herdan's C: {metrics['herdan_c']:.4f}\n")

            # Morphological Productivity
            f.write("\\n\\nMORPHOLOGICAL PRODUCTIVITY ANALYSIS\n")
            f.write("-" * 50 + "\n")

            # Find most productive features
            productivity_summary = defaultdict(list)
            for text_name, features in productivity_results.items():
                for feature_name, metrics in features.items():
                    if 'morphological' in self.feature_patterns.get(feature_name, {}):
                        productivity_summary[feature_name].append(metrics['productivity_score'])

            # Average productivity by feature
            avg_productivity = {feature: np.mean(scores) for feature, scores in productivity_summary.items()}

            f.write("Most productive morphological patterns (average across texts):\n")
            for feature, score in sorted(avg_productivity.items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"  {feature.replace('_', ' ')}: {score:.4f}\n")

            # Feature-specific analysis
            f.write("\\n\\nFEATURE-SPECIFIC TYPE-TOKEN ANALYSIS\n")
            f.write("-" * 50 + "\n")

            important_features = ['subjunctive_morphology', 'dual_morphology', 'agent_nouns',
                                'abstract_nouns', 'long_compounds', 'ritual_terms']

            for feature_name in important_features:
                if any(feature_name in self.token_results[text] for text in self.token_results):
                    f.write(f"\n{feature_name.replace('_', ' ').upper()}:\n")

                    for text_name in self.token_results:
                        if feature_name in self.token_results[text_name]:
                            tokens = self.token_results[text_name][feature_name]
                            types = self.type_results[text_name][feature_name]
                            ttr = types / tokens if tokens > 0 else 0

                            f.write(f"  {text_name:20s}: {tokens:4d} tokens, {types:4d} types, TTR = {ttr:.3f}\n")

            # Frequency Distribution Analysis
            f.write("\\n\\nFREQUENCY DISTRIBUTION ANALYSIS\n")
            f.write("-" * 50 + "\n")

            for text_name, features in distribution_results.items():
                f.write(f"\n{text_name}:\n")

                for feature_name, metrics in features.items():
                    if metrics['total_tokens'] > 10:  # Only show substantial features
                        f.write(f"  {feature_name}:\n")
                        f.write(f"    Zipf coefficient: {metrics['zipf_coefficient']:.3f} (R² = {metrics['zipf_r2']:.3f})\n")
                        f.write(f"    Shannon entropy: {metrics['shannon_entropy']:.3f}\n")
                        f.write(f"    Hapax ratio: {metrics['hapax_ratio']:.3f}\n")

            f.write("\\n\\nMETHODOLOGY\n")
            f.write("-" * 50 + "\n")
            f.write("• Type frequency: Count of unique forms\n")
            f.write("• Token frequency: Count of total occurrences\n")
            f.write("• TTR: Type-Token Ratio (lexical diversity)\n")
            f.write("• Morphological productivity: TTR for inflectional patterns\n")
            f.write("• Zipfian analysis: Frequency distribution patterns\n")
            f.write("• Shannon entropy: Information content measure\\n\n")

            f.write("Type-token analysis reveals morphological productivity patterns\n")
            f.write("and vocabulary richness evolution across Vedic periods.\n")

        print(f"Comprehensive type-token report saved to {output_file}")

class VedicTypeTokenVisualizer:
    """Visualization functions for type-token analysis"""

    def __init__(self, analyzer):
        self.analyzer = analyzer
        plt.style.use("default")
        sns.set_palette("husl")

    def plot_type_token_comparison(self, feature_subset=None):
        """Plot type vs token frequencies for linguistic features"""
        if feature_subset is None:
            feature_subset = ["subjunctive_morphology", "dual_morphology", "agent_nouns",
                            "abstract_nouns", "ritual_terms", "philosophical_terms"]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        for idx, feature in enumerate(feature_subset[:6]):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Collect data for this feature
            texts = []
            tokens = []
            types = []

            for text_name in self.analyzer.token_results:
                if feature in self.analyzer.token_results[text_name]:
                    texts.append(text_name)
                    tokens.append(self.analyzer.token_results[text_name][feature])
                    types.append(self.analyzer.type_results[text_name][feature])

            if tokens and types:
                # Create scatter plot
                colors = ["red"]*5 + ["blue"]*4 + ["green"]*6 + ["orange"]*3  # By period
                ax.scatter(tokens, types, c=colors[:len(tokens)], s=100, alpha=0.7)

                # Add diagonal line (TTR = 1.0)
                max_val = max(max(tokens), max(types))
                ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="TTR = 1.0")

                ax.set_xlabel("Token Frequency")
                ax.set_ylabel("Type Frequency")
                ax.set_title(f"{feature.replace('_', ' ').title()}")
                ax.grid(True, alpha=0.3)
                ax.legend()

        plt.tight_layout()
        plt.savefig("vedic_type_token_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_ttr_evolution(self, text_order):
        """Plot TTR evolution across chronologically ordered texts"""

        # Calculate diversity measures
        diversity_results = self.analyzer.calculate_lexical_diversity_measures()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Overall TTR evolution
        ax1 = axes[0, 0]
        ttrs = [diversity_results[text]["ttr"] for text in text_order if text in diversity_results]
        ax1.plot(range(len(ttrs)), ttrs, "o-", linewidth=2, markersize=8)
        ax1.set_title("Overall Type-Token Ratio Evolution", fontweight="bold")
        ax1.set_ylabel("TTR")
        ax1.set_xticks(range(len(text_order)))
        ax1.set_xticklabels(text_order, rotation=45, ha="right")
        ax1.grid(True, alpha=0.3)

        # Feature-specific TTR trends
        ax2 = axes[0, 1]
        important_features = ["subjunctive_morphology", "dual_morphology", "agent_nouns"]

        for feature in important_features:
            ttrs = []
            for text in text_order:
                if (text in self.analyzer.token_results and
                    feature in self.analyzer.token_results[text]):
                    tokens = self.analyzer.token_results[text][feature]
                    types = self.analyzer.type_results[text][feature]
                    ttr = types / tokens if tokens > 0 else 0
                    ttrs.append(ttr)
                else:
                    ttrs.append(0)

            ax2.plot(range(len(ttrs)), ttrs, "o-", label=feature.replace("_", " "),
                    linewidth=2, alpha=0.8)

        ax2.set_title("Feature-Specific TTR Evolution", fontweight="bold")
        ax2.set_ylabel("TTR")
        ax2.set_xticks(range(len(text_order)))
        ax2.set_xticklabels(text_order, rotation=45, ha="right")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Vocabulary size vs text length
        ax3 = axes[1, 0]
        vocab_sizes = [diversity_results[text]["total_types"] for text in text_order if text in diversity_results]
        text_lengths = [diversity_results[text]["total_tokens"] for text in text_order if text in diversity_results]

        colors = ["red"]*5 + ["blue"]*4 + ["green"]*6 + ["orange"]*3
        ax3.scatter(text_lengths, vocab_sizes, c=colors[:len(vocab_sizes)], s=100, alpha=0.7)

        ax3.set_xlabel("Text Length (tokens)")
        ax3.set_ylabel("Vocabulary Size (types)")
        ax3.set_title("Vocabulary Growth vs Text Length", fontweight="bold")
        ax3.grid(True, alpha=0.3)

        # TTR comparison
        ax4 = axes[1, 1]

        # Create box plot of TTR values
        ttr_values = [diversity_results[text]["ttr"] for text in text_order if text in diversity_results]
        herdan_values = [diversity_results[text]["herdan_c"] for text in text_order if text in diversity_results]

        box_data = [ttr_values, herdan_values]
        ax4.boxplot(box_data, labels=["TTR", "Herdan C"])
        ax4.set_title("Lexical Diversity Measure Comparison", fontweight="bold")
        ax4.set_ylabel("Diversity Value")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("vedic_ttr_evolution.png", dpi=300, bbox_inches="tight")
        plt.show()
