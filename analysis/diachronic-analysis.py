import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np
import re
from collections import defaultdict, Counter
from pathlib import Path

class VedicDiachronicAnalyzer:
    def __init__(self):
        self.features = {
            'retroflex_l': r'ḷ',
            'subjunctive_ati': r'\b\w+āti\b',
            'subjunctive_an': r'\b\w+ān\b',
            'particle_sma': r'\bsma\b',
            'particle_ha': r'\bha\b',
            'particle_vai': r'\bvai\b',
            'long_compounds': r'\b\w{15,}\b',  # 15+ character words
            'aorist_is': r'\b\w+īṣ\b|\b\w+iṣ\b',
            'infinitive_tum': r'\b\w+tum\b',
            'gerund_tvaa': r'\b\w+tvā\b'
        }
        self.results = defaultdict(lambda: defaultdict(int))

    def analyze_file(self, filepath, text_name):
        """Analyze a single text file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().lower()

        total_chars = len(text)
        words = re.findall(r'\b\w+\b', text)
        total_words = len(words)

        feature_counts = {}
        for feature_name, pattern in self.features.items():
            matches = re.findall(pattern, text)
            feature_counts[feature_name] = len(matches)

        for feature, count in feature_counts.items():
            self.results[text_name][feature] = (count / total_words) * 1000

        self.results[text_name]['total_words'] = total_words

    def analyze_corpus(self, file_mapping):
        """Analyze multiple files
        file_mapping: dict of {text_name: filepath}"""
        for text_name, filepath in file_mapping.items():
            print(f"Analyzing {text_name}...")
            self.analyze_file(filepath, text_name)

    def generate_report(self):
        """Print comparative analysis"""
        print("\nDiachronic Analysis Results")
        print("=" * 60)

        # Header
        texts = list(self.results.keys())
        print(f"{'Feature':<20}", end='')
        for text in texts:
            print(f"{text:<15}", end='')
        print()
        print("-" * 60)

        # Feature frequencies
        for feature in self.features:
            print(f"{feature:<20}", end='')
            for text in texts:
                freq = self.results[text][feature]
                print(f"{freq:>14.2f}", end='')
            print()

        print("\nTotal words analyzed:")
        for text in texts:
            print(f"{text}: {self.results[text]['total_words']:,}")

analyzer = VedicDiachronicAnalyzer()

corpus_files = {
    # Samhitas (Early to Middle Vedic)
    'Rigveda': '../texts/samhita/rig-samhita.txt',
    'Yajurveda': '../texts/samhita/yajur-samhita.txt',
    'Samaveda': '../texts/samhita/sama-samhita.txt',
    'Atharvaveda': '../texts/samhita/atharva-samhita.txt',

    # Brahmanas (Late Vedic)
    'Kausitaki-Br': '../texts/brahmana/rig-kausitaki.txt',
    'Pancavimsa-Br': '../texts/brahmana/sama-pancavimsa.txt',
    'Satapatha-Br': '../texts/brahmana/yajur-satapatha.txt',
    'Gopatha-Br': '../texts/brahmana/atharva-gopatha.txt',

    # Upanishads (Latest Vedic)
    'Brhadaranyaka-Up': '../texts/upanishad/yajur-brhadaranyaka.txt',
    'Chandogya-Up': '../texts/upanishad/sama-chandogya.txt',
    'Aitareya-Up': '../texts/upanishad/rig-aitareya.txt',
    'Mandukya-Up': '../texts/upanishad/atharva-mandukya.txt'
}

# Add this line after corpus_files:
text_order = list(corpus_files.keys())

analyzer.analyze_corpus(corpus_files)
analyzer.generate_report()

class VerbFormAnalyzer:
    def __init__(self):
        self.verb_endings = {
            # Present
            'pres_3sg': r'\b\w+ati\b',
            'pres_3pl': r'\b\w+anti\b',
            # Perfect
            'perf_3sg': r'\b\w+a\b(?!ti|nti)',  # ends in 'a' but not present
            # Aorist
            'aor_3sg': r'\b\w+īt\b|\b\w+at\b',
            # Subjunctive (declining feature)
            'subj_forms': r'\b\w+āti\b|\b\w+ān\b|\b\w+āt\b'
        }

    def count_verb_forms(self, text):
        forms_count = Counter()
        for form_type, pattern in self.verb_endings.items():
            matches = re.findall(pattern, text.lower())
            forms_count[form_type] = len(matches)
        return forms_count

class EnhancedVedicAnalyzer(VedicDiachronicAnalyzer):
    def __init__(self):
        super().__init__()
        self.morphological_features = {
            # Vedic-specific verb forms
            'injunctive': r'\b\w+at\b(?!i\b)',  # past without augment
            'precative': r'\b\w+yās\b|\b\w+yāt\b',
            'benedictive': r'\b\w+yāsam\b',

            # Nominal patterns
            'instrumental_a': r'\b\w+ā\b',  # archaic instrumental
            'gen_plural_am': r'\b\w+ām\b',
            'dual_forms': r'\b\w+au\b|\b\w+ī\b(?=\s)',

            # Vedic particles/adverbs
            'particle_id': r'\bid\b',
            'particle_u': r'\bu\b',
            'adverb_tra': r'\b\w+tra\b',  # locative suffix
        }
        self.features.update(self.morphological_features)
        def plot_diachronic_trends(analyzer):
            """Create visualization of feature changes"""
            df = pd.DataFrame(analyzer.results).T
            # Change this line:
            # text_order = ['Rigveda', 'Yajurveda', 'Samaveda', 'Atharvaveda']
            df = df.reindex(text_order)  # Use the global text_order

            # Update features to plot if needed
            features_to_plot = ['retroflex_l', 'subjunctive_ati', 'particle_sma',
                               'long_compounds', 'infinitive_tum', 'gerund_tvaa']

            # Change subplot layout to accommodate more features
            fig, axes = plt.subplots(3, 2, figsize=(15, 18))
            axes = axes.ravel()

            for idx, feature in enumerate(features_to_plot):
                if idx < len(axes):
                    ax = axes[idx]
                    values = df[feature].values

                    # Color code by text type
                    colors = ['red']*4 + ['blue']*4 + ['green']*4
                    ax.scatter(range(len(text_order)), values, c=colors, s=100, alpha=0.6)
                    ax.plot(range(len(text_order)), values, 'k-', alpha=0.3)

                    ax.set_title(f'{feature} frequency evolution')
                    ax.set_ylabel('Freq per 1000 words')
                    ax.set_xticks(range(len(text_order)))
                    ax.set_xticklabels(text_order, rotation=45, ha='right')

                    # Add period shading
                    ax.axvspan(-0.5, 3.5, alpha=0.1, color='red')
                    ax.axvspan(3.5, 7.5, alpha=0.1, color='blue')
                    ax.axvspan(7.5, 11.5, alpha=0.1, color='green')

                    # Trend line
                    x_numeric = np.arange(len(text_order))
                    z = np.polyfit(x_numeric, values, 2)
                    p = np.poly1d(z)
                    ax.plot(x_numeric, p(x_numeric), "r--", alpha=0.5)

            plt.tight_layout()
            plt.savefig('vedic_full_diachronic_trends.png', dpi=300)
            plt.show()

def plot_diachronic_trends(analyzer):
    """Create visualization of feature changes"""
    df = pd.DataFrame(analyzer.results).T
    # Change this line:
    # text_order = ['Rigveda', 'Yajurveda', 'Samaveda', 'Atharvaveda']
    df = df.reindex(text_order)  # Use the global text_order

    # Update features to plot if needed
    features_to_plot = ['retroflex_l', 'subjunctive_ati', 'particle_sma',
                       'long_compounds', 'infinitive_tum', 'gerund_tvaa']

    # Change subplot layout to accommodate more features
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.ravel()

    for idx, feature in enumerate(features_to_plot):
        if idx < len(axes):
            ax = axes[idx]
            values = df[feature].values

            # Color code by text type
            colors = ['red']*4 + ['blue']*4 + ['green']*4
            ax.scatter(range(len(text_order)), values, c=colors, s=100, alpha=0.6)
            ax.plot(range(len(text_order)), values, 'k-', alpha=0.3)

            ax.set_title(f'{feature} frequency evolution')
            ax.set_ylabel('Freq per 1000 words')
            ax.set_xticks(range(len(text_order)))
            ax.set_xticklabels(text_order, rotation=45, ha='right')

            # Add period shading
            ax.axvspan(-0.5, 3.5, alpha=0.1, color='red')
            ax.axvspan(3.5, 7.5, alpha=0.1, color='blue')
            ax.axvspan(7.5, 11.5, alpha=0.1, color='green')

            # Trend line
            x_numeric = np.arange(len(text_order))
            z = np.polyfit(x_numeric, values, 2)
            p = np.poly1d(z)
            ax.plot(x_numeric, p(x_numeric), "r--", alpha=0.5)

    plt.tight_layout()
    plt.savefig('vedic_full_diachronic_trends.png', dpi=300)
    plt.show()

def statistical_analysis(analyzer):
    """Test if changes between texts are significant"""
    print("\nStatistical Analysis by Period")
    print("=" * 60)

    # Change from individual texts to periods
    periods = {
        'Samhita': text_order[:4],
        'Brahmana': text_order[4:8],
        'Upanishad': text_order[8:]
    }

    features = ['subjunctive_ati', 'particle_sma', 'retroflex_l', 'long_compounds']

    for feature in features:
        print(f"\n{feature}:")
        print("-" * 40)

        for period, texts in periods.items():
            values = [analyzer.results[text][feature] for text in texts]
            avg = np.mean(values)
            std = np.std(values)
            print(f"{period}: {avg:.2f} ± {std:.2f}")

        # Calculate overall change
        samhita_avg = np.mean([analyzer.results[t][feature] for t in periods['Samhita']])
        upanishad_avg = np.mean([analyzer.results[t][feature] for t in periods['Upanishad']])

        if samhita_avg > 0:
            change = ((upanishad_avg - samhita_avg) / samhita_avg) * 100
            print(f"Change from Samhita to Upanishad: {change:+.1f}%")


def export_results(analyzer, filename='vedic_analysis.csv'):
    """Export results for further analysis"""
    df = pd.DataFrame(analyzer.results).T

    periods = ['Early Vedic']*4 + ['Late Vedic']*4 + ['Latest Vedic']*4
    dates = [1500, 1300, 1200, 1000] + [900, 850, 800, 750] + [700, 650, 600, 550]

    df['text_period'] = periods
    df['approx_date_bce'] = dates

    df['archaism_index'] = (df['retroflex_l'] + df['subjunctive_ati'] +
                            df['particle_sma']) / 3
    df['innovation_index'] = (df['long_compounds'] + df['infinitive_tum']) / 2

    df.to_csv(filename)
    print(f"Results exported to {filename}")

    features_df = df[analyzer.features.keys()].T
    features_df.to_csv('features_over_time.csv')

def analyze_with_sanskrit_tools(text_file):
    try:
        from sanskrit_parser import Parser
        from indic_transliteration import sanscript

        parser = Parser()

        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split into words (basic)
        words = text.split()

        parsed_results = []
        for word in words[:100]:
            try:
                word_dev = sanscript.transliterate(word, sanscript.IAST,
                                                   sanscript.DEVANAGARI)
                splits = parser.split(word_dev)
                if splits:
                    parsed_results.append({
                        'word': word,
                        'splits': splits[0]  # Take first split
                    })
            except:
                continue

        return parsed_results

    except ImportError:
        print("Install sanskrit_parser: pip install sanskrit_parser")
        return None

analyzer = EnhancedVedicAnalyzer()
analyzer.analyze_corpus(corpus_files)

plot_diachronic_trends(analyzer)

statistical_analysis(analyzer)

export_results(analyzer)

def create_summary_report(analyzer):
    with open('vedic_diachronic_report.txt', 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE VEDIC SANSKRIT DIACHRONIC ANALYSIS\n")
        f.write("="*50 + "\n\n")

        f.write("Corpus:\n")
        f.write("-"*30 + "\n")
        f.write("Samhitas: RV, YV, SV, AV\n")
        f.write("Brahmanas: Kausitaki, Pancavimsa, Satapatha, Gopatha\n")
        f.write("Upanishads: Brhadaranyaka, Chandogya, Aitareya, Mandukya\n\n")

        f.write("Key Findings:\n")
        f.write("-"*30 + "\n")

        # Use text_order instead of hardcoded list
        for feature in ['subjunctive_ati', 'retroflex_l', 'long_compounds']:
            values = [analyzer.results[t][feature] for t in text_order]

            f.write(f"\n{feature}:\n")

            # Handle zero values
            if values[0] == 0:
                if all(v == 0 for v in values):
                    f.write(f"  No occurrences in any text\n")
                else:
                    f.write(f"  First appears in {texts[values.index(next(v for v in values if v > 0))]}\n")
            else:
                trend = "decreasing" if values[0] > values[-1] else "increasing"
                change = ((values[-1] - values[0]) / values[0]) * 100
                f.write(f"  Trend: {trend}\n")
                f.write(f"  Change: {change:+.1f}% from RV to AV\n")

            f.write(f"  Values: {' → '.join([f'{v:.2f}' for v in values])}\n")

create_summary_report(analyzer)
