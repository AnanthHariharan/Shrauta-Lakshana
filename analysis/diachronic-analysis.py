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
    'Rigveda': '../texts/samhita/rig-samhita.txt',
    'Yajurveda': '../texts/samhita/yajur-samhita.txt',
    'Samaveda': '../texts/samhita/sama-samhita.txt',
    'Atharvaveda': '../texts/samhita/atharva-samhita.txt'
}

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
    text_order = ['Rigveda', 'Yajurveda', 'Samaveda', 'Atharvaveda']
    df = df.reindex(text_order)
    features_to_plot = ['retroflex_l', 'subjunctive_ati', 'particle_sma',
                       'long_compounds', 'dual_forms']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, feature in enumerate(features_to_plot):
        if idx < len(axes):
            ax = axes[idx]
            values = df[feature].values
            ax.plot(text_order, values, marker='o', linewidth=2)
            ax.set_title(f'{feature} frequency over time')
            ax.set_ylabel('Freq per 1000 words')
            ax.tick_params(axis='x', rotation=45)

            # Add trend line
            x_numeric = np.arange(len(text_order))
            z = np.polyfit(x_numeric, values, 1)
            p = np.poly1d(z)
            ax.plot(text_order, p(x_numeric), "r--", alpha=0.5)

    plt.tight_layout()
    plt.savefig('vedic_diachronic_trends.png')
    plt.show()

def statistical_analysis(analyzer):
    """Test if changes between texts are significant"""
    results = []

    texts = ['Rigveda', 'Yajurveda', 'Samaveda', 'Atharvaveda']
    features = ['subjunctive_ati', 'particle_sma', 'retroflex_l']

    for feature in features:
        print(f"\nStatistical tests for {feature}:")
        print("-" * 40)

        for i in range(len(texts)-1):
            text1, text2 = texts[i], texts[i+1]

            freq1 = analyzer.results[text1][feature]
            freq2 = analyzer.results[text2][feature]

            print(f"{text1} vs {text2}:")
            print(f"  {text1}: {freq1:.2f} per 1000 words")
            print(f"  {text2}: {freq2:.2f} per 1000 words")

            if freq1 == 0:
                if freq2 == 0:
                    print(f"  Change: No occurrences in either text")
                else:
                    print(f"  Change: New feature (0 → {freq2:.2f})")
            else:
                change = ((freq2-freq1)/freq1)*100
                print(f"  Change: {change:+.1f}%")

def export_results(analyzer, filename='vedic_analysis.csv'):
    """Export results for further analysis"""
    df = pd.DataFrame(analyzer.results).T

    df['text_period'] = ['Early Vedic', 'Middle Vedic', 'Middle Vedic', 'Late Vedic']
    df['approx_date_bce'] = [1500, 1200, 1000, 800]

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
        f.write("VEDIC SANSKRIT DIACHRONIC ANALYSIS\n")
        f.write("="*50 + "\n\n")

        f.write("Key Findings:\n")
        f.write("-"*30 + "\n")

        texts = ['Rigveda', 'Yajurveda', 'Samaveda', 'Atharvaveda']

        for feature in ['subjunctive_ati', 'retroflex_l', 'long_compounds']:
            values = [analyzer.results[t][feature] for t in texts]

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
