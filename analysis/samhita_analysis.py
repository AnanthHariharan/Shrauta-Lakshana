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
            
        # Basic stats
        total_chars = len(text)
        words = re.findall(r'\b\w+\b', text)
        total_words = len(words)
        
        # Count features
        feature_counts = {}
        for feature_name, pattern in self.features.items():
            matches = re.findall(pattern, text)
            feature_counts[feature_name] = len(matches)
            
        # Store normalized frequencies (per 1000 words)
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

# Usage
analyzer = VedicDiachronicAnalyzer()

# Map your files (adjust paths)
corpus_files = {
    'Rigveda': '../texts/samhita/rig-samhita.txt',
    'Yajurveda': '../texts/samhita/yajur-samhita.txt',
    'Samaveda': '../texts/samhita/sama-samhita.txt',
    'Atharvaveda': '../texts/samhita/atharva-samhita.txt'
}

analyzer.analyze_corpus(corpus_files)
analyzer.generate_report()

# Advanced: track specific verb forms
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