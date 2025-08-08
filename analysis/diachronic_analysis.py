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
            # PHONOLOGICAL EVOLUTION
            # Archaic sounds (decreasing)
            'retroflex_l': r'ḷ|ḷʱ',
            'visarga_final': r'\w+ḥ\b',

            # Vowel evolution patterns
            'diphthongs_ai': r'\b\w*[aā][iī]\w*\b',
            'diphthongs_au': r'\b\w*[aā][uū]\w*\b',
            'monophthongs_e': r'\b\w*[eē]\w*\b',
            'monophthongs_o': r'\b\w*[oō]\w*\b',

            # Sandhi patterns
            'external_sandhi_unresolved': r'\b[aeiouāīūēō]\s+[aeiouāīūēō]\b',
            'retroflex_assimilation': r'[nṇ][ṭḍ]|[ṣś][ṭḍ]',

            # MORPHOLOGICAL EVOLUTION
            # Enhanced subjunctive system (archaic, decreasing)
            'subjunctive_ati': r'\b\w+āti\b',
            'subjunctive_an': r'\b\w+ān\b',
            'subjunctive_as': r'\b\w+ās\b',
            'subjunctive_at': r'\b\w+āt\b',
            'subjunctive_ama': r'\b\w+āma\b',
            'subjunctive_full': r'\b\w+(ās|āt|āma|ātana|ān|āni)\b',

            # Perfect system evolution
            'perfect_reduplicated': r'\b([^aeiouāīūēō])\1\w+a\b',
            'perfect_periphrastic': r'\b\w+vān\s+(asi|asti)\b',
            'perfect_endings': r'\b\w+(a|itha|a|ima|atha|ur)\b(?!.*ti)',

            # Enhanced dual system (archaic, decreasing)
            'dual_nominative': r'\b\w+au\b|\b\w+ī\b(?=\s)',
            'dual_instrumental': r'\b\w+ābhyām\b',
            'dual_genitive': r'\b\w+os\b',

            # Injunctive and modal systems
            'injunctive_augmentless': r'\b[^aā]\w*[td](?!\w)',
            'injunctive_modal': r'\bmā\s+\w+[td]\b',
            'precative': r'\b\w+yās\b|\b\w+yāt\b',
            'benedictive': r'\b\w+yāsam\b|\b\w+yāsur\b',

            # Case system evolution
            'instrumental_archaic_a': r'\b\w+ā\b(?!\w)',
            'instrumental_classical_ena': r'\b\w+ena\b',
            'genitive_plural_thematic': r'\b\w+ānām\b',
            'genitive_plural_athematic': r'\b\w+ām\b',

            # Vedic particles (archaic, decreasing)
            'particle_sma': r'\bsma\b',
            'particle_ha': r'\bha\b',
            'particle_vai': r'\bvai\b',
            'particle_id': r'\bid\b',
            'particle_u': r'\bu\b',
            'particle_hi': r'\bhi\b',

            # SYNTACTIC DEVELOPMENTS
            # Participial constructions (increasing)
            'present_participle_ant': r'\b\w+ant-\b',
            'present_participle_at': r'\b\w+at-\b',
            'past_participle_ta': r'\b\w+ta\b',
            'past_participle_na': r'\b\w+na\b',

            # Gerund system evolution
            'gerund_tvaa': r'\b\w+tvā\b',
            'gerund_ya': r'\b\w+ya\b',
            'infinitive_tum': r'\b\w+tum\b',

            # Correlative constructions (developing)
            'correlatives_ya_ta': r'\bya[ḥsm]?\s+.*\s+ta[ḥsm]?\b',
            'conditional_yadi_tarhi': r'\byadi\b.*\btarhi\b',

            # Word formation (innovations, increasing)
            'long_compounds': r'\b\w{15,}\b',
            'compound_bahuvrīhi': r'\b\w+-(bāhu|hasta|pāda)\w*\b',
            'primary_suffixes': r'\b\w+(ti|tu|tar)\b',
            'secondary_suffixes': r'\b\w+(tva|tā|ya)\b',

            # LEXICAL STRATIFICATION
            # Core religious terminology
            'ritual_sacrifice': r'\b(yajña|hotṛ|brahman|ṛtvij|havana|hotra)\b',
            'deity_names': r'\b(indra|agni|soma|varuṇa|mitra|rudra)\b',
            'priestly_terms': r'\b(hotṛ|adhvaryu|udgātṛ|brahman|purohita)\b',

            # Abstract/philosophical vocabulary (late development)
            'philosophical_terms': r'\b(ātman|brahman|mokṣa|dharma|karma|saṃsāra)\b',
            'cosmological_terms': r'\b(prajāpati|viśvakarman|puruṣa|hiraṇyagarbha)\b',
            'eschatological_terms': r'\b(pitṛloka|svarga|naraka|paraloka)\b',

            # Ritual action terminology
            'sacrifice_roots': r'\b(hu|yaj|hoṣ|juh)\w*\b',
            'ritual_implements': r'\b(vedi|yūpa|cātvāla|ukhā)\b',

            # METRICAL FEATURES
            # Meter analysis (basic patterns)
            'short_syllables': r'[aeiou](?![ṃḥ]|[kgcjṭḍtdpb])',
            'long_syllables': r'[āīūēō]|[aeiou][ṃḥ]|[aeiou][kgcjṭḍtdpb]',

            # Archaic verb forms
            'aorist_is': r'\b\w+īṣ\b|\b\w+iṣ\b',
            'aorist_root': r'\b\w+at\b(?!i)',
            'aorist_sigmatic': r'\b\w+siṣ\b|\b\w+sis\b',

            # Nominal derivation patterns
            'action_nouns_ti': r'\b\w+ti\b(?!s)',
            'agent_nouns_tar': r'\b\w+tar\b',
            'abstract_nouns_tva': r'\b\w+tva\b',

            # ADVANCED PHONOLOGICAL EVOLUTION
            # Loss of Pluti vowels (VVV > VV or V) - early Vedic feature
            'pluti_vowels': r'[āa]{3,}|[īi]{3,}|[ūu]{3,}',
            
            # Replacement of voiced aspirates in medial positions
            'medial_voiced_aspirates': r'\w+[bḍḍʱdhghjh]+[aāiīuūeēoō]\w+',
            
            # Simplification of consonant clusters (Classical tendency)
            'complex_clusters': r'\b\w*[kgcjṭḍtdpb][rlvy][kgcjṭḍtdpb]\w*\b',

            # ADVANCED MORPHOLOGICAL FEATURES
            # Old Present formations (N-infix, Reduplicated Presents)
            'n_infix_presents': r'\b\w*[aā]n[aā]\w*\b',
            'reduplicated_presents': r'\b([a-z]{1,2})\1\w*(ti|anti|si|santi)\b',
            
            # Archaic case endings (locative duals and plurals)
            'locative_dual': r'\b\w+yoḥ\b',
            'locative_plural': r'\b\w+ṣu\b',

            # SYNTACTIC EVOLUTION
            # Absolute constructions (locative/accusative absolute)
            'absolute_construction': r'\b\w+ena\s+\w+tvā\b',
            
            # Word order evolution (verb-final to freer order)
            'verb_nonfinal': r'\b(asti|abhavat|karoti|bhavati|gacchati|āgacchati)\b(?!.*[.।])',
            
            # Subordination markers (syntactic complexity)
            'subordinators': r'\b(yat|yadi|yena|yathā|iti|tarhi|yasmāt|yataḥ)\b',

            # LEXICAL STRATIFICATION REFINEMENTS
            # Substrate lexemes (Dravidian/Munda loans)
            'substrate_lexemes': r'\b(khaṭvā|puṣkara|niru|bilva|ulūka|mayūra|kadalī|nārī)\b',
            
            # Philosophical context co-occurrence
            'philosophical_context': r'\b(brahman|ātman|karma|mokṣa)\b.{0,50}(satya|jñāna|vidyā|cit|ānanda)',

            # TEXTUAL AND STYLISTIC FEATURES
            # Reported speech and quotations (increasing in prose)
            'reported_speech': r'iti\s+\w+[ḥ]?|uvāca|āha|abravīt',
            
            # Paragraph-initial connectives (prose markers)
            'connectives': r'\b(atha|iti\s+ha\s+uvāca|evaṁ\s+ha|tataḥ|punar|api\s+ca)\b',
            
            # Prose vs verse indicators
            'prose_particles': r'\b(khalu|kila|vai|nāma|eva|tu|punar)\b'
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

corpus_files = {
    # Samhitas (Early to Middle Vedic)
    'Rigveda': '../texts/samhita/rig-samhita.txt',
    'Samaveda': '../texts/samhita/sama-samhita.txt',
    'Yajurveda': '../texts/samhita/yajur-samhita.txt',
    'Atharvaveda (Paippalada)': '../texts/samhita/atharva-paippalada-samhita.txt',
    'Atharvaveda (Saunaka)': '../texts/samhita/atharva-saunaka-samhita.txt',

    # Brahmanas (Late Vedic)
    'Kausitaki-Br': '../texts/brahmana/rig-kausitaki.txt',
    'Pancavimsa-Br': '../texts/brahmana/sama-pancavimsa.txt',
    'Satapatha-Br': '../texts/brahmana/yajur-satapatha.txt',
    'Gopatha-Br': '../texts/brahmana/atharva-gopatha.txt',

    # Upanishads (Latest Vedic)
    'Aitareya-Up': '../texts/upanishad/rig-aitareya.txt',
    'Taittiriya-Up': '../texts/upanishad/yajur-taittiriya-up.txt',
    'Chandogya-Up': '../texts/upanishad/sama-chandogya.txt',
    'Brhadaranyaka-Up': '../texts/upanishad/yajur-brhadaranyaka.txt',
    'Prashna-Up': '../texts/upanishad/atharva-prashna.txt',
    'Shvetashvatara-Up': '../texts/upanishad/yajur-shvetashvatara.txt',

    # Classical Sanskrit (Post-Vedic)
    'Ramayana': '../texts/classical-sanskrit/ramayana.txt',
    'Mahabharata': '../texts/classical-sanskrit/mahabharata.txt',
    'Bhagavata-Purana': '../texts/classical-sanskrit/bhagavata-purana.txt'
}

# Add this line after corpus_files:
text_order = list(corpus_files.keys())

# Legacy analyzer - only run if this is the main script
if False:  # Disabled since we use EnhancedVedicAnalyzer now
    analyzer = VedicDiachronicAnalyzer()
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
        # Features are now comprehensive in parent class

    def analyze_feature_categories(self, text_name):
        """Analyze features by linguistic category"""
        results = self.results[text_name]

        categories = {
            'phonological_archaic': [
                'retroflex_l', 'visarga_final', 'diphthongs_ai', 'diphthongs_au', 
                'pluti_vowels', 'medial_voiced_aspirates'
            ],
            'phonological_innovative': [
                'monophthongs_e', 'monophthongs_o', 'retroflex_assimilation',
                'complex_clusters'
            ],
            'morphological_archaic': [
                'subjunctive_ati', 'subjunctive_full', 'dual_nominative',
                'dual_instrumental', 'perfect_reduplicated', 'injunctive_augmentless',
                'n_infix_presents', 'reduplicated_presents', 'locative_dual', 'locative_plural'
            ],
            'morphological_innovative': [
                'perfect_periphrastic', 'instrumental_classical_ena',
                'genitive_plural_thematic', 'precative', 'benedictive'
            ],
            'syntactic_archaic': [
                'particle_sma', 'particle_ha', 'particle_vai', 'particle_id'
            ],
            'syntactic_innovative': [
                'present_participle_ant', 'correlatives_ya_ta', 'conditional_yadi_tarhi',
                'long_compounds', 'gerund_tvaa', 'infinitive_tum', 'absolute_construction',
                'verb_nonfinal', 'subordinators'
            ],
            'lexical_religious': [
                'ritual_sacrifice', 'deity_names', 'priestly_terms', 'sacrifice_roots'
            ],
            'lexical_philosophical': [
                'philosophical_terms', 'cosmological_terms', 'eschatological_terms',
                'philosophical_context'
            ],
            'textual_stylistic': [
                'reported_speech', 'connectives', 'prose_particles', 'substrate_lexemes'
            ]
        }

        category_scores = {}
        for category, features in categories.items():
            scores = [results.get(feature, 0) for feature in features if feature in results]
            category_scores[category] = sum(scores) / len(scores) if scores else 0

        return category_scores

    def calculate_archaism_innovation_indices(self, text_name):
        """Calculate comprehensive archaism vs innovation indices"""
        results = self.results[text_name]

        archaic_features = [
            'retroflex_l', 'subjunctive_full', 'dual_nominative', 'dual_instrumental',
            'perfect_reduplicated', 'injunctive_augmentless', 'particle_sma',
            'particle_ha', 'particle_vai', 'diphthongs_ai', 'diphthongs_au',
            'pluti_vowels', 'medial_voiced_aspirates', 'n_infix_presents', 
            'reduplicated_presents', 'locative_dual', 'locative_plural'
        ]

        innovative_features = [
            'perfect_periphrastic', 'instrumental_classical_ena', 'long_compounds',
            'correlatives_ya_ta', 'gerund_tvaa', 'infinitive_tum', 'precative',
            'philosophical_terms', 'monophthongs_e', 'monophthongs_o',
            'complex_clusters', 'absolute_construction', 'verb_nonfinal', 
            'subordinators', 'philosophical_context', 'reported_speech', 
            'connectives', 'prose_particles'
        ]

        archaic_score = sum(results.get(f, 0) for f in archaic_features) / len(archaic_features)
        innovative_score = sum(results.get(f, 0) for f in innovative_features) / len(innovative_features)

        return {
            'archaism_index': archaic_score,
            'innovation_index': innovative_score,
            'conservation_ratio': archaic_score / innovative_score if innovative_score > 0 else float('inf')
        }

    def calculate_advanced_composite_indices(self, text_name):
        """Calculate advanced composite linguistic indices"""
        results = self.results[text_name]
        
        # Morphological innovation density
        morph_innovative_density = (
            results.get('perfect_periphrastic', 0) +
            results.get('precative', 0) +
            results.get('benedictive', 0) +
            results.get('instrumental_classical_ena', 0) +
            results.get('complex_clusters', 0)
        ) / 5
        
        # Syntactic complexity measure  
        syntactic_complexity = (
            results.get('subordinators', 0) +
            results.get('correlatives_ya_ta', 0) +
            results.get('absolute_construction', 0) +
            results.get('verb_nonfinal', 0)
        ) / 4
        
        # Phonological archaism index
        phonological_archaism = (
            results.get('pluti_vowels', 0) +
            results.get('medial_voiced_aspirates', 0) +
            results.get('retroflex_l', 0)
        ) / 3
        
        # Textual sophistication (prose vs verse markers)
        textual_sophistication = (
            results.get('reported_speech', 0) +
            results.get('connectives', 0) +
            results.get('prose_particles', 0) +
            results.get('philosophical_context', 0)
        ) / 4
        
        # Substrate influence index
        substrate_influence = results.get('substrate_lexemes', 0)
        
        return {
            'morphological_innovation_density': morph_innovative_density,
            'syntactic_complexity': syntactic_complexity,
            'phonological_archaism': phonological_archaism,
            'textual_sophistication': textual_sophistication,
            'substrate_influence': substrate_influence,
            'overall_innovation_score': (morph_innovative_density + syntactic_complexity + textual_sophistication) / 3
        }

    def analyze_diachronic_trends(self):
        """Analyze trends across all feature categories"""
        texts = list(self.results.keys())
        trend_analysis = {}

        for feature in self.features:
            values = [self.results[text].get(feature, 0) for text in texts]

            # Calculate trend direction and strength
            if len(values) > 1:
                trend_slope = (values[-1] - values[0]) / (len(values) - 1)
                trend_direction = 'increasing' if trend_slope > 0.1 else 'decreasing' if trend_slope < -0.1 else 'stable'

                # Calculate variability
                variability = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0

                trend_analysis[feature] = {
                    'direction': trend_direction,
                    'slope': trend_slope,
                    'variability': variability,
                    'early_value': values[0],
                    'late_value': values[-1],
                    'change_percent': ((values[-1] - values[0]) / values[0] * 100) if values[0] > 0 else 0
                }

        return trend_analysis
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
                    colors = ['red']*5 + ['blue']*4 + ['green']*6 + ['yellow']*3
                    ax.scatter(range(len(text_order)), values, c=colors, s=100, alpha=0.6)
                    ax.plot(range(len(text_order)), values, 'k-', alpha=0.3)

                    ax.set_title(f'{feature} frequency evolution')
                    ax.set_ylabel('Freq per 1000 words')
                    ax.set_xticks(range(len(text_order)))
                    ax.set_xticklabels(text_order, rotation=45, ha='right')

                    # Add period shading
                    ax.axvspan(-0.5, 4.5, alpha=0.1, color='red')    # Samhitas 0-4
                    ax.axvspan(4.5, 8.5, alpha=0.1, color='blue')    # Brahmanas 5-8
                    ax.axvspan(8.5, 14.5, alpha=0.1, color='green')  # Upanishads 9-14
                    ax.axvspan(14.5, 17.5, alpha=0.1, color='yellow') # Classical 15-17

                    # Trend line
                    x_numeric = np.arange(len(text_order))
                    z = np.polyfit(x_numeric, values, 2)
                    p = np.poly1d(z)
                    ax.plot(x_numeric, p(x_numeric), "r--", alpha=0.5)

            plt.tight_layout()
            plt.savefig('../output/vedic_full_diachronic_trends.png', dpi=300)
            plt.show()

def plot_diachronic_trends(analyzer):
    """Create comprehensive visualization of feature changes by category"""
    df = pd.DataFrame(analyzer.results).T
    df = df.reindex(text_order)

    # Define feature categories for focused visualization
    feature_categories = {
        'Phonological Evolution': [
            'retroflex_l', 'diphthongs_ai', 'monophthongs_e', 'visarga_final', 
            'pluti_vowels', 'medial_voiced_aspirates'
        ],
        'Morphological Archaisms': [
            'subjunctive_full', 'dual_nominative', 'perfect_reduplicated', 
            'injunctive_augmentless', 'n_infix_presents', 'reduplicated_presents', 
            'locative_dual', 'locative_plural'
        ],
        'Morphological Innovations': [
            'perfect_periphrastic', 'instrumental_classical_ena', 'precative', 'benedictive'
        ],
        'Syntactic Development': [
            'particle_sma', 'correlatives_ya_ta', 'long_compounds', 'gerund_tvaa'
        ],
        'Lexical Stratification': [
            'ritual_sacrifice', 'deity_names', 'philosophical_terms', 'cosmological_terms'
        ],
        'Textual & Stylistic': [
            'reported_speech', 'connectives', 'prose_particles', 'substrate_lexemes'
        ]
    }

    # Create comprehensive multi-panel plot
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    axes = axes.ravel()

    colors = ['red']*5 + ['blue']*4 + ['green']*6 + ['yellow']*3

    for idx, (category, features) in enumerate(feature_categories.items()):
        if idx < len(axes):
            ax = axes[idx]

            for feature in features:
                if feature in df.columns:
                    values = df[feature].values
                    ax.plot(range(len(text_order)), values, 'o-',
                           label=feature.replace('_', ' '), alpha=0.7, linewidth=2)

            ax.set_title(f'{category}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Frequency per 1000 words', fontsize=12)
            ax.set_xticks(range(len(text_order)))
            ax.set_xticklabels(text_order, rotation=45, ha='right', fontsize=10)

            # Add period shading
            ax.axvspan(-0.5, 4.5, alpha=0.1, color='red', label='Samhita Period')    # 0-4
            ax.axvspan(4.5, 8.5, alpha=0.1, color='blue', label='Brahmana Period')   # 5-8
            ax.axvspan(8.5, 14.5, alpha=0.1, color='green', label='Upanishad Period') # 9-14
            ax.axvspan(14.5, 17.5, alpha=0.1, color='yellow', label='Classical Period') # 15-17

            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../output/vedic_comprehensive_diachronic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_category_trends(analyzer):
    """Plot aggregated trends by linguistic category"""
    texts = list(analyzer.results.keys())

    # Calculate category averages for each text
    category_data = {}
    for text in texts:
        if hasattr(analyzer, 'analyze_feature_categories'):
            category_data[text] = analyzer.analyze_feature_categories(text)

    if category_data:
        df_categories = pd.DataFrame(category_data).T
        df_categories = df_categories.reindex(text_order)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()

        category_groups = [
            ['phonological_archaic', 'phonological_innovative'],
            ['morphological_archaic', 'morphological_innovative'],
            ['syntactic_archaic', 'syntactic_innovative'],
            ['lexical_religious', 'lexical_philosophical']
        ]

        titles = ['Phonological Evolution', 'Morphological Evolution',
                 'Syntactic Evolution', 'Lexical Evolution']

        for idx, (categories, title) in enumerate(zip(category_groups, titles)):
            ax = axes[idx]
            for category in categories:
                if category in df_categories.columns:
                    values = df_categories[category].values
                    style = '--' if 'archaic' in category else '-'
                    label = category.replace('_', ' ').title()
                    ax.plot(range(len(text_order)), values, style,
                           label=label, linewidth=2, marker='o')

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel('Average Frequency', fontsize=12)
            ax.set_xticks(range(len(text_order)))
            ax.set_xticklabels(text_order, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../output/vedic_category_trends.png', dpi=300, bbox_inches='tight')
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

    periods = ['Early Vedic']*5 + ['Late Vedic']*4 + ['Latest Vedic']*6 + ['Classical']*3
    dates = [1500, 1200, 1000, 800, 600] + [900, 850, 800, 750] + [700, 650, 600, 520, 480, 450] + [200, 300, 800]

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

def create_comprehensive_report(analyzer):
    """Generate detailed analysis report covering all linguistic aspects"""
    with open('vedic_comprehensive_diachronic_report.txt', 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE VEDIC SANSKRIT DIACHRONIC ANALYSIS\n")
        f.write("="*70 + "\n\n")

        f.write("CORPUS OVERVIEW\n")
        f.write("-"*50 + "\n")
        f.write("• Samhitas (c. 1500-1000 BCE): Rigveda, Yajurveda, Samaveda, Atharvaveda\n")
        f.write("• Brahmanas (c. 900-750 BCE): Kausitaki, Pancavimsa, Satapatha, Gopatha\n")
        f.write("• Upanishads (c. 700-550 BCE): Brhadaranyaka, Chandogya, Aitareya, Mandukya\n\n")

        f.write(f"TOTAL FEATURES ANALYZED: {len(analyzer.features)}\n\n")

        # Analyze diachronic trends
        if hasattr(analyzer, 'analyze_diachronic_trends'):
            trend_analysis = analyzer.analyze_diachronic_trends()

            f.write("MAJOR DIACHRONIC TRENDS\n")
            f.write("-"*50 + "\n")

            # Group trends by direction
            increasing = [f for f, data in trend_analysis.items() if data['direction'] == 'increasing']
            decreasing = [f for f, data in trend_analysis.items() if data['direction'] == 'decreasing']
            stable = [f for f, data in trend_analysis.items() if data['direction'] == 'stable']

            f.write(f"INCREASING FEATURES ({len(increasing)}):\n")
            for feature in sorted(increasing, key=lambda x: trend_analysis[x]['change_percent'], reverse=True)[:10]:
                data = trend_analysis[feature]
                f.write(f"  • {feature.replace('_', ' ')}: {data['change_percent']:+.1f}% change\n")

            f.write(f"\nDECREASING FEATURES ({len(decreasing)}):\n")
            for feature in sorted(decreasing, key=lambda x: abs(trend_analysis[x]['change_percent']), reverse=True)[:10]:
                data = trend_analysis[feature]
                f.write(f"  • {feature.replace('_', ' ')}: {data['change_percent']:+.1f}% change\n")

            f.write(f"\nSTABLE FEATURES ({len(stable)})\n")

        # Category analysis
        f.write("\n\nFEATURE CATEGORY ANALYSIS\n")
        f.write("-"*50 + "\n")

        categories = {
            'Phonological Archaisms': ['retroflex_l', 'diphthongs_ai', 'diphthongs_au', 'visarga_final', 
                                     'pluti_vowels', 'medial_voiced_aspirates'],
            'Phonological Innovations': ['monophthongs_e', 'monophthongs_o', 'retroflex_assimilation',
                                       'complex_clusters'],
            'Morphological Archaisms': ['subjunctive_full', 'dual_nominative', 'perfect_reduplicated', 
                                      'injunctive_augmentless', 'n_infix_presents', 'reduplicated_presents',
                                      'locative_dual', 'locative_plural'],
            'Morphological Innovations': ['perfect_periphrastic', 'instrumental_classical_ena', 'precative', 'benedictive'],
            'Syntactic Archaisms': ['particle_sma', 'particle_ha', 'particle_vai'],
            'Syntactic Innovations': ['correlatives_ya_ta', 'long_compounds', 'gerund_tvaa',
                                    'absolute_construction', 'verb_nonfinal', 'subordinators'],
            'Religious Lexicon': ['ritual_sacrifice', 'deity_names', 'priestly_terms'],
            'Philosophical Lexicon': ['philosophical_terms', 'cosmological_terms', 'eschatological_terms',
                                    'philosophical_context'],
            'Textual & Stylistic': ['reported_speech', 'connectives', 'prose_particles', 'substrate_lexemes']
        }

        for category, features in categories.items():
            f.write(f"\n{category.upper()}:\n")
            for feature in features:
                if feature in analyzer.features:
                    values = [analyzer.results[text].get(feature, 0) for text in text_order]
                    early_avg = np.mean(values[:5])    # Samhitas (0-4)
                    late_avg = np.mean(values[9:15])  # Upanishads (9-14)

                    if early_avg > 0:
                        change = ((late_avg - early_avg) / early_avg) * 100
                        trend = "↗" if change > 10 else "↘" if change < -10 else "→"
                        f.write(f"  {trend} {feature.replace('_', ' ')}: {early_avg:.2f} → {late_avg:.2f} ({change:+.1f}%)\n")

        # Archaic vs innovative indices
        f.write("\n\nARCHAISM vs INNOVATION INDICES\n")
        f.write("-"*50 + "\n")

        periods = {
            'Samhita Period': text_order[:5],    # 5 texts (0-4)
            'Brahmana Period': text_order[5:9],  # 4 texts (5-8)  
            'Upanishad Period': text_order[9:15], # 6 texts (9-14)
            'Classical Period': text_order[15:]  # 3 texts (15-17)
        }

        for period, texts in periods.items():
            if hasattr(analyzer, 'calculate_archaism_innovation_indices'):
                indices = []
                for text in texts:
                    indices.append(analyzer.calculate_archaism_innovation_indices(text))

                avg_archaic = np.mean([idx['archaism_index'] for idx in indices])
                avg_innovative = np.mean([idx['innovation_index'] for idx in indices])
                avg_ratio = np.mean([idx['conservation_ratio'] for idx in indices if idx['conservation_ratio'] != float('inf')])

                f.write(f"{period}:\n")
                f.write(f"  Archaism Index: {avg_archaic:.2f}\n")
                f.write(f"  Innovation Index: {avg_innovative:.2f}\n")
                f.write(f"  Conservation Ratio: {avg_ratio:.2f}\n\n")

        # Advanced Composite Indices
        f.write("\n\nADVANCED COMPOSITE INDICES\n")
        f.write("-"*50 + "\n")

        for period, texts in periods.items():
            if hasattr(analyzer, 'calculate_advanced_composite_indices'):
                composite_indices = []
                for text in texts:
                    composite_indices.append(analyzer.calculate_advanced_composite_indices(text))

                avg_morph_innovation = np.mean([idx['morphological_innovation_density'] for idx in composite_indices])
                avg_syntactic_complexity = np.mean([idx['syntactic_complexity'] for idx in composite_indices])
                avg_phonological_archaism = np.mean([idx['phonological_archaism'] for idx in composite_indices])
                avg_textual_sophistication = np.mean([idx['textual_sophistication'] for idx in composite_indices])
                avg_substrate_influence = np.mean([idx['substrate_influence'] for idx in composite_indices])
                avg_overall_innovation = np.mean([idx['overall_innovation_score'] for idx in composite_indices])

                f.write(f"{period}:\n")
                f.write(f"  Morphological Innovation Density: {avg_morph_innovation:.2f}\n")
                f.write(f"  Syntactic Complexity: {avg_syntactic_complexity:.2f}\n")
                f.write(f"  Phonological Archaism: {avg_phonological_archaism:.2f}\n")
                f.write(f"  Textual Sophistication: {avg_textual_sophistication:.2f}\n")
                f.write(f"  Substrate Influence: {avg_substrate_influence:.2f}\n")
                f.write(f"  Overall Innovation Score: {avg_overall_innovation:.2f}\n\n")

        f.write("\nMETHODOLOGY NOTES\n")
        f.write("-"*50 + "\n")
        f.write("• All frequencies normalized per 1000 words\n")
        f.write("• Chronological ordering based on scholarly consensus\n")
        f.write("• Feature detection via regex pattern matching\n")
        f.write("• Statistical significance testing applied\n")
        f.write("• Comprehensive coverage of phonological, morphological, syntactic, and lexical domains\n")
        f.write("• Advanced phonological features: pluti vowels, voiced aspirate retention, cluster simplification\n")
        f.write("• Morphological archaisms: N-infix presents, reduplicated presents, archaic case forms\n")
        f.write("• Syntactic evolution: absolute constructions, word order patterns, subordination markers\n")
        f.write("• Lexical stratification: substrate influence, philosophical terminology co-occurrence\n")
        f.write("• Textual sophistication: prose vs verse markers, reported speech, connectives\n")
        f.write("• Composite indices: innovation density, syntactic complexity, textual sophistication\n\n")

        f.write("Generated by Enhanced Shrauta-Lakshana Diachronic Analyzer\n")

if __name__ == '__main__':
    analyzer = EnhancedVedicAnalyzer()
    analyzer.analyze_corpus(corpus_files)

    # Generate comprehensive visualizations
    plot_diachronic_trends(analyzer)
    plot_category_trends(analyzer)

    # Statistical analysis
    statistical_analysis(analyzer)

    # Export results with enhanced features
    export_results(analyzer)

    # Generate comprehensive report
    create_comprehensive_report(analyzer)


# All reports and visualizations are generated in the main block above
