# Shrauta-Lakshana

A computational linguistics toolkit for diachronic analysis of Vedic Sanskrit texts, focusing on morphological and syntactic features that track the historical development of the language across the Samhita corpus.

## Overview

Shrauta-Lakshana analyzes linguistic patterns in the four Vedic Samhitas (Rig, Yajur, Sama, and Atharva) to identify diachronic changes in Sanskrit morphology, syntax, and lexicon. The project provides quantitative analysis of archaic features, emerging innovations, and statistical significance testing for linguistic evolution.

## Features

- **Diachronic Feature Analysis**: Tracks 15+ linguistic features across chronological periods
- **Statistical Analysis**: Significance testing for linguistic changes between texts
- **Visualization**: Time-series plots showing feature frequency evolution
- **Export Capabilities**: CSV output for further statistical analysis
- **Sanskrit Integration**: Optional Sanskrit parsing for morphological analysis

## Project Structure

```
Shrauta-Lakshana/
├── analysis/
│   ├── samhita-analysis.py      # Core diachronic analyzer
│   ├── diachronic-analysis.py   # Enhanced analysis with visualization
│   └── vedic_diachronic_trends.png
├── texts/
│   ├── samhita/
│   │   ├── rig-samhita.txt      # Rigveda corpus (Śākala Śākhā)
│   │   ├── yajur-samhita.txt    # Yajurveda corpus (Kṛṣṇa Yajurveda – Maitrāyaṇī Śākhā)
│   │   ├── sama-samhita.txt     # Samaveda corpus (Kauthuma Śākhā)
│   │   └── atharva-samhita.txt  # Atharvaveda corpus (Śaunaka Śākhā)
│   ├── brahmana/
│   │   ├── rig-kausitaki.txt    # Kauṣītaki Brāhmaṇa (Rigveda)
│   │   ├── yajur-satapatha.txt  # Śatapatha Brāhmaṇa (Yajurveda)
│   │   ├── sama-pancavimsa.txt  # Pañcaviṃśa Brāhmaṇa (Samaveda)
│   │   └── atharva-gopatha.txt  # Gopatha Brāhmaṇa (Atharvaveda)
│   └── upanishad/
│       ├── rig-aitareya.txt     # Aitareya Upaniṣad (Rigveda)
│       ├── yajur-brhadaranyaka.txt # Bṛhadāraṇyaka Upaniṣad (Yajurveda)
│       ├── sama-chandogya.txt   # Chāndogya Upaniṣad (Samaveda)
│       ├── atharva-mandukya.txt # Māṇḍūkya Upaniṣad (Atharvaveda)
├── preprocessing/               # Text processing tools
          └── rm_blank.py
└── README.md
```

## Linguistic Features Analyzed

### Archaic Features (Decreasing Over Time)
- **Retroflex ḷ**: Archaic liquid consonant
- **Subjunctive forms**: -āti, -ān endings
- **Vedic particles**: sma, ha, vai
- **Injunctive**: Past tense without augment
- **Dual forms**: Archaic number category

### Innovations (Increasing Over Time)
- **Long compounds**: 15+ character compound words
- **Infinitive -tum**: Later grammatical development
- **Gerund -tvā**: Participial forms
- **Precative/Benedictive**: Modal verb forms

### Morphological Patterns
- **Verb forms**: Present, perfect, aorist distributions
- **Nominal patterns**: Instrumental, genitive plural forms
- **Particle usage**: Frequency of discourse markers

## Installation

### Prerequisites
- Python 3.7+
- Required packages:
  ```bash
  pip install matplotlib pandas scipy numpy
  ```

### Optional Dependencies
For advanced Sanskrit parsing:
```bash
pip install sanskrit_parser indic_transliteration
```

## Usage

### Basic Analysis
```python
from analysis.samhita_analysis import VedicDiachronicAnalyzer

analyzer = VedicDiachronicAnalyzer()
corpus_files = {
    'Rigveda': 'texts/samhita/rig-samhita.txt',
    'Yajurveda': 'texts/samhita/yajur-samhita.txt',
    'Samaveda': 'texts/samhita/sama-samhita.txt',
    'Atharvaveda': 'texts/samhita/atharva-samhita.txt'
}

analyzer.analyze_corpus(corpus_files)
analyzer.generate_report()
```

### Enhanced Analysis with Visualization
```python
from analysis.diachronic_analysis import EnhancedVedicAnalyzer, plot_diachronic_trends

analyzer = EnhancedVedicAnalyzer()
analyzer.analyze_corpus(corpus_files)

# Generate plots
plot_diachronic_trends(analyzer)

# Export results
analyzer.export_results('vedic_analysis.csv')
```

### Running Analysis Scripts
```bash
cd analysis/
python samhita-analysis.py          # Basic frequency analysis
python diachronic-analysis.py       # Full analysis with plots
```

## Output

### Console Report
```
Diachronic Analysis Results
============================================================
Feature             Rigveda        Yajurveda      Samaveda       Atharvaveda
------------------------------------------------------------
retroflex_l            2.34           1.89           1.45           0.87
subjunctive_ati        5.67           4.23           3.12           1.98
particle_sma           3.45           2.78           2.01           1.23
long_compounds         1.23           2.45           3.67           4.89
```

### Generated Files
- `vedic_diachronic_trends.png`: Time-series visualization
- `vedic_analysis.csv`: Detailed frequency data
- `features_over_time.csv`: Feature matrix for statistical analysis
- `vedic_diachronic_report.txt`: Summary findings

## Methodology

### Frequency Normalization
All features are normalized to occurrences per 1,000 words to account for corpus size differences.

### Chronological Ordering
Texts are analyzed in traditional chronological sequence:
1. **Rigveda** (c. 1500 BCE) - Early Vedic
2. **Yajurveda** (c. 1200 BCE) - Middle Vedic
3. **Samaveda** (c. 1000 BCE) - Middle Vedic
4. **Atharvaveda** (c. 800 BCE) - Late Vedic

### Statistical Testing
- Trend analysis using polynomial regression
- Significance testing for inter-textual differences
- Archaism vs. Innovation indices

## Research Applications

- **Historical Linguistics**: Quantifying Sanskrit's diachronic development
- **Philology**: Dating and stratification of Vedic texts
- **Computational Linguistics**: Feature extraction from ancient corpora
- **Digital Humanities**: Corpus-based analysis of sacred texts

## Text Preprocessing

Use the included utility to clean texts:
```bash
cd texts/samhita/
python rm_blank.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add new linguistic features or analysis methods
4. Include appropriate tests and documentation
5. Submit a pull request

## License

This project is released under the MIT License.

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue or contact [ahari8@illinois.edu].

---

*Shrauta-Lakshana* - From the Sanskrit श्रौत (śrauta, "Vedic") + लक्षण (lakṣaṇa, "characteristic"), referring to the distinctive features that mark different branches of Vedic tradition.
