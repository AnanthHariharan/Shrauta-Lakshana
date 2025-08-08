# Shrauta-Lakshana

A comprehensive computational linguistics toolkit for advanced diachronic analysis of Vedic Sanskrit literature, spanning from the Samhitas through Classical Sanskrit. The project employs sophisticated statistical modeling, type-token frequency analysis, and machine learning techniques to quantify the historical development of Sanskrit across multiple linguistic domains.

## Overview

Shrauta-Lakshana provides multi-layered analysis of Sanskrit texts from c. 1500 BCE to 800 CE, covering the complete Vedic corpus plus Classical Sanskrit literature. The toolkit combines traditional philological approaches with cutting-edge computational methods to reveal patterns of linguistic evolution across phonological, morphological, syntactic, and lexical domains.

## Key Features

- **Comprehensive Diachronic Analysis**: Tracks 60+ linguistic features across chronological periods
- **Advanced Statistical Modeling**: Machine learning models (Random Forest, Gradient Boosting, PCA) for linguistic prediction and classification
- **Type-Token Frequency Analysis**: Morphological productivity and lexical diversity measurements
- **Multi-Domain Coverage**: Phonological, morphological, syntactic, lexical, and stylistic feature analysis
- **Hierarchical Text Clustering**: Automated grouping of texts by linguistic similarity
- **Comprehensive Visualization**: Time-series plots, heatmaps, PCA plots, regression diagnostics
- **Export Capabilities**: CSV output, detailed reports, and statistical summaries
- **Metadata Integration**: Text classification by genre, period, recension, and geographical origin

## Project Structure

```
Shrauta-Lakshana/
├── analysis/                           # Core analysis modules
│   ├── diachronic_analysis.py         # Comprehensive diachronic feature analyzer
│   ├── samhita_analysis.py            # Basic corpus analysis tools
│   ├── statistical_modeling.py        # Advanced statistical models & ML
│   ├── type_token_analysis.py         # Morphological productivity & lexical diversity
│   ├── run_statistical_analysis.py    # Statistical analysis runner
│   └── run_type_token_analysis.py     # Type-token analysis runner
├── output/                            # Generated results & visualizations
│   ├── *.csv                         # Data exports & statistical summaries
│   ├── *.png                         # Visualizations & plots
│   └── *.txt                         # Detailed analysis reports
├── texts/                             # Sanskrit corpus collection
│   ├── samhita/                      # Early Vedic (1500-800 BCE)
│   │   ├── rig-samhita.txt           # Rigveda (Śākala Śākhā)
│   │   ├── yajur-samhita.txt         # Yajurveda (Maitrāyaṇī Śākhā)
│   │   ├── sama-samhita.txt          # Samaveda (Kauthuma Śākhā)
│   │   ├── atharva-saunaka-samhita.txt  # Atharvaveda (Śaunaka Śākhā)
│   │   └── atharva-paippalada-samhita.txt # Atharvaveda (Paippalāda Śākhā)
│   ├── brahmana/                     # Late Vedic prose (900-750 BCE)
│   │   ├── rig-kausitaki.txt         # Kauṣītaki Brāhmaṇa
│   │   ├── yajur-satapatha.txt       # Śatapatha Brāhmaṇa
│   │   ├── sama-pancavimsa.txt       # Pañcaviṃśa Brāhmaṇa
│   │   └── atharva-gopatha.txt       # Gopatha Brāhmaṇa
│   ├── upanishad/                    # Latest Vedic (700-450 BCE)
│   │   ├── rig-aitareya.txt          # Aitareya Upaniṣad
│   │   ├── yajur-brhadaranyaka.txt   # Bṛhadāraṇyaka Upaniṣad
│   │   ├── yajur-taittiriya-up.txt   # Taittirīya Upaniṣad
│   │   ├── yajur-shvetashvatara.txt  # Śvetāśvatara Upaniṣad
│   │   ├── sama-chandogya.txt        # Chāndogya Upaniṣad
│   │   ├── atharva-mandukya.txt      # Māṇḍūkya Upaniṣad
│   │   └── atharva-prashna.txt       # Praśna Upaniṣad
│   └── classical-sanskrit/           # Post-Vedic (200-800 CE)
│       ├── mahabharata.txt           # Mahābhārata epic
│       ├── ramayana.txt              # Rāmāyaṇa epic
│       └── bhagavata-purana.txt      # Bhāgavata Purāṇa
├── preprocessing/                     # Text processing utilities
│   ├── para_format.py               # Text formatting tools
│   └── rm_blank.py                  # Text cleaning utilities
├── requirements.txt                   # Python dependencies
└── README.md                         # Project documentation
```

## Linguistic Domains Analyzed

### Phonological Evolution (15+ features)
- **Archaic sounds**: Retroflex ḷ, pluti vowels, medial voiced aspirates
- **Vowel system**: Diphthongs (ai, au) → monophthongs (e, o)
- **Consonant changes**: Retroflex assimilation, cluster simplification
- **Sandhi patterns**: External sandhi resolution

### Morphological Systems (20+ features)
- **Verbal archaisms**: Subjunctive system, injunctive forms, reduplicated presents
- **Innovative forms**: Periphrastic perfect, precative/benedictive modals
- **Nominal evolution**: Dual number decline, case system changes
- **Derivational patterns**: Agent nouns, abstract formations, productivity measures

### Syntactic Development (10+ features)
- **Particle systems**: Vedic discourse markers (sma, ha, vai)
- **Participial constructions**: Present and past participles
- **Complex syntax**: Correlative constructions, absolute constructions
- **Word order**: Verb positioning patterns
- **Subordination**: Development of complex clause structures

### Lexical Stratification (15+ features)
- **Religious terminology**: Ritual vocabulary, deity names, priestly terms
- **Philosophical vocabulary**: Abstract concepts, cosmological terms
- **Substrate influence**: Non-Indo-European lexical borrowings
- **Compound formation**: Length and complexity evolution
- **Semantic fields**: Co-occurrence patterns and contextual analysis

### Textual & Stylistic Features (10+ features)
- **Genre markers**: Prose vs. verse indicators
- **Discourse structure**: Reported speech, connectives
- **Register variation**: Formal vs. informal linguistic features
- **Metrical patterns**: Syllable structure and prosodic analysis

## Installation

### Prerequisites
- Python 3.7+
- Required packages (see requirements.txt):
  ```bash
  pip install -r requirements.txt
  ```

### Core Dependencies
```bash
# Essential packages
numpy>=1.21.0          # Scientific computing
pandas>=1.3.0          # Data manipulation
scipy>=1.7.0           # Statistical functions
matplotlib>=3.4.0      # Plotting
seaborn>=0.11.0        # Advanced visualization
scikit-learn>=1.0.0    # Machine learning models
```

### Optional Dependencies
For advanced Sanskrit parsing:
```bash
pip install sanskrit_parser indic_transliteration
```

## Usage

### 1. Comprehensive Diachronic Analysis
```python
from analysis.diachronic_analysis import EnhancedVedicAnalyzer, plot_diachronic_trends

# Initialize analyzer with 60+ linguistic features
analyzer = EnhancedVedicAnalyzer()

# Analyze complete corpus (Samhitas → Classical Sanskrit)
analyzer.analyze_corpus(corpus_files)

# Generate comprehensive visualizations
plot_diachronic_trends(analyzer)
plot_category_trends(analyzer)

# Export results to output/
analyzer.export_results()
```

### 2. Advanced Statistical Modeling
```python
from analysis.statistical_modeling import VedicStatisticalModeler, VedicTextMetadata

# Initialize with metadata integration
metadata = VedicTextMetadata()
modeler = VedicStatisticalModeler(analyzer, metadata)

# Build predictive models (Random Forest, Gradient Boosting, etc.)
models = modeler.build_predictive_models()

# Analyze feature importance
importance = modeler.analyze_feature_importance()

# Perform clustering analysis
clusters = modeler.cluster_texts_by_features()

# PCA analysis
pca_results = modeler.principal_component_analysis()
```

### 3. Type-Token Frequency Analysis
```python
from analysis.type_token_analysis import VedicTypeTokenAnalyzer

# Morphological productivity analysis
tt_analyzer = VedicTypeTokenAnalyzer()
tt_analyzer.analyze_text_comprehensive(filepath, text_name)

# Calculate productivity measures
productivity = tt_analyzer.calculate_morphological_productivity()
diversity = tt_analyzer.calculate_lexical_diversity_measures()

# Generate comprehensive report
tt_analyzer.generate_comprehensive_report()
```

### 4. Running Complete Analysis Pipelines
```bash
cd analysis/

# Run comprehensive diachronic analysis
python diachronic_analysis.py

# Run statistical modeling pipeline  
python run_statistical_analysis.py

# Run type-token analysis pipeline
python run_type_token_analysis.py
```

## Output

### Generated Visualizations (output/*.png)
- **Diachronic Analysis**: 
  - `vedic_comprehensive_diachronic_analysis.png`: Multi-panel feature evolution
  - `vedic_category_trends.png`: Linguistic category comparisons
- **Statistical Modeling**:
  - `vedic_statistical_model_performance.png`: ML model comparisons  
  - `vedic_feature_importance_heatmap.png`: Feature importance across models
  - `vedic_pca_analysis.png`: Principal component analysis plots
  - `vedic_regression_diagnostics_*.png`: Detailed regression diagnostics
  - `vedic_text_clustering_dendrogram.png`: Hierarchical clustering results
- **Type-Token Analysis**:
  - `vedic_type_token_comparison.png`: Morphological productivity plots
  - `vedic_ttr_evolution.png`: Lexical diversity evolution

### Data Exports (output/*.csv)
- `vedic_analysis.csv`: Complete feature frequency matrix
- `features_over_time.csv`: Chronologically ordered feature data
- `vedic_type_token_statistical_summary.csv`: Type-token analysis results

### Detailed Reports (output/*.txt)
- `vedic_comprehensive_diachronic_report.txt`: Complete linguistic analysis
- `vedic_statistical_analysis_report.txt`: Statistical modeling results
- `vedic_type_token_analysis_report.txt`: Morphological productivity analysis

### Sample Console Output
```
COMPREHENSIVE VEDIC SANSKRIT DIACHRONIC ANALYSIS
================================================================
CORPUS OVERVIEW: 18 texts analyzed (Samhitas → Classical)
TOTAL FEATURES ANALYZED: 63

MAJOR DIACHRONIC TRENDS:
INCREASING FEATURES (10):
  • long_compounds: +285.3% change (Samhita → Classical)
  • philosophical_terms: +156.7% change
  • subordinators: +98.4% change

DECREASING FEATURES (15):
  • subjunctive_full: -78.9% change
  • retroflex_l: -67.3% change
  • particle_sma: -89.1% change

MODEL PERFORMANCE:
  Random Forest R² = 0.847 (subjunctive_full prediction)
  PCA: 4 components explain 82.3% of variance
  Clustering: 3 distinct chronological groups identified
```

## Methodology

### Multi-Layered Analysis Framework

#### 1. Feature Extraction & Normalization  
- **Pattern Recognition**: 60+ regex-based linguistic feature patterns
- **Frequency Normalization**: All frequencies normalized per 1,000 words
- **Type-Token Distinction**: Morphological productivity via type/token ratios
- **Contextual Analysis**: Co-occurrence patterns and semantic fields

#### 2. Statistical Modeling
- **Machine Learning Models**: Random Forest, Gradient Boosting, Linear Regression
- **Cross-Validation**: 5-fold CV for model validation and overfitting detection
- **Feature Importance**: Quantified contribution of linguistic predictors
- **Regression Diagnostics**: Residual analysis, normality tests, R² metrics

#### 3. Dimensionality Reduction & Clustering
- **Principal Component Analysis**: Identify major linguistic variation patterns
- **Hierarchical Clustering**: Ward linkage for text similarity grouping
- **Metadata Integration**: Period, genre, recension, geographical classification

#### 4. Chronological Framework
Texts analyzed in refined chronological sequence:
- **Early Vedic** (1500-1000 BCE): Samhitas
- **Late Vedic** (900-750 BCE): Brahmanas  
- **Latest Vedic** (700-450 BCE): Upanishads
- **Classical** (200-800 CE): Epics & Puranas

#### 5. Advanced Composite Indices
- **Archaism Index**: Weighted average of declining features
- **Innovation Index**: Weighted average of emerging features  
- **Morphological Productivity**: Type-token ratios for inflectional patterns
- **Syntactic Complexity**: Multi-feature syntactic development measure

## Research Applications

### Historical Linguistics
- **Diachronic Change Quantification**: Precise measurement of linguistic evolution rates
- **Feature Dating**: Statistical models for linguistic feature chronology
- **Language Contact**: Substrate influence detection and quantification
- **Comparative Analysis**: Cross-linguistic diachronic pattern comparison

### Digital Philology  
- **Text Dating**: Machine learning models for chronological classification
- **Manuscript Tradition**: Recension identification via linguistic markers
- **Authorship Analysis**: Style-based text attribution methods
- **Critical Edition Support**: Statistical evidence for textual variants

### Computational Sanskrit Studies
- **Morphological Analysis**: Automated productivity and diversity measurement
- **Syntax Evolution**: Quantified syntactic complexity development  
- **Lexical Stratification**: Statistical identification of vocabulary layers
- **Corpus Linguistics**: Large-scale pattern detection across Sanskrit literature

### Digital Humanities
- **Cultural Evolution**: Language change as proxy for cultural transformation
- **Religious Studies**: Theological vocabulary development tracking
- **Educational Applications**: Data-driven Sanskrit historical grammar
- **Visualization**: Interactive exploration of linguistic change patterns

## Text Preprocessing

Utilities for corpus preparation:
```bash
cd preprocessing/
python rm_blank.py      # Remove blank lines and normalize spacing
python para_format.py   # Format paragraph structure
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
