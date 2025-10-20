#!/usr/bin/env python3
"""
Calculate exact word counts by period from the latest documentation
"""

# Individual text word counts from comprehensive_results_analysis.py
estimated_word_counts = {
    'Rigveda': 153000, 'Samaveda': 65000, 'Yajurveda (Taittiriya)': 86000,
    'Yajurveda (Maitrayani)': 71000, 'Atharvaveda (Paippalada)': 72000,
    'Atharvaveda (Saunaka)': 74000, 'Kausitaki-Br': 67000,
    'Pancavimsa-Br': 58000, 'Taittiriya-Br': 95000, 'Gopatha-Br': 45000,
    'Aitareya-Up': 15000, 'Taittiriya-Up': 25000, 'Chandogya-Up': 35000,
    'Brhadaranyaka-Up': 48000, 'Prashna-Up': 8000, 'Shvetashvatara-Up': 12000,
    'Ramayana': 200000, 'Mahabharata': 400000, 'Bhagavata-Purana': 175000
}

# Period mapping
period_mapping = {
    'Rigveda': 'Early Vedic (Samhita)',
    'Samaveda': 'Early Vedic (Samhita)',
    'Yajurveda (Taittiriya)': 'Early Vedic (Samhita)',
    'Yajurveda (Maitrayani)': 'Early Vedic (Samhita)',
    'Atharvaveda (Paippalada)': 'Early Vedic (Samhita)',
    'Atharvaveda (Saunaka)': 'Early Vedic (Samhita)',
    'Kausitaki-Br': 'Late Vedic (Brahmana)',
    'Pancavimsa-Br': 'Late Vedic (Brahmana)',
    'Taittiriya-Br': 'Late Vedic (Brahmana)',
    'Gopatha-Br': 'Late Vedic (Brahmana)',
    'Aitareya-Up': 'Latest Vedic (Upanishad)',
    'Taittiriya-Up': 'Latest Vedic (Upanishad)',
    'Chandogya-Up': 'Latest Vedic (Upanishad)',
    'Brhadaranyaka-Up': 'Latest Vedic (Upanishad)',
    'Prashna-Up': 'Latest Vedic (Upanishad)',
    'Shvetashvatara-Up': 'Latest Vedic (Upanishad)',
    'Ramayana': 'Classical (Epic/Purana)',
    'Mahabharata': 'Classical (Epic/Purana)',
    'Bhagavata-Purana': 'Classical (Epic/Purana)'
}

# Calculate period totals
period_totals = {}
for text, word_count in estimated_word_counts.items():
    period = period_mapping[text]
    if period not in period_totals:
        period_totals[period] = 0
    period_totals[period] += word_count

print("📊 WORD COUNTS BY PERIOD (Latest Documentation)")
print("=" * 60)

for period, total in period_totals.items():
    print(f"{period:<30} {total:>10,} words")

print("-" * 60)
grand_total = sum(period_totals.values())
print(f"{'TOTAL':<30} {grand_total:>10,} words")

print("\n📋 DETAILED BREAKDOWN:")
print("=" * 60)

current_period = None
for text, word_count in estimated_word_counts.items():
    period = period_mapping[text]
    if period != current_period:
        if current_period is not None:
            print()
        print(f"\n{period}:")
        current_period = period
    print(f"  {text:<25} {word_count:>8,}")