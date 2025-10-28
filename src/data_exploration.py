"""
PhishGuard - Data Exploration Script
Explore CEAS CSV dataset structure and characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PHISHGUARD - DATA EXPLORATION")
print("=" * 70)

# ============================================
# CONFIGURATION
# ============================================
print("\n⚙️ Configuration")
print("-" * 70)

# Update this path to your CEAS CSV file
CEAS_CSV_PATH = "../data/raw/CEAS_08.csv"
OUTPUT_DIR = "../data/analysis/"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Dataset path: {CEAS_CSV_PATH}")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================
# LOAD DATASET
# ============================================
print("\n📂 Step 1: Loading Dataset")
print("-" * 70)

try:
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            df = pd.read_csv(CEAS_CSV_PATH, encoding=encoding)
            print(f"✅ Successfully loaded with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise Exception("Could not decode file with any common encoding")

except FileNotFoundError:
    print(f"❌ File not found: {CEAS_CSV_PATH}")
    print("💡 Please update CEAS_CSV_PATH variable")
    exit(1)
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit(1)

# ============================================
# BASIC DATASET INFORMATION
# ============================================
print("\n📊 Step 2: Dataset Information")
print("-" * 70)

print(f"Dataset shape: {df.shape}")
print(f"Number of rows: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]}")

print("\n📋 Column Names and Data Types:")
print(df.dtypes)

print("\n🔍 First 10 rows:")
print(df.head(10))

print("\n📝 Column Details:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype} | Unique values: {df[col].nunique()}")

# ============================================
# DATA QUALITY CHECKS
# ============================================
print("\n🔍 Step 3: Data Quality Checks")
print("-" * 70)

print("\n📊 Missing Values:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
})
print(missing_df[missing_df['Missing Count'] > 0])

print("\n📊 Duplicate Rows:")
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")

# ============================================
# USE CLEAN DATAFRAME
# ============================================
df_clean = df.copy()

# ============================================
# LABEL DISTRIBUTION ANALYSIS
# ============================================
print("\n📊 Step 4: Label Distribution Analysis")
print("-" * 70)

if 'label' in df_clean.columns:
    # Check unique labels
    unique_labels = df_clean['label'].unique()
    print(f"Unique labels: {unique_labels}")

    # Map to standard labels - handle both string and integer labels
    if df_clean['label'].dtype == object:
        label_mapping = {
            'spam': 'Phishing Email',
            'phishing': 'Phishing Email',
            'malicious': 'Phishing Email',
            'ham': 'Safe Email',
            'safe': 'Safe Email',
            'legitimate': 'Safe Email'
        }
        df_clean['email_type'] = df_clean['label'].str.lower().map(label_mapping)

        # Fill any unmapped values
        unmapped_mask = df_clean['email_type'].isnull()
        if unmapped_mask.any():
            print(f"⚠️ {unmapped_mask.sum()} unmapped labels, using original labels")
            df_clean.loc[unmapped_mask, 'email_type'] = df_clean.loc[unmapped_mask, 'label']
    else:
        # Assume 1 is phishing, 0 is safe
        df_clean['email_type'] = df_clean['label'].map({1: 'Phishing Email', 0: 'Safe Email'})

    # Count distribution
    label_counts = df_clean['email_type'].value_counts()
    label_percentages = df_clean['email_type'].value_counts(normalize=True) * 100

    print("\n📈 Label Distribution:")
    for label, count in label_counts.items():
        percentage = label_percentages[label]
        print(f"  {label}: {count} ({percentage:.2f}%)")

    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(label_counts.index, label_counts.values, color=['#ff6b6b', '#51cf66'])
    plt.title('Email Type Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Email Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}label_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

else:
    print("❌ No label column found in dataset")
    print("💡 Available columns:", list(df.columns))

# ============================================
# TEXT STATISTICS
# ============================================
print("\n📊 Step 5: Text Statistics")
print("-" * 70)

# Combine subject and body for analysis
if 'subject' in df_clean.columns and 'body' in df_clean.columns:
    df_clean['email_text'] = df_clean['subject'].fillna('') + ' ' + df_clean['body'].fillna('')
    text_column = 'email_text'
elif 'body' in df_clean.columns:
    text_column = 'body'
    df_clean['email_text'] = df_clean['body']
else:
    # Find any text-like column
    text_columns = [col for col in df_clean.columns if any(word in col.lower() for word in ['text', 'content', 'body', 'message'])]
    if text_columns:
        text_column = text_columns[0]
        df_clean['email_text'] = df_clean[text_column]
    else:
        print("❌ No text columns found")
        text_column = None

if text_column:
    # Calculate text statistics
    df_clean['text_length'] = df_clean['email_text'].astype(str).str.len()
    df_clean['word_count'] = df_clean['email_text'].astype(str).str.split().str.len()

    print("\n📝 Text Length Statistics:")
    print(f"  Mean length: {df_clean['text_length'].mean():.0f} characters")
    print(f"  Median length: {df_clean['text_length'].median():.0f} characters")
    print(f"  Min length: {df_clean['text_length'].min()} characters")
    print(f"  Max length: {df_clean['text_length'].max()} characters")
    print(f"  Standard deviation: {df_clean['text_length'].std():.0f} characters")

    print(f"\n📝 Word Count Statistics:")
    print(f"  Mean words: {df_clean['word_count'].mean():.0f}")
    print(f"  Median words: {df_clean['word_count'].median():.0f}")
    print(f"  Min words: {df_clean['word_count'].min()}")
    print(f"  Max words: {df_clean['word_count'].max()}")

    # Create histogram of text lengths
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(df_clean['text_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Email Text Lengths', fontweight='bold')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency')
    plt.axvline(df_clean['text_length'].mean(), color='red', linestyle='--', label=f'Mean: {df_clean["text_length"].mean():.0f}')
    plt.legend()

    plt.subplot(1, 2, 2)
    # Log scale for better visualization if there are outliers
    plt.hist(df_clean['text_length'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black', log=True)
    plt.title('Email Text Lengths (Log Scale)', fontweight='bold')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency (log)')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}text_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Text length by email type
    if 'email_type' in df_clean.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_clean, x='email_type', y='text_length', palette=['#ff6b6b', '#51cf66'])
        plt.title('Text Length Distribution by Email Type', fontweight='bold')
        plt.xlabel('Email Type')
        plt.ylabel('Text Length (characters)')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}text_length_by_type.png', dpi=300, bbox_inches='tight')
        plt.show()

# ============================================
# SAMPLE EMAILS
# ============================================
print("\n📧 Step 6: Sample Email Analysis")
print("-" * 70)

if 'email_type' in df_clean.columns and text_column:
    print("\n🔍 Sample Phishing Emails:")
    phishing_samples = df_clean[df_clean['email_type'] == 'Phishing Email'].head(3)
    for idx, (_, row) in enumerate(phishing_samples.iterrows()):
        print(f"\n--- Phishing Sample {idx+1} ---")
        if 'subject' in df_clean.columns:
            print(f"Subject: {row['subject']}")
        sample_text = str(row['email_text'])[:300] + "..." if len(str(row['email_text'])) > 300 else str(row['email_text'])
        print(f"Text: {sample_text}")
        print(f"Length: {len(str(row['email_text']))} characters")

    print("\n🔍 Sample Safe Emails:")
    safe_samples = df_clean[df_clean['email_type'] == 'Safe Email'].head(3)
    for idx, (_, row) in enumerate(safe_samples.iterrows()):
        print(f"\n--- Safe Sample {idx+1} ---")
        if 'subject' in df_clean.columns:
            print(f"Subject: {row['subject']}")
        sample_text = str(row['email_text'])[:300] + "..." if len(str(row['email_text'])) > 300 else str(row['email_text'])
        print(f"Text: {sample_text}")
        print(f"Length: {len(str(row['email_text']))} characters")

# ============================================
# TEXT ANALYSIS
# ============================================
print("\n🔤 Step 7: Text Analysis")
print("-" * 70)

if 'email_type' in df_clean.columns and text_column:

    def get_common_words(text_series, n=20):
        """Get most common words from text series"""
        all_words = []
        for text in text_series:
            if pd.notna(text) and isinstance(text, str):
                # Use regex to find words with at least 3 alphanumeric characters
                words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', text.lower())
                # Filter out words that are only numbers
                words = [word for word in words if not word.isdigit()]
                all_words.extend(words)
        return Counter(all_words).most_common(n)

    # Get common words by email type
    phishing_texts = df_clean[df_clean['email_type'] == 'Phishing Email']['email_text']
    safe_texts = df_clean[df_clean['email_type'] == 'Safe Email']['email_text']

    phishing_words = get_common_words(phishing_texts)
    safe_words = get_common_words(safe_texts)

    print("\n📊 Most Common Words in Phishing Emails:")
    for word, count in phishing_words[:10]:
        print(f"  {word}: {count}")

    print("\n📊 Most Common Words in Safe Emails:")
    for word, count in safe_words[:10]:
        print(f"  {word}: {count}")

    # Create word frequency plots
    if phishing_words and safe_words:
        # Phishing words
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 1, 1)
        words, counts = zip(*phishing_words[:15])
        plt.barh(words, counts, color='#ff6b6b')
        plt.title('Most Common Words - Phishing Emails', fontweight='bold')
        plt.xlabel('Frequency')

        plt.subplot(2, 1, 2)
        words, counts = zip(*safe_words[:15])
        plt.barh(words, counts, color='#51cf66')
        plt.title('Most Common Words - Safe Emails', fontweight='bold')
        plt.xlabel('Frequency')

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}word_frequency.png', dpi=300, bbox_inches='tight')
        plt.show()

# ============================================
# URL ANALYSIS (if available)
# ============================================
print("\n🌐 Step 8: URL Analysis")
print("-" * 70)

if 'urls' in df_clean.columns:
    # Count URLs
    df_clean['url_count'] = df_clean['urls'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)

    print(f"📊 URL Statistics:")
    print(f"  Emails with URLs: {(df_clean['url_count'] > 0).sum()}")
    print(f"  Average URLs per email: {df_clean['url_count'].mean():.2f}")
    print(f"  Max URLs in one email: {df_clean['url_count'].max()}")

    # URL count by email type
    if 'email_type' in df_clean.columns:
        url_by_type = df_clean.groupby('email_type')['url_count'].mean()
        print(f"\n📊 Average URLs by Email Type:")
        for email_type, avg_urls in url_by_type.items():
            print(f"  {email_type}: {avg_urls:.2f} URLs")

# ============================================
# SAVE SUMMARY STATISTICS
# ============================================
print("\n💾 Step 9: Saving Summary Statistics")
print("-" * 70)

# Create comprehensive summary
summary_file = f"{OUTPUT_DIR}dataset_summary.txt"

with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("PHISHGUARD - DATASET SUMMARY REPORT\n")
    f.write("=" * 50 + "\n\n")

    f.write("BASIC INFORMATION:\n")
    f.write(f"- Total emails: {len(df):,}\n")
    f.write(f"- Number of features: {len(df.columns)}\n")
    f.write(f"- Dataset shape: {df.shape}\n\n")

    f.write("COLUMNS:\n")
    for col in df.columns:
        f.write(f"- {col}: {df[col].dtype} | Unique: {df[col].nunique()}\n")
    f.write("\n")

    if 'email_type' in df_clean.columns:
        f.write("LABEL DISTRIBUTION:\n")
        for label, count in label_counts.items():
            percentage = label_percentages[label]
            f.write(f"- {label}: {count} ({percentage:.2f}%)\n")
        f.write("\n")

    if text_column:
        f.write("TEXT STATISTICS:\n")
        f.write(f"- Mean text length: {df_clean['text_length'].mean():.0f} chars\n")
        f.write(f"- Median text length: {df_clean['text_length'].median():.0f} chars\n")
        f.write(f"- Min text length: {df_clean['text_length'].min()} chars\n")
        f.write(f"- Max text length: {df_clean['text_length'].max()} chars\n")
        f.write(f"- Mean word count: {df_clean['word_count'].mean():.0f} words\n\n")

    f.write("DATA QUALITY:\n")
    f.write(f"- Missing values: {df.isnull().sum().sum()}\n")
    f.write(f"- Duplicate rows: {duplicates}\n")

    if 'urls' in df_clean.columns:
        f.write(f"- Emails with URLs: {(df_clean['url_count'] > 0).sum()}\n")

print(f"✅ Summary saved to: {summary_file}")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 70)
print("✅ DATA EXPLORATION COMPLETE!")
print("=" * 70)

print(f"\n📊 EXPLORATION SUMMARY:")
print(f"   • Dataset: {len(df):,} emails, {len(df.columns)} features")
if 'email_type' in df_clean.columns:
    phishing_count = (df_clean['email_type'] == 'Phishing Email').sum()
    safe_count = (df_clean['email_type'] == 'Safe Email').sum()
    print(f"   • Phishing emails: {phishing_count}")
    print(f"   • Safe emails: {safe_count}")
print(f"   • Visualizations saved: {OUTPUT_DIR}")
print(f"   • Summary report: {summary_file}")

print("\n🚀 NEXT STEPS:")
print("   1. Review the dataset summary")
print("   2. Check data quality issues")
print("   3. Run: src/data_preprocessing.py")

print("\n💡 RECOMMENDATIONS:")
if duplicates > 0:
    print("   ⚠️  Consider removing duplicate rows")
if df.isnull().sum().sum() > 0:
    print("   ⚠️  Handle missing values in preprocessing")
if 'email_type' not in df_clean.columns:
    print("   ⚠️  Verify label column mapping")

print("=" * 70)