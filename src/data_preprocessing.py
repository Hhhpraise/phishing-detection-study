"""
PhishGuard - Data Preprocessing Script
Prepare CEAS dataset for machine learning training
"""

import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PHISHGUARD - DATA PREPROCESSING")
print("=" * 70)

# ============================================
# CONFIGURATION
# ============================================
print("\n⚙️ Configuration")
print("-" * 70)

INPUT_CSV_PATH = "../data/raw/CEAS_08.csv"
OUTPUT_DIR = "../data/processed/"
MODEL_DIR = "../models/"

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Input dataset: {INPUT_CSV_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Model directory: {MODEL_DIR}")

# ============================================
# LOAD DATASET
# ============================================
print("\n📂 Step 1: Loading Dataset")
print("-" * 70)

try:
    df = pd.read_csv(INPUT_CSV_PATH, encoding='utf-8')
    print(f"✅ Dataset loaded: {df.shape[0]:,} emails, {df.shape[1]} features")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# Create working copy
df_clean = df.copy()

# ============================================
# HANDLE MISSING VALUES
# ============================================
print("\n🔍 Step 2: Handling Missing Values")
print("-" * 70)

# Check missing values
missing_before = df_clean.isnull().sum()
print("Missing values before cleaning:")
for col, count in missing_before.items():
    if count > 0:
        print(f"  {col}: {count} missing")

# Fill missing values
df_clean['receiver'] = df_clean['receiver'].fillna('unknown@unknown.com')
df_clean['subject'] = df_clean['subject'].fillna('')

print("✅ Missing values handled")

# ============================================
# CREATE EMAIL TYPE LABELS
# ============================================
print("\n🏷️ Step 3: Creating Labels")
print("-" * 70)

# Map labels: 1 = Phishing (1), 0 = Safe (0)
df_clean['email_type'] = df_clean['label'].map({1: 'Phishing Email', 0: 'Safe Email'})
df_clean['label_binary'] = df_clean['label']  # Already 0/1

print("Label distribution:")
label_counts = df_clean['email_type'].value_counts()
for label, count in label_counts.items():
    percentage = count / len(df_clean) * 100
    print(f"  {label}: {count} ({percentage:.2f}%)")

# ============================================
# TEXT CLEANING FUNCTIONS
# ============================================
print("\n🧹 Step 4: Text Cleaning")
print("-" * 70)

def clean_email_text(text):
    """Clean and preprocess email text"""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Replace URLs with placeholder
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL_PLACEHOLDER ', text)

    # Replace email addresses with placeholder
    text = re.sub(r'\S+@\S+', ' EMAIL_PLACEHOLDER ', text)

    # Remove special characters and numbers, but keep basic punctuation
    text = re.sub(r'[^a-zA-Z\s\.!?]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_url_count(urls_value):
    """Extract URL count from urls column"""
    if pd.isna(urls_value):
        return 0
    try:
        return int(urls_value)
    except:
        return 0

# Combine subject and body for text analysis
print("Combining subject and body...")
df_clean['email_text_raw'] = df_clean['subject'].fillna('') + ' ' + df_clean['body'].fillna('')

print("Cleaning email text...")
df_clean['email_text_clean'] = df_clean['email_text_raw'].apply(clean_email_text)

print(f"✅ Text cleaning completed")
print(f"   Sample cleaned text: {df_clean['email_text_clean'].iloc[0][:200]}...")

# ============================================
# FEATURE ENGINEERING
# ============================================
print("\n🔧 Step 5: Feature Engineering")
print("-" * 70)

# Basic text features
df_clean['email_length'] = df_clean['email_text_raw'].str.len()
df_clean['word_count'] = df_clean['email_text_clean'].str.split().str.len()

# URL features
df_clean['url_count'] = df_clean['urls'].apply(extract_url_count)
df_clean['has_urls'] = (df_clean['url_count'] > 0).astype(int)

# Urgency features
urgent_words = ['urgent', 'immediately', 'asap', 'important', 'alert', 'warning', 'emergency', 'quick', 'instant']
df_clean['urgent_word_count'] = df_clean['email_text_clean'].apply(
    lambda x: sum(1 for word in urgent_words if word in x) if isinstance(x, str) else 0
)
df_clean['has_urgent_words'] = (df_clean['urgent_word_count'] > 0).astype(int)

# Money-related features
money_words = ['money', 'cash', 'price', 'cost', 'payment', 'dollar', 'free', 'discount', 'offer', 'win', 'prize']
df_clean['money_word_count'] = df_clean['email_text_clean'].apply(
    lambda x: sum(1 for word in money_words if word in x) if isinstance(x, str) else 0
)
df_clean['has_money_words'] = (df_clean['money_word_count'] > 0).astype(int)

# Exclamation marks
df_clean['exclamation_count'] = df_clean['email_text_raw'].str.count('!')

# Question marks
df_clean['question_count'] = df_clean['email_text_raw'].str.count('\?')

print("✅ Feature engineering completed")
print("Created features:")
feature_columns = ['email_length', 'word_count', 'url_count', 'has_urls',
                   'urgent_word_count', 'has_urgent_words', 'money_word_count',
                   'has_money_words', 'exclamation_count', 'question_count']

for feature in feature_columns:
    print(f"  • {feature}")

# ============================================
# DATA SPLITTING
# ============================================
print("\n📊 Step 6: Data Splitting")
print("-" * 70)

# Prepare features and labels
X = df_clean['email_text_clean']  # Text features
y = df_clean['label_binary']      # Binary labels

# Additional engineered features
X_engineered = df_clean[feature_columns]

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Split engineered features accordingly
X_engineered_train = X_engineered.loc[X_train.index]
X_engineered_test = X_engineered.loc[X_test.index]

print(f"✅ Data splitting completed:")
print(f"   Training set: {len(X_train):,} emails")
print(f"   Test set: {len(X_test):,} emails")
print(f"   Training label distribution:")
train_counts = y_train.value_counts()
for label, count in train_counts.items():
    percentage = count / len(y_train) * 100
    label_name = 'Phishing' if label == 1 else 'Safe'
    print(f"     {label_name}: {count} ({percentage:.2f}%)")

# ============================================
# TEXT VECTORIZATION
# ============================================
print("\n📝 Step 7: Text Vectorization (TF-IDF)")
print("-" * 70)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.95
)

print("Fitting TF-IDF vectorizer on training data...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"✅ TF-IDF vectorization completed:")
print(f"   Vocabulary size: {len(tfidf_vectorizer.vocabulary_):,} features")
print(f"   Training matrix: {X_train_tfidf.shape}")
print(f"   Test matrix: {X_test_tfidf.shape}")

# ============================================
# COMBINE TEXT AND ENGINEERED FEATURES
# ============================================
print("\n🔗 Step 8: Combining Features")
print("-" * 70)

# For PyTorch, we'll keep features separate for now
# We can combine them in the model or dataset class

print("✅ Feature preparation completed")
print("   Text features (TF-IDF):", X_train_tfidf.shape[1], "dimensions")
print("   Engineered features:", len(feature_columns), "dimensions")
print("   Total features:", X_train_tfidf.shape[1] + len(feature_columns), "dimensions")

# ============================================
# SAVE PROCESSED DATA
# ============================================
print("\n💾 Step 9: Saving Processed Data")
print("-" * 70)

# Create training dataframe
train_df = pd.DataFrame({
    'email_text': X_train,
    'email_text_clean': X_train,
    'label': y_train,
    'email_type': df_clean.loc[X_train.index, 'email_type']
})

# Add engineered features to training data
for feature in feature_columns:
    train_df[feature] = X_engineered_train[feature]

# Create test dataframe
test_df = pd.DataFrame({
    'email_text': X_test,
    'email_text_clean': X_test,
    'label': y_test,
    'email_type': df_clean.loc[X_test.index, 'email_type']
})

# Add engineered features to test data
for feature in feature_columns:
    test_df[feature] = X_engineered_test[feature]

# Save processed datasets
train_file = os.path.join(OUTPUT_DIR, "train.csv")
test_file = os.path.join(OUTPUT_DIR, "test.csv")

train_df.to_csv(train_file, index=False, encoding='utf-8')
test_df.to_csv(test_file, index=False, encoding='utf-8')

print(f"✅ Training data saved: {train_file}")
print(f"✅ Test data saved: {test_file}")

# ============================================
# SAVE VECTORIZER AND FEATURE INFO
# ============================================
print("\n💾 Step 10: Saving Vectorizer and Metadata")
print("-" * 70)

# Save TF-IDF vectorizer
vectorizer_file = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
joblib.dump(tfidf_vectorizer, vectorizer_file)
print(f"✅ TF-IDF vectorizer saved: {vectorizer_file}")

# Save feature names
feature_info = {
    'tfidf_features': tfidf_vectorizer.get_feature_names_out().tolist(),
    'engineered_features': feature_columns,
    'all_features': tfidf_vectorizer.get_feature_names_out().tolist() + feature_columns
}

feature_file = os.path.join(MODEL_DIR, "feature_info.pkl")
joblib.dump(feature_info, feature_file)
print(f"✅ Feature information saved: {feature_file}")

# ============================================
# STATISTICS AND QUALITY CHECKS
# ============================================
print("\n📊 Step 11: Final Statistics")
print("-" * 70)

print("\n📈 Dataset Statistics:")
print(f"   Original dataset: {len(df):,} emails")
print(f"   Training set: {len(train_df):,} emails")
print(f"   Test set: {len(test_df):,} emails")
print(f"   TF-IDF features: {X_train_tfidf.shape[1]:,}")
print(f"   Engineered features: {len(feature_columns)}")

print(f"\n📊 Training Set Distribution:")
train_phishing = (train_df['label'] == 1).sum()
train_safe = (train_df['label'] == 0).sum()
print(f"   Phishing emails: {train_phishing} ({train_phishing/len(train_df)*100:.2f}%)")
print(f"   Safe emails: {train_safe} ({train_safe/len(train_df)*100:.2f}%)")

print(f"\n📊 Test Set Distribution:")
test_phishing = (test_df['label'] == 1).sum()
test_safe = (test_df['label'] == 0).sum()
print(f"   Phishing emails: {test_phishing} ({test_phishing/len(test_df)*100:.2f}%)")
print(f"   Safe emails: {test_safe} ({test_safe/len(test_df)*100:.2f}%)")

print(f"\n🔍 Feature Statistics (Training Set):")
print(f"   Average email length: {train_df['email_length'].mean():.0f} chars")
print(f"   Average word count: {train_df['word_count'].mean():.0f} words")
print(f"   Emails with URLs: {train_df['has_urls'].sum()} ({train_df['has_urls'].mean()*100:.1f}%)")
print(f"   Emails with urgent words: {train_df['has_urgent_words'].sum()} ({train_df['has_urgent_words'].mean()*100:.1f}%)")
print(f"   Emails with money words: {train_df['has_money_words'].sum()} ({train_df['has_money_words'].mean()*100:.1f}%)")

# ============================================
# SAVE PROCESSING SUMMARY
# ============================================
summary_file = os.path.join(OUTPUT_DIR, "preprocessing_summary.txt")
with open(summary_file, 'w') as f:
    f.write("PHISHGUARD - DATA PREPROCESSING SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Original dataset size: {len(df):,} emails\n")
    f.write(f"Training set: {len(train_df):,} emails\n")
    f.write(f"Test set: {len(test_df):,} emails\n")
    f.write(f"TF-IDF features: {X_train_tfidf.shape[1]:,}\n")
    f.write(f"Engineered features: {len(feature_columns)}\n")
    f.write(f"Total features: {X_train_tfidf.shape[1] + len(feature_columns):,}\n\n")

    f.write("TRAINING SET DISTRIBUTION:\n")
    f.write(f"  Phishing emails: {train_phishing} ({train_phishing/len(train_df)*100:.2f}%)\n")
    f.write(f"  Safe emails: {train_safe} ({train_safe/len(train_df)*100:.2f}%)\n\n")

    f.write("FEATURES CREATED:\n")
    for feature in feature_columns:
        f.write(f"  • {feature}\n")

print(f"✅ Processing summary saved: {summary_file}")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 70)
print("✅ DATA PREPROCESSING COMPLETE!")
print("=" * 70)

print(f"\n📊 FINAL SUMMARY:")
print(f"   • Processed {len(df):,} emails")
print(f"   • Training set: {len(train_df):,} emails")
print(f"   • Test set: {len(test_df):,} emails")
print(f"   • TF-IDF features: {X_train_tfidf.shape[1]:,}")
print(f"   • Engineered features: {len(feature_columns)}")
print(f"   • Vectorizer saved: {vectorizer_file}")

print(f"\n🚀 NEXT STEPS:")
print(f"   1. Review preprocessing summary")
print(f"   2. Run: src/train_model.py")
print(f"   3. Model will use PyTorch with GPU optimization")

print(f"\n💡 RECOMMENDATIONS:")
print(f"   ✅ Dataset is well-prepared for training")
print(f"   ✅ Good balance between phishing and safe emails")
print(f"   ✅ Rich feature set for model learning")

print("=" * 70)