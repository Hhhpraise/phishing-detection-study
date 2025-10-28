"""
PhishGuard - Select Test Emails for Web Interface
Curate 15-20 diverse emails from test set for user testing
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PHISHGUARD - SELECT TEST EMAILS FOR USER STUDY")
print("=" * 70)

# ============================================
# CONFIGURATION
# ============================================
print("\n⚙️ Configuration")
print("-" * 70)

# Paths
MODEL_PATH = "../models/phishing_model_improved.pth"
TEST_DATA_PATH = "../data/processed/test.csv"
VECTORIZER_PATH = "../models/tfidf_vectorizer.pkl"
FEATURE_INFO_PATH = "../models/feature_info.pkl"
OUTPUT_FILE = "../data/processed/test_emails_selected.json"

print(f"Model: {MODEL_PATH}")
print(f"Test data: {TEST_DATA_PATH}")
print(f"Output: {OUTPUT_FILE}")

# ============================================
# HELPER FUNCTIONS
# ============================================
def clean_email_display(text):
    """Clean email text for web display"""
    if not isinstance(text, str):
        return ""

    # Limit length for display (keep first 1000 chars)
    display_text = text[:1000]

    # Add ellipsis if truncated
    if len(text) > 1000:
        display_text += "..."

    # Basic cleaning for display
    display_text = display_text.replace('\n', ' ').replace('\r', ' ')
    display_text = ' '.join(display_text.split())  # Normalize whitespace

    return display_text

# ============================================
# LOAD MODEL AND DATA
# ============================================
print("\n📂 Step 1: Loading Model and Test Data")
print("-" * 70)

# Load test data
test_df = pd.read_csv(TEST_DATA_PATH)
print(f"✅ Test data loaded: {len(test_df):,} emails")

# Load vectorizer and feature info
vectorizer = joblib.load(VECTORIZER_PATH)
feature_info = joblib.load(FEATURE_INFO_PATH)
print(f"✅ Vectorizer loaded: {len(feature_info['tfidf_features']):,} TF-IDF features")

# Define the model architecture
class ImprovedPhishingDetector(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], dropout_rate=0.4):
        super(ImprovedPhishingDetector, self).__init__()
        self.input_bn = nn.BatchNorm1d(input_size)
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_sizes[i]))
        self.output_layer = nn.Linear(hidden_sizes[-1], 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.relu(self.input_layer(x))
        x = self.bn1(x)
        x = self.dropout(x)
        for i in range(0, len(self.hidden_layers), 2):
            x = self.relu(self.hidden_layers[i](x))
            x = self.hidden_layers[i+1](x)
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

# Initialize and load model
input_size = 5010
model = ImprovedPhishingDetector(input_size=input_size)
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded successfully")
print(f"   Model architecture: {[512, 256, 128, 64]} → 2 output classes")

# ============================================
# PREPARE TEST DATA
# ============================================
print("\n🔧 Step 2: Preparing Test Data for Analysis")
print("-" * 70)

# Convert text to TF-IDF features
X_test_tfidf = vectorizer.transform(test_df['email_text_clean']).astype(np.float32)

# Get engineered features
engineered_features = feature_info['engineered_features']
X_test_engineered = test_df[engineered_features].values.astype(np.float32)

# Combine features
X_test_combined = np.hstack([X_test_tfidf.toarray(), X_test_engineered])

# Prepare labels and texts
y_test = test_df['label'].values.astype(np.int64)
email_texts = test_df['email_text'].values
email_subjects = test_df.get('subject', pd.Series([''] * len(test_df))).values

print(f"Test features: {X_test_combined.shape}")
print(f"Test labels: {y_test.shape}")
print(f"Class distribution: Safe={sum(y_test == 0)}, Phishing={sum(y_test == 1)}")

# ============================================
# GENERATE PREDICTIONS FOR ALL TEST EMAILS
# ============================================
print("\n🎯 Step 3: Generating Predictions for All Test Emails")
print("-" * 70)

test_predictions = []
test_probabilities = []
test_confidence = []

# Convert to torch tensors
X_test_tensor = torch.FloatTensor(X_test_combined)

# Predict in batches
batch_size = 256
with torch.no_grad():
    for i in tqdm(range(0, len(X_test_tensor), batch_size), desc="Generating predictions"):
        batch = X_test_tensor[i:i+batch_size]
        outputs = model(batch)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)

        test_predictions.extend(predictions.cpu().numpy())
        test_probabilities.extend(probabilities.cpu().numpy())
        # Get confidence as max probability
        confidence = torch.max(probabilities, dim=1)[0]
        test_confidence.extend(confidence.cpu().numpy())

test_predictions = np.array(test_predictions)
test_probabilities = np.array(test_probabilities)
test_confidence = np.array(test_confidence)

# Get probabilities for class 1 (phishing)
test_phishing_probs = test_probabilities[:, 1]

print(f"✅ Predictions generated for {len(test_predictions):,} test emails")

# ============================================
# CLASSIFY EMAILS BY DIFFICULTY
# ============================================
print("\n📊 Step 4: Classifying Emails by Difficulty")
print("-" * 70)

# Calculate correctness
correct_predictions = (test_predictions == y_test)

# Classify by difficulty
difficulty_levels = []
for i in range(len(test_predictions)):
    confidence = test_confidence[i]
    is_correct = correct_predictions[i]
    phishing_prob = test_phishing_probs[i]

    if is_correct:
        if confidence > 0.8 or confidence < 0.2:
            difficulty = "easy"
        elif confidence > 0.6 or confidence < 0.4:
            difficulty = "medium"
        else:
            difficulty = "hard"
    else:
        difficulty = "hard"  # Wrong predictions are always hard

    difficulty_levels.append(difficulty)

# Add to dataframe
test_df = test_df.copy()
test_df['ai_prediction'] = test_predictions
test_df['ai_confidence'] = test_confidence
test_df['ai_phishing_prob'] = test_phishing_probs
test_df['correct_prediction'] = correct_predictions
test_df['difficulty'] = difficulty_levels

# Count difficulty levels
difficulty_counts = pd.Series(difficulty_levels).value_counts()
print("📊 Difficulty Distribution:")
for level, count in difficulty_counts.items():
    print(f"   {level}: {count} emails")

# ============================================
# SELECT BALANCED SET OF TEST EMAILS
# ============================================
print("\n🎯 Step 5: Selecting Balanced Test Email Set")
print("-" * 70)

# Target: 20 emails total
TARGET_TOTAL = 20
TARGET_PER_CLASS = TARGET_TOTAL // 2  # 10 safe, 10 phishing

selected_emails = []

# Select emails for each label and difficulty combination
for label in [0, 1]:  # 0 = Safe, 1 = Phishing
    label_name = "Safe Email" if label == 0 else "Phishing Email"
    print(f"\n🔍 Selecting {TARGET_PER_CLASS} {label_name} emails:")

    label_mask = (test_df['label'] == label)
    label_df = test_df[label_mask]

    # Adjust targets based on available difficulty distribution
    available_difficulties = label_df['difficulty'].value_counts()

    # Calculate proportional targets
    difficulty_targets = {}
    total_available = min(TARGET_PER_CLASS, len(label_df))

    for difficulty in ['easy', 'medium', 'hard']:
        available = available_difficulties.get(difficulty, 0)
        if available > 0:
            # Allocate proportionally, but ensure at least 1 of each available difficulty
            proportion = available / len(label_df)
            target = max(1, int(total_available * proportion))
            difficulty_targets[difficulty] = min(target, available)

    # Adjust if we allocated too many
    total_allocated = sum(difficulty_targets.values())
    if total_allocated > total_available:
        # Reduce from easiest category first
        for difficulty in ['easy', 'medium', 'hard']:
            if difficulty in difficulty_targets and difficulty_targets[difficulty] > 1:
                reduction = min(difficulty_targets[difficulty] - 1, total_allocated - total_available)
                difficulty_targets[difficulty] -= reduction
                total_allocated -= reduction
                if total_allocated <= total_available:
                    break

    for difficulty, target_count in difficulty_targets.items():
        difficulty_mask = (label_df['difficulty'] == difficulty)
        available_emails = label_df[difficulty_mask]

        if len(available_emails) > 0:
            # Select representative samples
            if len(available_emails) <= target_count:
                selected = available_emails
            else:
                # Stratified sampling by text length
                selected = available_emails.sample(n=target_count, random_state=42)

            print(f"   • {difficulty}: {len(selected)} emails")

            for _, email in selected.iterrows():
                selected_emails.append({
                    'id': f"email_{len(selected_emails):03d}",
                    'text': clean_email_display(email['email_text']),
                    'subject': str(email.get('subject', ''))[:100],
                    'ground_truth': "safe" if label == 0 else "phishing",
                    'ai_prediction': "safe" if email['ai_prediction'] == 0 else "phishing",
                    'ai_confidence': float(email['ai_confidence']),
                    'ai_phishing_prob': float(email['ai_phishing_prob']),
                    'difficulty': difficulty,
                    'correct_prediction': bool(email['correct_prediction']),
                    'text_length': len(str(email['email_text'])),
                    'word_count': len(str(email['email_text']).split())
                })

# If we didn't get enough emails, add more
if len(selected_emails) < TARGET_TOTAL:
    remaining = TARGET_TOTAL - len(selected_emails)
    print(f"\n⚠️  Only selected {len(selected_emails)} emails. Adding {remaining} more...")

    # Add random emails to reach target
    remaining_emails = test_df[~test_df.index.isin([email.get('original_index', -1) for email in selected_emails])]
    if len(remaining_emails) > 0:
        additional = remaining_emails.sample(n=min(remaining, len(remaining_emails)), random_state=42)
        for _, email in additional.iterrows():
            selected_emails.append({
                'id': f"email_{len(selected_emails):03d}",
                'text': clean_email_display(email['email_text']),
                'subject': str(email.get('subject', ''))[:100],
                'ground_truth': "safe" if email['label'] == 0 else "phishing",
                'ai_prediction': "safe" if email['ai_prediction'] == 0 else "phishing",
                'ai_confidence': float(email['ai_confidence']),
                'ai_phishing_prob': float(email['ai_phishing_prob']),
                'difficulty': email['difficulty'],
                'correct_prediction': bool(email['correct_prediction']),
                'text_length': len(str(email['email_text'])),
                'word_count': len(str(email['email_text']).split())
            })

# Shuffle the selected emails
random.shuffle(selected_emails)

print(f"\n✅ Selected {len(selected_emails)} emails for user testing")

# ============================================
# ANALYZE SELECTED EMAIL SET
# ============================================
print("\n📊 Step 6: Analyzing Selected Email Set")
print("-" * 70)

# Create summary of selected emails
selected_df = pd.DataFrame(selected_emails)

print("📈 Selected Email Distribution:")
print(f"   Total emails: {len(selected_emails)}")
print(f"   Safe emails: {sum(1 for e in selected_emails if e['ground_truth'] == 'safe')}")
print(f"   Phishing emails: {sum(1 for e in selected_emails if e['ground_truth'] == 'phishing')}")

print(f"\n🎯 Difficulty Distribution:")
difficulty_counts_selected = selected_df['difficulty'].value_counts()
for level, count in difficulty_counts_selected.items():
    print(f"   {level}: {count} emails")

print(f"\n🤖 AI Performance on Selected Set:")
correct_count = sum(1 for e in selected_emails if e['correct_prediction'])
ai_accuracy = correct_count / len(selected_emails) if len(selected_emails) > 0 else 0
print(f"   AI Accuracy: {ai_accuracy:.2%} ({correct_count}/{len(selected_emails)})")

print(f"\n📏 Text Length Statistics:")
if len(selected_emails) > 0:
    avg_length = selected_df['text_length'].mean()
    avg_words = selected_df['word_count'].mean()
    print(f"   Average length: {avg_length:.0f} characters")
    print(f"   Average words: {avg_words:.0f} words")
    print(f"   Min length: {selected_df['text_length'].min()} characters")
    print(f"   Max length: {selected_df['text_length'].max()} characters")
else:
    avg_length = avg_words = 0
    print("   No emails selected for analysis")

# ============================================
# SAVE SELECTED EMAILS TO JSON
# ============================================
print("\n💾 Step 7: Saving Selected Emails to JSON")
print("-" * 70)

# Create the final output structure
output_data = {
    "metadata": {
        "total_emails": len(selected_emails),
        "selection_criteria": {
            "safe_emails": sum(1 for e in selected_emails if e['ground_truth'] == 'safe'),
            "phishing_emails": sum(1 for e in selected_emails if e['ground_truth'] == 'phishing'),
            "easy_emails": sum(1 for e in selected_emails if e['difficulty'] == 'easy'),
            "medium_emails": sum(1 for e in selected_emails if e['difficulty'] == 'medium'),
            "hard_emails": sum(1 for e in selected_emails if e['difficulty'] == 'hard')
        },
        "ai_performance": {
            "accuracy": ai_accuracy,
            "correct_predictions": correct_count,
            "total_predictions": len(selected_emails)
        },
        "text_statistics": {
            "avg_length": float(avg_length),
            "avg_words": float(avg_words),
            "min_length": float(selected_df['text_length'].min() if len(selected_emails) > 0 else 0),
            "max_length": float(selected_df['text_length'].max() if len(selected_emails) > 0 else 0)
        }
    },
    "emails": selected_emails
}

# Save to JSON file
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"✅ Selected emails saved: {OUTPUT_FILE}")

# ============================================
# CREATE SELECTION REPORT
# ============================================
print("\n📋 Step 8: Creating Selection Report")
print("-" * 70)

report_file = "../data/processed/email_selection_report.txt"

with open(report_file, 'w') as f:
    f.write("PHISHGUARD - TEST EMAIL SELECTION REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write("SELECTION SUMMARY:\n")
    f.write(f"  Total emails selected: {len(selected_emails)}\n")
    f.write(f"  Safe emails: {sum(1 for e in selected_emails if e['ground_truth'] == 'safe')}\n")
    f.write(f"  Phishing emails: {sum(1 for e in selected_emails if e['ground_truth'] == 'phishing')}\n\n")

    f.write("DIFFICULTY DISTRIBUTION:\n")
    for level in ['easy', 'medium', 'hard']:
        count = sum(1 for e in selected_emails if e['difficulty'] == level)
        f.write(f"  {level}: {count} emails\n")
    f.write("\n")

    f.write("AI PERFORMANCE ON SELECTED SET:\n")
    f.write(f"  Accuracy: {ai_accuracy:.2%}\n")
    f.write(f"  Correct predictions: {correct_count}/{len(selected_emails)}\n\n")

    if len(selected_emails) > 0:
        f.write("TEXT CHARACTERISTICS:\n")
        f.write(f"  Average length: {avg_length:.0f} characters\n")
        f.write(f"  Average words: {avg_words:.0f} words\n")
        f.write(f"  Length range: {selected_df['text_length'].min()} - {selected_df['text_length'].max()} characters\n\n")

    f.write("SAMPLE EMAILS:\n")
    for i, email in enumerate(selected_emails[:3]):
        f.write(f"  Email {i+1}:\n")
        f.write(f"    ID: {email['id']}\n")
        f.write(f"    Ground truth: {email['ground_truth']}\n")
        f.write(f"    AI prediction: {email['ai_prediction']} (confidence: {email['ai_confidence']:.3f})\n")
        f.write(f"    Difficulty: {email['difficulty']}\n")
        f.write(f"    Correct: {email['correct_prediction']}\n")
        f.write(f"    Preview: {email['text'][:100]}...\n\n")

print(f"✅ Selection report saved: {report_file}")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 70)
print("✅ TEST EMAIL SELECTION COMPLETE!")
print("=" * 70)

print(f"\n🎯 SELECTION SUMMARY:")
print(f"   • Total emails selected: {len(selected_emails)}")
print(f"   • Safe emails: {sum(1 for e in selected_emails if e['ground_truth'] == 'safe')}")
print(f"   • Phishing emails: {sum(1 for e in selected_emails if e['ground_truth'] == 'phishing')}")
print(f"   • Difficulty mix: Easy={difficulty_counts_selected.get('easy', 0)}, Medium={difficulty_counts_selected.get('medium', 0)}, Hard={difficulty_counts_selected.get('hard', 0)}")
print(f"   • AI accuracy on selection: {ai_accuracy:.2%}")

if len(selected_emails) > 0:
    print(f"\n📊 EMAIL CHARACTERISTICS:")
    print(f"   • Average length: {avg_length:.0f} characters")
    print(f"   • Average words: {avg_words:.0f} words")
    print(f"   • Length range: {selected_df['text_length'].min()} - {selected_df['text_length'].max()} chars")

print(f"\n💾 OUTPUT FILES:")
print(f"   • Selected emails: {OUTPUT_FILE}")
print(f"   • Selection report: {report_file}")

print(f"\n🚀 NEXT STEPS:")
print(f"   1. Review the selected email set")
print(f"   2. Proceed to Task 8: Microsoft Excel API Setup")
print(f"   3. The emails are ready for user testing!")

print(f"\n🎉 SUCCESS: Curated {len(selected_emails)} diverse test emails!")
print("=" * 70)