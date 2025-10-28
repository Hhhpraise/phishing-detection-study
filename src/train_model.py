"""
PhishGuard - Improved Model Training with Class Balancing
Train phishing detection model with proper class balancing and improved architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PHISHGUARD - IMPROVED MODEL TRAINING")
print("=" * 70)

# ============================================
# CONFIGURATION
# ============================================
print("\n⚙️ Configuration")
print("-" * 70)

# Paths
TRAIN_DATA_PATH = "../data/processed/train.csv"
TEST_DATA_PATH = "../data/processed/test.csv"
VECTORIZER_PATH = "../models/tfidf_vectorizer.pkl"
FEATURE_INFO_PATH = "../models/feature_info.pkl"
MODEL_SAVE_PATH = "../models/phishing_model_improved.pth"
OUTPUT_DIR = "../models/"

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.0005  # Reduced learning rate
EPOCHS = 100
PATIENCE = 10  # Increased patience

print(f"Training data: {TRAIN_DATA_PATH}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")

# ============================================
# GPU SETUP
# ============================================
print("\n🎮 Step 1: GPU Configuration")
print("-" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.cuda.empty_cache()
else:
    print("⚠️  CUDA not available. Training on CPU (slower)")

# ============================================
# LOAD PROCESSED DATA
# ============================================
print("\n📂 Step 2: Loading Processed Data")
print("-" * 70)

try:
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    print(f"✅ Training data loaded: {len(train_df):,} emails")
    print(f"✅ Test data loaded: {len(test_df):,} emails")

    # Check class distribution
    train_phishing = (train_df['label'] == 1).sum()
    train_safe = (train_df['label'] == 0).sum()
    print(f"📊 Training class distribution:")
    print(f"   Phishing: {train_phishing} ({train_phishing/len(train_df)*100:.2f}%)")
    print(f"   Safe: {train_safe} ({train_safe/len(train_df)*100:.2f}%)")

except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Load vectorizer and feature info
try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    feature_info = joblib.load(FEATURE_INFO_PATH)
    print(f"✅ Vectorizer loaded: {len(feature_info['tfidf_features']):,} TF-IDF features")
except Exception as e:
    print(f"❌ Error loading vectorizer: {e}")
    exit(1)

# ============================================
# PREPARE FEATURES AND LABELS WITH CLASS BALANCING
# ============================================
print("\n🔧 Step 3: Preparing Features with Class Balancing")
print("-" * 70)

# Convert text to TF-IDF features
print("Converting text to TF-IDF features...")
X_train_tfidf = vectorizer.transform(train_df['email_text_clean']).astype(np.float32)
X_test_tfidf = vectorizer.transform(test_df['email_text_clean']).astype(np.float32)

# Get engineered features
engineered_features = feature_info['engineered_features']
X_train_engineered = train_df[engineered_features].values.astype(np.float32)
X_test_engineered = test_df[engineered_features].values.astype(np.float32)

# Combine features
X_train_combined = np.hstack([X_train_tfidf.toarray(), X_train_engineered])
X_test_combined = np.hstack([X_test_tfidf.toarray(), X_test_engineered])

print(f"Combined features - Train: {X_train_combined.shape}, Test: {X_test_combined.shape}")

# Prepare labels
y_train = train_df['label'].values.astype(np.int64)  # Changed to int64 for CrossEntropy
y_test = test_df['label'].values.astype(np.int64)

print(f"Labels - Train: {y_train.shape}, Test: {y_test.shape}")

# ============================================
# COMPUTE CLASS WEIGHTS
# ============================================
print("\n⚖️ Step 4: Computing Class Weights")
print("-" * 70)

# Compute class weights for imbalance
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.FloatTensor(class_weights).to(device)

print(f"Class weights: {class_weights.cpu().numpy()}")
print(f"   Weight for class 0 (Safe): {class_weights[0]:.4f}")
print(f"   Weight for class 1 (Phishing): {class_weights[1]:.4f}")

# ============================================
# CREATE BALANCED DATASETS
# ============================================
print("\n📊 Step 5: Creating Balanced Datasets")
print("-" * 70)

class BalancedPhishingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)  # Changed to LongTensor for CrossEntropy

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create datasets
train_dataset = BalancedPhishingDataset(X_train_combined, y_train)
test_dataset = BalancedPhishingDataset(X_test_combined, y_test)

# Create weighted sampler for training to balance classes
class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in y_train])
samples_weight = torch.from_numpy(samples_weight).double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"✅ Training batches: {len(train_loader)}")
print(f"✅ Test batches: {len(test_loader)}")
print("✅ Using WeightedRandomSampler for class balancing")

# ============================================
# IMPROVED NEURAL NETWORK MODEL
# ============================================
print("\n🧠 Step 6: Defining Improved Neural Network")
print("-" * 70)

class ImprovedPhishingDetector(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.4):
        super(ImprovedPhishingDetector, self).__init__()

        self.input_bn = nn.BatchNorm1d(input_size)

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_sizes[i]))

        # Output layer - 2 classes for CrossEntropy
        self.output_layer = nn.Linear(hidden_sizes[-1], 2)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Input batch normalization
        x = self.input_bn(x)

        # Input layer
        x = self.relu(self.input_layer(x))
        x = self.bn1(x)
        x = self.dropout(x)

        # Hidden layers
        for i in range(0, len(self.hidden_layers), 2):
            x = self.relu(self.hidden_layers[i](x))
            x = self.hidden_layers[i+1](x)
            x = self.dropout(x)

        # Output layer
        x = self.output_layer(x)
        return x  # Return logits for CrossEntropyLoss

# Initialize model
input_size = X_train_combined.shape[1]
model = ImprovedPhishingDetector(input_size=input_size,
                                hidden_sizes=[512, 256, 128, 64],
                                dropout_rate=0.4)

# Move model to GPU
model = model.to(device)

print(f"✅ Improved model initialized:")
print(f"   Input size: {input_size}")
print(f"   Architecture: {[512, 256, 128, 64]} → 2 output classes")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Model device: {next(model.parameters()).device}")

# ============================================
# SETUP TRAINING WITH CLASS WEIGHTS
# ============================================
print("\n⚙️ Step 7: Setting Up Training with Class Weights")
print("-" * 70)

# Use CrossEntropyLoss with class weights for better imbalance handling
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

print(f"✅ Loss function: CrossEntropyLoss with class weights")
print(f"✅ Optimizer: AdamW (lr={LEARNING_RATE})")
print(f"✅ Learning rate scheduler: CosineAnnealingWarmRestarts")

# ============================================
# IMPROVED TRAINING FUNCTIONS
# ============================================
print("\n🚀 Step 8: Starting Improved Training")
print("-" * 70)

def train_epoch_improved(model, dataloader, criterion, optimizer, device):
    """Train for one epoch with improved metrics"""
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update statistics
        running_loss += loss.item()

        # Get predictions
        _, predictions = torch.max(output, 1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy_score(all_labels, all_predictions):.4f}'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = accuracy_score(all_labels, all_predictions)

    return epoch_loss, epoch_accuracy

def validate_epoch_improved(model, dataloader, criterion, device):
    """Validate for one epoch with comprehensive metrics"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()

            # Get predictions and probabilities
            probabilities = torch.softmax(output, dim=1)
            _, predictions = torch.max(output, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)

    # Calculate comprehensive metrics
    epoch_accuracy = accuracy_score(all_labels, all_predictions)
    epoch_precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    epoch_recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    epoch_f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)

    return (epoch_loss, epoch_accuracy, epoch_precision,
            epoch_recall, epoch_f1, all_probabilities, all_predictions, all_labels)

# ============================================
# IMPROVED TRAINING LOOP
# ============================================
print("Starting improved training loop...\n")

# Track metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
val_f1_scores = []

best_val_f1 = 0.0
patience_counter = 0
best_model_state = None

start_time = time.time()

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 40)

    # Training phase
    train_loss, train_acc = train_epoch_improved(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation phase
    val_loss, val_acc, val_precision, val_recall, val_f1, val_probs, val_preds, val_labels = validate_epoch_improved(
        model, test_loader, criterion, device
    )
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    val_f1_scores.append(val_f1)

    # Update learning rate
    scheduler.step()

    # Print epoch results
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    print(f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")
    print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")

    # Early stopping based on F1 score
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        best_model_state = model.state_dict().copy()

        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': best_val_f1,
            'val_accuracy': val_acc,
        }, MODEL_SAVE_PATH)

        print(f"✅ New best model saved! (F1: {val_f1:.4f}, Acc: {val_acc:.4f})")
    else:
        patience_counter += 1
        print(f"⏳ Early stopping counter: {patience_counter}/{PATIENCE}")

    # Check early stopping
    if patience_counter >= PATIENCE:
        print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
        break

training_time = time.time() - start_time
print(f"\n✅ Training completed in {training_time:.2f} seconds")

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("✅ Best model loaded for evaluation")

# ============================================
# COMPREHENSIVE EVALUATION
# ============================================
print("\n📊 Step 9: Comprehensive Model Evaluation")
print("-" * 70)

# Evaluate on test set
model.eval()
test_predictions = []
test_probabilities = []
test_labels = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)

        probabilities = torch.softmax(output, dim=1)
        _, predictions = torch.max(output, 1)

        test_predictions.extend(predictions.cpu().numpy())
        test_probabilities.extend(probabilities.cpu().numpy())
        test_labels.extend(target.cpu().numpy())

# Convert to numpy arrays
test_predictions = np.array(test_predictions)
test_probabilities = np.array(test_probabilities)
test_labels = np.array(test_labels)

# Get probabilities for class 1 (phishing)
test_phishing_probs = test_probabilities[:, 1]

# Calculate comprehensive metrics
test_accuracy = accuracy_score(test_labels, test_predictions)
test_precision = precision_score(test_labels, test_predictions, average='binary', zero_division=0)
test_recall = recall_score(test_labels, test_predictions, average='binary', zero_division=0)
test_f1 = f1_score(test_labels, test_predictions, average='binary', zero_division=0)

# Confusion matrix
cm = confusion_matrix(test_labels, test_predictions)
tn, fp, fn, tp = cm.ravel()

# Classification report
class_report = classification_report(test_labels, test_predictions,
                                    target_names=['Safe Email', 'Phishing Email'])

print(f"📈 Comprehensive Test Performance:")
print(f"   Accuracy:  {test_accuracy:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")
print(f"   F1-Score:  {test_f1:.4f}")

print(f"\n📊 Confusion Matrix:")
print(f"                   Predicted Safe   Predicted Phishing")
print(f"Actual Safe        {tn:>12}   {fp:>16}")
print(f"Actual Phishing    {fn:>12}   {tp:>16}")

print(f"\n📋 Classification Report:")
print(class_report)

# ============================================
# CONFIDENCE ANALYSIS
# ============================================
print("\n🎯 Step 10: Detailed Confidence Analysis")
print("-" * 70)

correct_predictions = (test_predictions == test_labels)
confidence_correct = test_phishing_probs[correct_predictions]
confidence_incorrect = test_phishing_probs[~correct_predictions]

# Confidence bins analysis
confidence_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
bin_accuracies = []

for i in range(len(confidence_bins)-1):
    low, high = confidence_bins[i], confidence_bins[i+1]
    mask = (test_phishing_probs >= low) & (test_phishing_probs < high)
    if mask.sum() > 0:
        bin_accuracy = (test_labels[mask] == test_predictions[mask]).mean()
        bin_accuracies.append(bin_accuracy)
        print(f"   Confidence {low:.1f}-{high:.1f}: {bin_accuracy:.4f} accuracy ({mask.sum()} samples)")

print(f"\n📊 Confidence Statistics:")
print(f"   Average confidence (correct): {np.mean(confidence_correct):.4f}")
print(f"   Average confidence (incorrect): {np.mean(confidence_incorrect):.4f}")
print(f"   Confidence difference: {np.mean(confidence_correct) - np.mean(confidence_incorrect):.4f}")

# ============================================
# CREATE VISUALIZATIONS
# ============================================
print("\n📈 Step 11: Creating Enhanced Visualizations")
print("-" * 70)

# Create comprehensive plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Loss plot
axes[0, 0].plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
axes[0, 0].plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy plot
axes[0, 1].plot(train_accuracies, label='Training Accuracy', color='blue', alpha=0.7)
axes[0, 1].plot(val_accuracies, label='Validation Accuracy', color='red', alpha=0.7)
axes[0, 1].set_title('Training and Validation Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# F1-Score plot
axes[0, 2].plot(val_f1_scores, label='Validation F1-Score', color='green', alpha=0.7)
axes[0, 2].set_title('Validation F1-Score')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('F1-Score')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# Confidence distribution
axes[1, 1].hist(confidence_correct, bins=30, alpha=0.7, label='Correct', color='green')
axes[1, 1].hist(confidence_incorrect, bins=30, alpha=0.7, label='Incorrect', color='red')
axes[1, 1].set_title('Confidence Distribution')
axes[1, 1].set_xlabel('Confidence Score')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

# Confidence vs Accuracy
axes[1, 2].bar(range(len(bin_accuracies)), bin_accuracies, color='purple', alpha=0.7)
axes[1, 2].set_title('Accuracy by Confidence Level')
axes[1, 2].set_xlabel('Confidence Bin')
axes[1, 2].set_ylabel('Accuracy')
axes[1, 2].set_xticks(range(len(bin_accuracies)))
axes[1, 2].set_xticklabels(['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}improved_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Enhanced visualizations saved: {OUTPUT_DIR}improved_training_history.png")

# ============================================
# SAVE COMPREHENSIVE METADATA
# ============================================
print("\n💾 Step 12: Saving Comprehensive Metadata")
print("-" * 70)

# Save model information
model_info = {
    'input_size': input_size,
    'architecture': [512, 256, 128, 64],
    'training_params': {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs_trained': len(train_losses),
        'class_weights': class_weights.cpu().numpy().tolist(),
        'training_time': training_time
    },
    'performance': {
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'confusion_matrix': cm.tolist(),
        'best_val_f1': best_val_f1
    },
    'feature_info': {
        'tfidf_features': len(feature_info['tfidf_features']),
        'engineered_features': len(feature_info['engineered_features']),
        'total_features': input_size
    }
}

# Save model info
model_info_path = f"{OUTPUT_DIR}improved_model_info.pkl"
joblib.dump(model_info, model_info_path)
print(f"✅ Model metadata saved: {model_info_path}")

# Save comprehensive report
report_path = f"{OUTPUT_DIR}improved_training_report.txt"
with open(report_path, 'w') as f:
    f.write("PHISHGUARD - IMPROVED MODEL TRAINING REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Training completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Training time: {training_time:.2f} seconds\n")
    f.write(f"Epochs trained: {len(train_losses)}\n\n")

    f.write("CLASS BALANCING:\n")
    f.write(f"  Class weights - Safe: {class_weights[0]:.4f}, Phishing: {class_weights[1]:.4f}\n")
    f.write(f"  Training sampler: WeightedRandomSampler\n\n")

    f.write("FINAL PERFORMANCE:\n")
    f.write(f"  Accuracy:  {test_accuracy:.4f}\n")
    f.write(f"  Precision: {test_precision:.4f}\n")
    f.write(f"  Recall:    {test_recall:.4f}\n")
    f.write(f"  F1-Score:  {test_f1:.4f}\n\n")

    f.write("CONFUSION MATRIX:\n")
    f.write(f"  True Negatives (Safe correctly identified):  {tn}\n")
    f.write(f"  False Positives (Safe misclassified as Phishing): {fp}\n")
    f.write(f"  False Negatives (Phishing misclassified as Safe): {fn}\n")
    f.write(f"  True Positives (Phishing correctly identified): {tp}\n\n")

    f.write("CLASSIFICATION REPORT:\n")
    f.write(class_report)

print(f"✅ Comprehensive report saved: {report_path}")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 70)
print("✅ IMPROVED MODEL TRAINING COMPLETE!")
print("=" * 70)

print(f"\n🎯 TRAINING SUMMARY:")
print(f"   • Model: Improved Neural Network with class balancing")
print(f"   • Input features: {input_size:,}")
print(f"   • Training time: {training_time:.2f} seconds")
print(f"   • Best validation F1: {best_val_f1:.4f}")
print(f"   • Final test accuracy: {test_accuracy:.4f}")

print(f"\n📊 PERFORMANCE METRICS:")
print(f"   • Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   • Precision: {test_precision:.4f}")
print(f"   • Recall:    {test_recall:.4f}")
print(f"   • F1-Score:  {test_f1:.4f}")

print(f"\n💾 SAVED FILES:")
print(f"   • Model: {MODEL_SAVE_PATH}")
print(f"   • Training history: {OUTPUT_DIR}improved_training_history.png")
print(f"   • Model metadata: {model_info_path}")
print(f"   • Training report: {report_path}")

print(f"\n🚀 NEXT STEPS:")
if test_accuracy >= 0.80:
    print(f"   ✅ Target achieved! Proceed to model evaluation and conversion")
    print(f"   1. Run: src/evaluate_model.py")
    print(f"   2. Run: src/convert_to_tfjs.py")
else:
    print(f"   ⚠️  Performance below target. Consider:")
    print(f"      - Feature engineering improvements")
    print(f"      - Hyperparameter tuning")
    print(f"      - Different model architectures")
    print(f"      - More training data")

print("=" * 70)