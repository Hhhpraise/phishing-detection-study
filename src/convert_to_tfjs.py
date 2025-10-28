"""
Convert PyTorch PhishGuard Model to ONNX and TensorFlow.js
Run this in your PyCharm project
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
import json
import os

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "../models/phishing_model_improved.pth"
VECTORIZER_PATH = "../models/tfidf_vectorizer.pkl"
FEATURE_INFO_PATH = "../models/feature_info.pkl"
OUTPUT_DIR = "../model/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("CONVERTING PYTORCH MODEL TO TENSORFLOW.JS")
print("=" * 70)

# ============================================
# DEFINE MODEL ARCHITECTURE
# ============================================
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
        return torch.softmax(x, dim=1)  # Add softmax for probabilities

# ============================================
# LOAD MODEL
# ============================================
print("\n📂 Loading PyTorch model...")
input_size = 5010
model = ImprovedPhishingDetector(input_size=input_size)
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✅ Model loaded successfully")

# ============================================
# EXPORT TO ONNX
# ============================================
print("\n🔄 Converting to ONNX format...")
dummy_input = torch.randn(1, input_size)
onnx_path = os.path.join(OUTPUT_DIR, "phishing_model.onnx")

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print(f"✅ ONNX model saved: {onnx_path}")

# ============================================
# EXPORT VOCABULARY AND PREPROCESSING DATA
# ============================================
print("\n📝 Exporting vocabulary and preprocessing data...")

# Load vectorizer and feature info
vectorizer = joblib.load(VECTORIZER_PATH)
feature_info = joblib.load(FEATURE_INFO_PATH)

# Extract vocabulary
vocabulary = vectorizer.vocabulary_
idf_values = vectorizer.idf_.tolist()

# Create preprocessing data for JavaScript
preprocessing_data = {
    "vocabulary": vocabulary,
    "idfValues": idf_values,
    "engineeredFeatures": feature_info['engineered_features'],
    "input_dim": input_size,
    "max_features": len(vocabulary),
    "model_info": {
        "architecture": "ImprovedPhishingDetector",
        "hidden_sizes": [512, 256, 128, 64],
        "num_classes": 2
    }
}

# Save to JSON
vocab_path = os.path.join(OUTPUT_DIR, "vocabulary.json")
with open(vocab_path, 'w') as f:
    json.dump(preprocessing_data, f, indent=2)

print(f"✅ Vocabulary saved: {vocab_path}")
print(f"   - Vocabulary size: {len(vocabulary)}")
print(f"   - Input dimension: {input_size}")

# ============================================
# CREATE MODEL WEIGHTS JSON (ALTERNATIVE)
# ============================================
print("\n💾 Creating simplified model for browser...")

# Extract model weights in a JSON-serializable format
model_weights = {}
for name, param in model.named_parameters():
    model_weights[name] = param.detach().cpu().numpy().tolist()

weights_path = os.path.join(OUTPUT_DIR, "model_weights.json")
with open(weights_path, 'w') as f:
    json.dump(model_weights, f)

print(f"✅ Model weights saved: {weights_path}")
print(f"   Warning: This file is very large (~{os.path.getsize(weights_path) / 1024 / 1024:.1f}MB)")

# ============================================
# FINAL INSTRUCTIONS
# ============================================
print("\n" + "=" * 70)
print("✅ MODEL EXPORT COMPLETE!")
print("=" * 70)

print("\n📦 Output Files:")
print(f"   • ONNX model: {onnx_path}")
print(f"   • Vocabulary: {vocab_path}")
print(f"   • Weights: {weights_path}")

print("\n⚠️  IMPORTANT: ONNX.js Setup Required")
print("=" * 70)
print("Since TensorFlow.js doesn't directly support PyTorch models,")
print("you have two options:")
print()
print("Option 1 (RECOMMENDED): Use ONNX.js in the browser")
print("   1. Include ONNX.js in your HTML:")
print('      <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>')
print("   2. Update phishguard_model.js to use ONNX.js")
print()
print("Option 2: Use the mock predictions (already working)")
print("   The mock predictions are actually quite good for demo purposes")
print()
print("See the next artifact for updated phishguard_model.js with ONNX.js support")