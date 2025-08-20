# 🛒 Basket Transformer
*A lightweight transformer model for predicting the next grocery product in a basket sequence.*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Optimized-green.svg)](https://developer.apple.com/metal/)

---

## 🎯 Project Overview

This project demonstrates how transformer architectures can be adapted for grocery basket sequence prediction. Using a custom GPT-style model with **Focal Loss** for handling class imbalance, we predict the next product a customer is likely to add to their basket.

**Key Features:**
- 🚀 **Apple Silicon Optimized** (MPS backend)
- 📊 **Focal Loss** for imbalanced data handling
- ⚖️ **Weighted Sampling** for head/tail balance
- ⚡ **Fast Training** (~70 minutes on M2 Pro)
- 🎯 **Real-world Performance** on 400K baskets

---

## 🧠 Model Architecture

### Core Configuration
```python
# Model specs (actual from code)
vocab_size     = 5_000      # Top-5K products from Dunnhumby
context_length = 30         # Sequence length
emb_dim        = 128        # Embedding dimension  
n_heads        = 8          # Multi-head attention
n_layers       = 3          # Transformer layers
drop_rate      = 0.05       # Minimal dropout
```

### Key Innovations

**1. Focal Loss Integration**
```python
# Handles class imbalance better than CrossEntropy
from focal_loss_custom_04 import FocalLoss
loss_fn = FocalLoss()  # Focus on hard examples
```

**2. Smart Weighted Sampling**
```python
# Down-weight frequent products (0, 1, 2)
head_ids = {0, 1, 2}  
weights = [0.03 if y in head_ids else 1.0 for y in train_ds.targets]
sampler = WeightedRandomSampler(weights, ...)
```

**3. Efficient Data Pipeline**
```python
# Fast parquet loading with proper target handling
df["target"] = df["target"].fillna(0).astype(int)
# Pin memory for faster GPU transfer
DataLoader(..., pin_memory=True, non_blocking=True)
```

---

## 📊 Training Process

### Data Preparation
The model uses the processed Dunnhumby dataset:
- **Input**: `data/shrink_bal/train.parquet` (400K samples)
- **Validation**: `data/shrink_bal/val.parquet` (50K samples)
- **Format**: Each row contains `seq` (product sequence) and `target` (next product)

### Training Loop
```python
# Last-token prediction (GPT style)
logits = model(x)[:, -1, :]  # Only predict next token
loss = loss_fn(logits, y)    # Focal loss for imbalance

# Standard optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Performance Monitoring
The training script tracks:
- ✅ **Loss curves** (train/val plotted automatically)
- ✅ **Memory usage** (optimized for M2 Pro)
- ✅ **Training speed** (~4,167 steps/epoch)
- ✅ **Model checkpointing** (saved to `checkpoints/`)

---

## 🚀 Quick Start

### 1. Prerequisites
```bash
pip install torch pandas pyarrow matplotlib tqdm focal-loss-torch
```

### 2. Data Setup
Ensure your data structure matches:
```
data/
└── shrink_bal/
    ├── train.parquet    # Training sequences
    └── val.parquet      # Validation sequences
```

### 3. Run Training
```bash
python train_basket_transformer.py
```

**Expected Output:**
```
Training on mps | 400000 samples | 4167 steps/epoch | 20835 total steps
Epoch 1/5: 100%|██████| 4167/4167 [14:32<00:00, loss=2.1234]
Epoch 1 | train 2.1234 | val 2.0987
...
Training done. Model saved to checkpoints/basket_gpt.pt
```

### 4. Results
- **Model**: Saved to `checkpoints/basket_gpt.pt`
- **Config**: Saved to `checkpoints/config.json`
- **Loss Plot**: Displayed automatically via matplotlib

---

## 📈 Expected Performance

### Training Metrics
Based on the actual implementation:

| Metric | Value |
|--------|--------|
| **Training Time** | ~70 minutes (M2 Pro) |
| **Memory Usage** | ~2-3GB peak |
| **Model Size** | ~1M parameters |
| **Steps/Epoch** | ~4,167 (400K/96 batch) |
| **Convergence** | 2-3 epochs typically |

### Health Indicators
```python
# What the code monitors:
✅ GOOD: Train loss ↓, Val loss follows then plateaus, No NaN/inf
⚠️  WARNING: Large train/val gap (>0.8), Loss oscillations
❌ BAD: NaN losses, Memory errors, No improvement after 3 epochs
```

---

## 🔧 Code Structure

### Main Components

**Dataset Class:**
```python
class BasketDataset(Dataset):
    # Loads parquet files efficiently
    # Handles missing targets (fillna(0))
    # Returns (sequence, target) pairs
```

**Training Function:**
```python
def train():
    # 1. Load datasets with weighted sampling
    # 2. Initialize GPT model + Focal Loss
    # 3. Training loop with last-token prediction
    # 4. Validation evaluation
    # 5. Model checkpointing
```

### Key Files
```
basket-transformer/
├── train_basket_transformer.py    # Main training script (actual code)
├── model_03.py                   # GPT model implementation
├── focal_loss_custom_04.py       # Focal loss for imbalance
├── data/shrink_bal/              # Processed data
└── checkpoints/                  # Model outputs
```

---

## ⚙️ Configuration Details

### Current Settings (from code)
```python
# Optimized for 2-hour M2 Pro training
batch_size = 96          # Larger batches = fewer steps
lr = 3e-4               # Higher LR for faster convergence  
num_epochs = 5          # Sufficient for convergence
eval_every = 400        # Infrequent evaluation for speed
patience = 2            # Quick early stopping
```

### Class Imbalance Handling
```python
# Smart weighting strategy
head_ids = {0, 1, 2}    # Most frequent products
weights = [0.03 if y in head_ids else 1.0 for y in targets]
# Result: 97% down-sampling of frequent items
```

---

## 📊 Data Pipeline

### Input Format
```python
# Expected parquet structure:
df.columns = ['seq', 'target']
# seq: List of product IDs [1, 45, 123, ...]  
# target: Next product ID (int)
```

### Data Loading
```python
# Efficient loading with proper types
df["target"] = df["target"].fillna(0).astype(int)
# Fast GPU transfer
DataLoader(..., pin_memory=True, non_blocking=True)
```

---

## 🎯 Model Outputs

### Training Artifacts
After training completes:
```
checkpoints/
├── basket_gpt.pt       # Model state dict
└── config.json         # Training configuration
```

### Loss Visualization
The script automatically generates:
- **Training loss curve** (blue line)
- **Validation loss curve** (orange line)  
- **Matplotlib display** showing convergence

---

## 🔍 Troubleshooting

### Common Issues

**Memory Errors:**
```python
# Reduce batch size if OOM
batch_size = 64  # instead of 96
```

**Slow Training:**
```python
# Increase batch size if memory allows
batch_size = 128  # faster convergence
```

**Loss Not Decreasing:**
```python
# Check data loading
print(f"Train samples: {len(train_ds)}")
print(f"Unique targets: {len(set(train_ds.targets))}")
```

### Expected Behavior
- **Epoch 1**: Loss should drop significantly (>50% reduction)
- **Epoch 2-3**: Gradual improvement with validation tracking train
- **Epoch 4-5**: Convergence with possible early stopping

---

## 📝 Next Steps

### Immediate Improvements
- [ ] Add **learning rate scheduling** (warmup + cosine decay)
- [ ] Implement **gradient clipping** for stability
- [ ] Add **top-k accuracy** metrics beyond loss
- [ ] Include **inference script** for basket predictions

### Advanced Features
- [ ] **Beam search** for multiple predictions
- [ ] **Attention visualization** for interpretability  
- [ ] **A/B testing framework** for model comparison
- [ ] **Production API** for real-time inference

---
