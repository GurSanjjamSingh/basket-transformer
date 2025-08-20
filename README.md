# 🛒 Basket Transformer
*A lightweight transformer model for predicting the next grocery product in a basket sequence.*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Optimized-green.svg)](https://developer.apple.com/metal/)

---

## 🎯 Project Overview

This project demonstrates how transformer architectures can be adapted for grocery basket sequence prediction. Using a custom GPT-style model, we predict the next product a customer is likely to add to their basket based on their current items.

**Key Features:**
- 🚀 **Optimized for Apple Silicon** (M1/M2/M3 Macs)
- 📊 **Balanced training data** with smart down-sampling
- ⚡ **Fast training** (~2 hours on M2 Pro)
- 🎯 **Real-world performance** with interpretable results

---

## 🧠 Technical Architecture

### Data Pipeline
The project includes a sophisticated data preprocessing pipeline:

1. **Data Shrinking**: Reduces vocabulary from 92K → 5K most frequent products
2. **Balanced Sampling**: Down-samples popular items (20% kept) while preserving long-tail diversity
3. **Smart Mapping**: Remaps product IDs to create efficient vocabulary
4. **Target Sizing**: Produces ~400K training / ~50K validation samples

### Model Configuration

**Smoke Test (15 minutes):**
```python
emb_dim=128, n_heads=8, n_layers=3, batch_size=128
context_length=25, vocab_size=5_000
```

**Full Training (~2 hours on M2 Pro):**
```python
emb_dim=128, n_heads=8, n_layers=3, batch_size=96
context_length=30, vocab_size=5_000, epochs=5
```

**Advanced Features:**
- ✅ **Learning Rate Warmup** with cosine decay
- ✅ **Gradient Clipping** (norm=1.0) 
- ✅ **AdamW Optimizer** with proper weight decay
- ✅ **Layer Normalization** (pre-norm architecture)
- ✅ **Early Stopping** with validation monitoring
- ✅ **Smart Checkpointing** for long training runs

### Training Process

The model uses next-token prediction with cross-entropy loss:

1. **Sequence Processing**: Baskets → product ID sequences (max length 30)
2. **Causal Masking**: Prevents future token leakage
3. **Loss Calculation**: Cross-entropy on vocabulary predictions
4. **Optimization**: AdamW with lr warmup + cosine decay

---

## 📊 Results & Performance

### Training Metrics
- **Training Time**: ~70 minutes (M2 Pro)
- **Memory Usage**: ~2-3GB peak
- **Convergence**: Stable loss decrease within 2-3 epochs
- **Model Size**: ~1M parameters

### Prediction Quality
The model demonstrates strong sequential learning:

**Example Prediction:**
```
Current Basket: ['ORGANIC BANANAS', 'GREEK YOGURT', 'WHOLE MILK']
Top-3 Predictions: ['BREAD', 'EGGS', 'CHEESE']
```

**Performance Indicators:**
- ✅ Coherent product relationships (dairy → bread/eggs)
- ✅ No repetition of current basket items
- ✅ Reasonable category associations
- ✅ Stable training without overfitting

---

## 🚀 Quick Start

### Prerequisites
```bash
# Required packages
pip install torch pandas pyarrow tqdm matplotlib
```

### 1. Data Preparation
```bash
# Download Dunnhumby dataset to data/ directory
python data_preprocessing.py    # Creates splits
python shrink_bal.py           # Balances & shrinks dataset
```

### 2. Model Training
```bash
# Smoke test (15 min)
python train.py --config smoke_test_cfg.py

# Full training (2 hours)  
python train.py --config full_cfg.py
```

### 3. Inference
```bash
python predict.py --model checkpoints/best_model.pt
```

---

## 📚 Dataset

**Source**: [Dunnhumby - The Complete Journey](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey)

**Original Scale:**
- 2M+ baskets from 2,500 households
- 92,341 unique products
- 2 years of transaction history

**Processed Scale:**
- 400K training baskets
- 50K validation baskets  
- 5K most frequent products
- Balanced head/tail distribution

### Key Files
```
data/
├── raw/
│   ├── product.csv              # Product catalog
│   └── transaction_data.csv     # Raw transactions
├── splits/
│   ├── train.parquet           # Training sequences
│   └── val.parquet             # Validation sequences
└── shrink_bal/
    ├── train.parquet           # Balanced training data
    ├── val.parquet             # Balanced validation data
    └── id2product.json         # Product mapping
```

---

## 🛠️ Project Structure

```
basket-transformer/
├── data_preprocessing.py       # Raw data → sequences
├── shrink_bal.py              # Vocabulary shrinking & balancing
├── smoke_test_cfg.py          # Quick test configuration
├── full_cfg.py                # Full training configuration  
├── model.py                   # Transformer implementation
├── train.py                   # Training loop
├── predict.py                 # Inference & examples
├── utils.py                   # Helper functions
└── README.md                  # This file
```

---

## 💡 Key Innovations

### 1. **Smart Data Balancing**
Instead of naive sampling, we:
- Identify actual top-3 frequent items after vocabulary remapping
- Down-sample only these items (20% kept)
- Preserve all long-tail diversity
- Apply global caps to meet target dataset sizes

### 2. **Apple Silicon Optimization**
- MPS backend compatibility (no mixed precision)
- Memory-efficient batch sizes
- Conservative model sizing for 2-hour training constraint
- Proper gradient clipping for MPS stability

### 3. **Production-Ready Training**
- Learning rate warmup prevents early instability
- Cosine decay maintains performance through training
- Early stopping prevents overfitting
- Smart checkpointing for recovery

---

## 🎯 Future Improvements

- [ ] **Larger Models**: Scale to 6-12 layers with more GPU/memory
- [ ] **Advanced Architectures**: Try RoPE, Flash Attention, or Mamba
- [ ] **Better Features**: Include price, category, seasonality data
- [ ] **Evaluation Metrics**: Implement BLEU, diversity metrics
- [ ] **Production Deployment**: Add inference API and model serving

---

