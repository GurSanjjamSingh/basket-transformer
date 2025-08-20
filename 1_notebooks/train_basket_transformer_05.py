# train_basket_transformer.py

import os, json, random
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from model_03 import GPTModel
from focal_loss_custom_04 import FocalLoss   # pip install focal-loss-torch

# === smoke_test_cfg.py ===
import torch

'''CFG = dict(
    # Model architecture
    vocab_size     = 5_000,      # from shrink_bal.py
    context_length = 25,
    emb_dim        = 128,        # must be divisible by n_heads
    n_heads        = 8,          # 128/8 = 16 dim per head (good ratio)
    n_layers       = 3,          
    drop_rate      = 0.1,        
    qkv_bias       = False,
    
    # Training hyperparameters
    batch_size     = 128,        
    lr             = 3e-4,       # base learning rate
    num_epochs     = 5,          
    device         = 'mps',
    
    # CRITICAL MISSING FEATURES ADDED:
    # 1. Learning rate scheduling (essential for transformers)
    use_lr_warmup  = True,
    warmup_steps   = 100,        # warmup over first 100 steps
    lr_decay       = 'cosine',   # 'cosine', 'linear', or 'constant'
    min_lr_ratio   = 0.1,        # minimum lr as ratio of base lr
    
    # 2. Gradient clipping (prevents exploding gradients)
    grad_clip_norm = 1.0,        # clip gradients to max norm of 1.0
    
    # 3. Better optimizer settings
    optimizer      = 'adamw',    # AdamW is better than Adam for transformers
    weight_decay   = 0.01,       # L2 regularization (exclude bias/layernorm)
    betas          = (0.9, 0.95), # momentum parameters
    eps            = 1e-8,       # numerical stability
    
    # 4. Advanced regularization
    use_dropout    = True,       # dropout in attention/ffn
    use_layer_norm = True,       # layer normalization
    norm_first     = True,       # pre-norm (more stable than post-norm)
    
    # 5. Training stability
    mixed_precision = False,     # set True for GPU training
    compile_model  = False,      # torch.compile for speed (set True on GPU)
    
    # 6. Monitoring and checkpointing
    eval_every     = 50,         
    log_every      = 25,         
    save_model     = False,      # smoke test doesn't need saves
    early_stop     = True,       
    patience       = 3,          
    monitor_metric = 'val_loss', # what to monitor for early stopping
    
    # 7. Data loading
    num_workers    = 2,          # parallel data loading
    pin_memory     = True,       # faster GPU transfer
    
    # 8. Reproducibility
    seed           = 42,
    deterministic  = True,       # for reproducible results
    
    # 9. Loss function
    label_smoothing = 0.0,       # can help with overconfident predictions
    ignore_index   = -100,       # standard for padding tokens
    
    # 10. Model initialization
    init_std       = 0.02,       # standard deviation for weight init
)'''

# What to look for in smoke test:
# ✅ GOOD SIGNS:
# - Train loss decreases smoothly
# - Val loss decreases initially, then plateaus/slightly increases
# - Gap between train/val loss stays reasonable (< 0.5)
# - Model converges within 2-3 epochs
# - No NaN/inf losses
# - Memory usage stable

# ⚠️  WARNING SIGNS:
# - Train loss stagnates (underfitting)
# - Val loss much higher than train (overfitting) 
# - Both losses oscillate wildly (lr too high)
# - Loss explodes to NaN (gradient explosion)
# - Very slow convergence (lr too low or model too small)

# ---------- full hyper-parameters (commented) ----------
# EVEN FASTER 2-HOUR CONFIG:
CFG = dict(
    # Aggressive size reduction
    vocab_size     = 5_000,
    context_length = 30,         # shorter sequences
    emb_dim        = 128,        # smaller model (128/8 = 16 per head)
    n_heads        = 8,          
    n_layers       = 3,          # very shallow
    drop_rate      = 0.05,       # minimal dropout
    qkv_bias       = False,
    
    # Aggressive training settings
    batch_size     = 96,         # larger batches for fewer steps
    lr             = 3e-4,       # higher LR for faster convergence
    num_epochs     = 5,          # fewer epochs
    device         = 'mps',
    
    # Minimal scheduling
    use_lr_warmup  = True,
    warmup_steps   = 100,        # very short warmup
    lr_decay       = 'cosine',
    min_lr_ratio   = 0.2,
    
    # Standard settings
    grad_clip_norm = 1.0,
    optimizer      = 'adamw',
    weight_decay   = 0.005,      # very light
    betas          = (0.9, 0.95),
    eps            = 1e-8,
    
    # Minimal regularization  
    use_dropout    = True,
    use_layer_norm = True,
    norm_first     = True,
    
    # Speed optimizations
    mixed_precision = False,
    compile_model  = False,
    
    # Minimal monitoring
    eval_every     = 400,        # infrequent eval
    log_every      = 150,
    save_model     = True,
    save_every     = 3000,
    early_stop     = True,
    patience       = 2,          # very quick early stop
    monitor_metric = 'val_loss',
    
    # Fast data loading
    num_workers    = 2,
    pin_memory     = False,
    
    seed           = 42,
    deterministic  = False,
    label_smoothing = 0.0,
    ignore_index   = -100,
    init_std       = 0.02,
)

# FAST CONFIG CALCULATIONS:
# Model: 128 emb × 3 layers = ~1M parameters
# Memory: ~2-3GB peak
# Steps per epoch: 400k / 96 = ~4,167 steps  
# Time per step: ~0.2s (smaller model)
# Time per epoch: 4,167 × 0.2s = ~14 minutes
# Total for 5 epochs: ~70 minutes = 1.2 hours ✅


# HEALTH CHECKS - These should be verified:
assert CFG['emb_dim'] % CFG['n_heads'] == 0, "emb_dim must be divisible by n_heads"
assert CFG['lr'] > 0, "Learning rate must be positive"
assert 0 <= CFG['drop_rate'] <= 1, "Dropout rate must be between 0 and 1"
assert CFG['warmup_steps'] > 0, "Warmup steps must be positive"

train_losses, val_losses = [], [] 

# ---------- Dataset ----------
class BasketDataset(Dataset):
    def __init__(self, parquet_path, sample_frac=0.05):
        df = pd.read_parquet(parquet_path)          #.sample(frac=sample_frac, random_state=42)
        df = df.copy()  # avoid chained assignment warnings
        df["target"] = df["target"].fillna(0).astype(int)
        self.seqs    = df["seq"].tolist()
        self.targets = df["target"].tolist()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.long), \
               torch.tensor(self.targets[idx], dtype=torch.long)

# ---------- Training ----------
def train():
    train_ds = BasketDataset("data/shrink_bal/train.parquet")
    val_ds   = BasketDataset("data/shrink_bal/val.parquet")

    # ---------------- Weighted sampler for head classes ----------------
    head_ids = {0, 1, 2}  # the most frequent products
    weights = [0.03 if y in head_ids else 1.0 for y in train_ds.targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"],
                          sampler=sampler, drop_last=True, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=CFG["batch_size"],
                          shuffle=False, pin_memory=True)

    model = GPTModel(CFG).to(CFG["device"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    loss_fn = FocalLoss()  # works with softmax logits for multiclass

    steps_per_epoch = len(train_dl)
    total_steps     = steps_per_epoch * CFG["num_epochs"]
    print(f"Training on {CFG['device']} | "
          f"{len(train_ds)} samples | {steps_per_epoch} steps/epoch | "
          f"{total_steps} total steps")

    for epoch in range(CFG["num_epochs"]):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{CFG['num_epochs']}")
        for x, y in pbar:
            x, y = x.to(CFG["device"], non_blocking=True), \
                   y.to(CFG["device"], non_blocking=True)
            logits = model(x)[:, -1, :]
            loss   = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = running_loss / len(train_dl)

        # fast validation (only last token)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(CFG["device"]), y.to(CFG["device"])
                logits = model(x)[:, -1, :]
                val_loss += loss_fn(logits, y).item()
        avg_val = val_loss / len(val_dl)

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        print(f"Epoch {epoch+1} | train {avg_train:.4f} | val {avg_val:.4f}")

    # Save model and config
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/basket_gpt.pt")
    json.dump(CFG, open("checkpoints/config.json", "w"), indent=2)
    print("Training done. Model saved to checkpoints/basket_gpt.pt")

# SMOKE TEST SUCCESS CRITERIA:
SUCCESS_CRITERIA = dict(
    # Loss should decrease
    train_loss_decrease = True,   # train loss should go down
    val_loss_reasonable = True,   # val loss shouldn't explode
    
    # No numerical issues
    no_nan_losses      = True,    # no NaN/inf in losses
    gradients_finite   = True,    # gradients should be finite
    
    # Convergence speed
    converge_epochs    = 3,       # should see improvement within 3 epochs
    max_train_val_gap  = 0.8,     # train/val gap shouldn't be huge
    
    # Memory/speed
    memory_stable      = True,    # no memory leaks
    reasonable_speed   = True,    # not too slow
)

# What to watch for:
print("🔍 SMOKE TEST MONITORING:")
print("✅ GOOD: Train loss ↓, Val loss follows then plateaus, No NaN/inf")
print("⚠️  WARNING: Large train/val gap (>0.8), Loss oscillations, Slow convergence")
print("❌ BAD: NaN losses, Gradient explosion, Memory errors, No improvement after 3 epochs")


# ---------- Run ----------
if __name__ == "__main__":
    train()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.show()