import torch, json, os, random
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import GPTModel

# ---------- hyper-parameters ----------
CFG = dict(
    vocab_size     = 92_341,
    context_length = 50,
    emb_dim        = 128,      
    n_heads        = 4,
    n_layers       = 4,
    drop_rate      = 0.1,
    qkv_bias       = False,
    batch_size     = 32,       
    lr             = 3e-4,
    num_epochs     = 1,       
    device         = 'cuda' if torch.cuda.is_available() else 'cpu'
)
# --------------------------------------

train_losses, val_losses = [], []

class BasketDataset(Dataset):
    def __init__(self, parquet_path):
        df = pd.read_parquet(parquet_path).sample(frac=0.05, random_state=42)
        self.seqs    = df["seq"].tolist()
        self.targets = df["target"].tolist()
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.long), \
               torch.tensor(self.targets[idx], dtype=torch.long)

def train():
    train_ds = BasketDataset("/Users/gursanjjam/Documents/basket-transformer/notebooks/data/splits/train.parquet")
    val_ds   = BasketDataset("/Users/gursanjjam/Documents/basket-transformer/notebooks/data/splits/val.parquet")
    train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"],
                          shuffle=True, drop_last=True, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                          shuffle=False, pin_memory=True)

    model = GPTModel(CFG).to(CFG["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
    loss_fn   = torch.nn.CrossEntropyLoss()

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
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(CFG["device"]), y.to(CFG["device"])
                logits = model(x)[:, -1, :]
                val_loss += loss_fn(logits, y).item()
        avg_val = val_loss / len(val_dl)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
    
        print(f"Epoch {epoch+1} | train {avg_train:.4f} | val {avg_val:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/basket_gpt.pt")
    json.dump(CFG, open("checkpoints/config.json", "w"), indent=2)
    print("Training done. Model saved to checkpoints/basket_gpt.pt")

if __name__ == "__main__":
    train()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('epoch'); plt.ylabel('CE loss'); plt.legend(); plt.show()