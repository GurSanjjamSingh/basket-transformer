import os, json, pandas as pd
from collections import Counter

train_path      = "data/splits/train.parquet"
val_path        = "data/splits/val.parquet"
id2product_path = "data/mappings/id2product.json"
out_dir         = "data/shrink_bal"
os.makedirs(out_dir, exist_ok=True)

# ---------- load mappings ----------
id2product = json.load(open(id2product_path))

# ---------- build top-5 k product set ----------
train_targets = pd.read_parquet(train_path)["target"].dropna().astype(int)
top5k = [t for t, _ in Counter(train_targets).most_common(5_000)]

new_id2product = {new_id: id2product[str(int(old_id))]
                  for new_id, old_id in enumerate(top5k)}
new_product2id = {v: k for k, v in new_id2product.items()}

# ---------- helpers ----------
def remap(seq):
    return [new_product2id.get(id2product.get(str(int(tok)), "UNK"), 0)
            for tok in seq]

def remap_target(tgt):
    return new_product2id.get(id2product.get(str(int(tgt)), "UNK"), 0)

frac_keep = 0.20   # keep 20% of head rows

# Target dataset sizes
target_train_size = 400_000
target_val_size = 50_000

# First pass: remap train targets to find the ACTUAL top-3 after remapping
train_df = pd.read_parquet(train_path)
train_df["target"] = train_df["target"].map(remap_target)

# Find the ACTUAL top-3 most frequent targets after remapping
actual_top3 = [t for t, _ in Counter(train_df["target"]).most_common(3)]
print(f"ACTUAL top-3 most frequent targets after remapping: {actual_top3}")
print(f"Corresponding products: {[new_id2product[tid] for tid in actual_top3]}")

head_ids = set(actual_top3)

for split, path in zip(["train", "val"], [train_path, val_path]):
    df = pd.read_parquet(path)
    target_size = target_train_size if split == "train" else target_val_size
    
    # 1. remap target first
    df["target"] = df["target"].map(remap_target)
    
    # 2. down-sample head using the ACTUAL top-3 IDs
    mask_head = df["target"].isin(head_ids)
    df_head   = df[mask_head].sample(frac=frac_keep, random_state=42)
    df_tail   = df[~mask_head]
    
    print(f"{split}: head rows before down-sampling: {len(df[mask_head])}")
    print(f"{split}: head rows after down-sampling: {len(df_head)}")
    print(f"{split}: tail rows before thinning: {len(df_tail)}")
    
    # 3. thin the tail to meet target dataset size
    remaining_budget = target_size - len(df_head)
    if len(df_tail) > remaining_budget:
        df_tail = df_tail.sample(n=remaining_budget, random_state=42)
        print(f"{split}: tail rows after thinning: {len(df_tail)} (randomly sampled)")
    else:
        print(f"{split}: tail rows after thinning: {len(df_tail)} (kept all)")
    
    # 4. remap sequences on remaining rows
    df_bal = pd.concat([df_head, df_tail])
    df_bal["seq"] = df_bal["seq"].apply(lambda s: [int(x) for x in remap(s)])
    df_bal = df_bal.reset_index(drop=True)
    
    # Verify the final distribution and size
    final_counts = Counter(df_bal["target"])
    print(f"{split}: final dataset size: {len(df_bal)} (target: {target_size})")
    print(f"{split}: final target distribution (top-10): {final_counts.most_common(10)}")
    
    out_path = os.path.join(out_dir, f"{split}.parquet")
    df_bal.to_parquet(out_path, index=False)
    print(f"{split} balanced: {len(df_bal)} rows → {out_path}")
    print()

# ---------- save mappings ----------
json.dump(new_id2product, open(os.path.join(out_dir, "id2product.json"), "w"), indent=2)
print("New vocab size:", len(new_id2product))

