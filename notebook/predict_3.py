import torch, json, pandas as pd
from model import GPTModel

CFG   = json.load(open("checkpoints/config.json"))
state = torch.load("checkpoints/basket_gpt.pt", map_location="cpu")
model = GPTModel(CFG); model.load_state_dict(state); model.eval()

id2product = json.load(open("/Users/gursanjjam/Documents/basket-transformer/notebooks/data/mappings/id2product.json"))
prod_df    = pd.read_csv("/Users/gursanjjam/Documents/basket-transformer/dunnhumby/product.csv")   # original file

token2name = {int(k): prod_df.loc[prod_df.PRODUCT_ID == int(v), "COMMODITY_DESC"].iloc[0]
              for k, v in id2product.items()}

# demo
bought_tokens = [6533, 7270, 8279]
next_tokens   = torch.softmax(model(torch.tensor([bought_tokens]))[:, -1, :], dim=-1).topk(3).indices.squeeze().tolist()

print("bought      :", [token2name[t] for t in bought_tokens])
print("next 3 items:", [token2name[t] for t in next_tokens])