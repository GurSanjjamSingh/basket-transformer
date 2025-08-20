import torch, json, pandas as pd
from model_03 import GPTModel
from collections import defaultdict

def load_model_and_mappings():
    """Load the model and create mappings"""
    CFG   = json.load(open("checkpoints/config.json"))
    state = torch.load("checkpoints/basket_gpt.pt", map_location="cpu")

    model = GPTModel(CFG)
    model.load_state_dict(state)
    model.eval()

    id2product = json.load(open("data/mappings/id2product.json"))
    prod_df    = pd.read_csv("/Users/gursanjjam/Documents/basket-transformer/dunnhumby/product.csv")

    token2name = {
        int(k): prod_df.loc[prod_df.PRODUCT_ID == int(v), "COMMODITY_DESC"].iloc[0]
        for k, v in id2product.items()
    }
    
    return model, token2name

def group_items_by_category(token2name):
    """Group items by category and create ranges"""
    categories = defaultdict(list)
    
    # Group by category name
    for token_id, name in token2name.items():
        categories[name].append(token_id)
    
    # Sort IDs within each category and create display info
    category_info = {}
    for category, ids in categories.items():
        ids.sort()
        category_info[category] = {
            'ids': ids,
            'range': f"({min(ids)}-{max(ids)})",
            'count': len(ids)
        }
    
    return category_info

def display_categories_menu(category_info):
    """Display categories in a compact format"""
    print("\n" + "="*100)
    print("🛒 PRODUCT CATEGORIES - Choose items by entering their ID numbers")
    print("="*100)
    
    # Sort categories by name for consistent display
    sorted_categories = sorted(category_info.items())
    
    # Display categories with ranges and sample IDs
    for i, (category, info) in enumerate(sorted_categories):
        # Show category with range and a few sample IDs
        sample_ids = info['ids'][:5]  # Show first 5 IDs as examples
        sample_str = ', '.join(map(str, sample_ids))
        if len(info['ids']) > 5:
            sample_str += f", ... ({info['count']} total)"
        
        # Color coding for different category types
        if 'VEGETABLE' in category.upper():
            icon = "🥕"
        elif 'MEAT' in category.upper() or 'BEEF' in category.upper() or 'POULTRY' in category.upper():
            icon = "🥩"
        elif 'DAIRY' in category.upper() or 'MILK' in category.upper() or 'CHEESE' in category.upper():
            icon = "🧀"
        elif 'BREAD' in category.upper() or 'BAKERY' in category.upper():
            icon = "🍞"
        elif 'FRUIT' in category.upper():
            icon = "🍎"
        elif 'BEVERAGE' in category.upper() or 'DRINK' in category.upper():
            icon = "🥤"
        else:
            icon = "📦"
        
        print(f"{icon} {category:<35} {info['range']:<15} Examples: {sample_str}")
    
    print("="*100)
    print("💡 TIP: Enter any ID number from the ranges above to select that item")

def get_user_basket(token2name, category_info):
    """Get 3 items from user with category context"""
    display_categories_menu(category_info)
    
    selected_items = []
    selected_tokens = []
    valid_ids = set(token2name.keys())
    
    print(f"\nSelect 3 items by entering their ID numbers:")
    
    for i in range(3):
        while True:
            try:
                user_input = input(f"\nItem {i+1}/3 ID: ").strip()
                item_id = int(user_input)
                
                if item_id in valid_ids:
                    if item_id not in selected_tokens:
                        selected_tokens.append(item_id)
                        item_name = token2name[item_id]
                        selected_items.append(item_name)
                        
                        # Show category context
                        category_found = None
                        for cat, info in category_info.items():
                            if item_id in info['ids']:
                                category_found = cat
                                break
                        
                        print(f"✓ You chose: {item_name}")
                        if category_found:
                            print(f"  📂 Category: {category_found}")
                        break
                    else:
                        print("❌ Already selected! Choose a different item.")
                else:
                    print(f"❌ Invalid ID! Please enter a valid ID from the ranges shown above.")
                    print("💡 Hint: Look at the range numbers like (297-74877) and pick any number in between")
                    
            except ValueError:
                print("❌ Please enter a valid number.")
    
    return selected_items, selected_tokens

def predict_next_item(model, token2name, bought_tokens):
    """Predict next item using nucleus sampling"""
    temp = 1.2
    top_p = 0.9
    
    with torch.no_grad():
        logits = model(torch.tensor([bought_tokens]))[:, -1, :]
        probs = torch.softmax(logits / temp, dim=-1).squeeze()

        # Get top 8 for display
        top_probs, top_indices = torch.topk(probs, 8)
        
        # Nucleus sampling
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        sorted_probs[(cumsum - sorted_probs) > top_p] = 0.0
        sorted_probs /= sorted_probs.sum()
        
        next_token = sorted_idx[torch.multinomial(sorted_probs, 1)].item()
        
    return next_token, top_indices.tolist(), top_probs.tolist()

def main():
    print("🛒 SMART BASKET PREDICTION SYSTEM")
    print("Loading model and organizing products...")
    
    model, token2name = load_model_and_mappings()
    category_info = group_items_by_category(token2name)
    
    print(f"✅ Ready! Organized {len(token2name)} products into {len(category_info)} categories")
    
    while True:
        # Get user basket
        selected_items, selected_tokens = get_user_basket(token2name, category_info)
        
        # Show selected basket with categories
        print("\n" + "="*60)
        print("🛍️  YOUR BASKET:")
        for i, (item, token) in enumerate(zip(selected_items, selected_tokens), 1):
            # Find category for context
            category = None
            for cat, info in category_info.items():
                if token in info['ids']:
                    category = cat
                    break
            print(f"   {i}. {item}")
            if category:
                print(f"      └── {category}")
        
        # Make prediction
        print(f"\n🔮 PREDICTING NEXT ITEM...")
        predicted_token, top_tokens, top_probs = predict_next_item(model, token2name, selected_tokens)
        
        predicted_item = token2name[predicted_token]
        print(f"\n🎯 MAIN PREDICTION: {predicted_item}")
        
        # Find category for prediction
        pred_category = None
        for cat, info in category_info.items():
            if predicted_token in info['ids']:
                pred_category = cat
                break
        if pred_category:
            print(f"   📂 Category: {pred_category}")
        
        print(f"\n📊 TOP SUGGESTIONS:")
        for i, (token, prob) in enumerate(zip(top_tokens, top_probs), 1):
            item_name = token2name[token]
            # Find category
            item_category = None
            for cat, info in category_info.items():
                if token in info['ids']:
                    item_category = cat
                    break
            
            print(f"   {i}. {item_name:<35} ({prob:.1%})")
            if item_category:
                print(f"      └── {item_category}")
        
        # Continue?
        print("\n" + "="*60)
        again = input("Predict again? (y/n): ").strip().lower()
        if again not in ['y', 'yes']:
            break
    
    print("Thanks for using Smart Basket Prediction! 🛒✨")

if __name__ == "__main__":
    main()


'''
📋 Category Organization:

Groups identical items (like "VEGETABLES - SHELF STABLE") into ranges
Shows format like: 🥕 VEGETABLES - SHELF STABLE (297-64830) Examples: 297, 298, 311, 488, 505, ... (500+ total)

🎨 Visual Categories:

🥕 Vegetables
🥩 Meat/Poultry/Beef
🧀 Dairy/Milk/Cheese
🍞 Bread/Bakery
🍎 Fruits
🥤 Beverages
📦 Other items

Smart Display:

Shows category ranges instead of hundreds of duplicate lines
Provides sample IDs and total count for each category
Users can pick ANY number within the range
Shows category context when items are selected

Predictions:

Shows 8 top suggestions 
Displays category for each predicted item
Clear hierarchy with item → category structure

Example Flow:
🥕 VEGETABLES - SHELF STABLE    (297-64830)     Examples: 297, 298, 311, 488, 505, ... (500+ total)
🥩 MEAT - FRESH                 (1245-8934)     Examples: 1245, 1247, 1250, ... (45 total)

Item 1/3 ID: 500
✓ You chose: VEGETABLES - SHELF STABLE
  📂 Category: VEGETABLES - SHELF STABLE

🎯 MAIN PREDICTION: PASTA
   📂 Category: PASTA - DRY GOODS'''
