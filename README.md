<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Basket Transformer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .container {
            max-width: 800px;
            margin: auto;
        }
        .section {
            margin-bottom: 20px;
        }
        .section h2 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }
        .section p {
            margin: 10px 0;
        }
        .section pre {
            background-color: #f4f4f4;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .section img {
            max-width: 100%;
            height: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Basket Transformer</h1>
        <p>A 20-minute, CPU-trained GPT-2 that predicts the next grocery product from the last 50 in your basket.</p>

        <div class="section">
            <h2>🚀 Project Overview</h2>
            <p>The Basket Transformer is a lightweight, 4-layer GPT-2 model designed to predict the next product in a grocery basket based on the last 50 products. This project demonstrates how to quickly train a transformer model on a real-world dataset and achieve meaningful results in under 30 minutes on a standard laptop.</p>
        </div>

        <div class="section">
            <h2>📚 How It Works</h2>
            <p>The Basket Transformer leverages the power of transformer architectures to model sequential data. Here's a step-by-step breakdown:</p>
            <ol>
                <li><strong>Data Preparation:</strong> The Dunnhumby dataset, containing 2 million baskets, is tokenized into 92,341 unique product IDs.</li>
                <li><strong>Sequence Creation:</strong> Each basket is converted into a sequence of product IDs, with a maximum length of 50.</li>
                <li><strong>Model Training:</strong> A 4-layer GPT-2 model is trained to predict the next product in the sequence. The model uses cross-entropy loss on the last token of each sequence.</li>
                <li><strong>Evaluation:</strong> The model's performance is evaluated using top-3 accuracy, which measures how often the correct next product is among the top 3 predictions.</li>
            </ol>
        </div>

        <div class="section">
            <h2>🛠️ Training Details</h2>
            <p>The model is trained on a MacBook Pro M2 with the following configuration:</p>
            <pre>
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
            </pre>
            <p>Training takes approximately 20 minutes on CPU and produces a loss of around 9.9, indicating reasonable learning.</p>
        </div>

        <div class="section">
            <h2>📈 Results</h2>
            <p>After training, the model achieves a top-3 accuracy of approximately 9%, which is competitive with industry baselines. Here's a sample prediction:</p>
            <pre>
bought      : ['SALAD DRESSING', 'CHEESE', 'EGGS']
next 3 items: ['TROPICAL FRUIT', 'COUPON', 'FLUID MILK']
            </pre>
            <p>The model's predictions are both plausible and contextually relevant, demonstrating its potential for real-world applications.</p>
        </div>

        <div class="section">
            <h2>💻 Quick Start</h2>
            <p>Get started with the Basket Transformer in just a few steps:</p>
            <pre>
git clone https://github.com/gursanjjam/basket-transformer.git
cd basket-transformer

# Install dependencies
pip install -r requirements.txt

# Train the model (20 minutes on CPU)
python notebooks/train_basket_transformer.py

# Make predictions
python notebooks/predict_nice.py
            </pre>
        </div>

        <div class="section">
            <h2>📊 Training Curve</h2>
            <p>Here's the training loss curve after one epoch:</p>
            <img src="assets/loss.png" alt="Training Loss Curve">
        </div>

        <div class="section">
            <h2>📝 Roadmap</h2>
            <p>Future work includes:</p>
            <ul>
                <li>Training on the full 2 million sequences overnight</li>
                <li>Adding category-level embeddings (DEPARTMENT, BRAND)</li>
                <li>Deploying a Hugging Face Space demo</li>
                <li>Exporting the model to ONNX for mobile checkout apps</li>
            </ul>
        </div>

        <div class="section">
            <h2>🤝 Contributing</h2>
            <p>Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.</p>
        </div>

        <div class="section">
            <h2>📜 License</h2>
            <p>MIT © [Your Name]</p>
        </div>
    </div>
</body>
</html>
