# Model Training & Tokenization

This document outlines the process used to train the BERT-based Japanese Sentiment Analysis model and the tokenization strategy required for inference.

## 1. Dataset Overview
The model was fine-tuned on a curated subset of Japanese customer reviews.
- **Source:** [Amazon Reviews (Japanese)](https://huggingface.co/datasets/amazon_reviews_multi)
- **File:** `data/japanese_reviews_500_preprocessed.csv`
- **Size:** 500 pre-processed reviews.
- **Classes:** 
    - `0`: Negative
    - `1`: Positive
- **Pre-processing:** The text was cleaned to remove HTML tags and special characters, with a `clean_text` column used as the primary input.

## 2. Base Model
- **Architecture:** BERT (Bidirectional Encoder Representations from Transformers)
- **Pre-trained Model:** [`tohoku-nlp/bert-base-japanese-v3`](https://huggingface.co/tohoku-nlp/bert-base-japanese-v3)
- **Why this model?** It is the industry standard for Japanese NLP, pre-trained on Japanese Wikipedia using the MeCab morphological analyzer.

## 3. Tokenization Process
Japanese does not use spaces between words, so tokenization is a two-step process:
1.  **Morphological Analysis:** Using **MeCab** (via the `fugashi` library in Python) to segment sentences into morphemes (words).
2.  **Subword Segmentation:** Using **WordPiece** to break words into smaller tokens based on the model's vocabulary.

**For Inference in Ruby:**
We will use the [`tokenizers` gem](https://github.com/huggingface/tokenizers) which can load the `tokenizer.json` or `vocab.txt` generated during training to ensure identical tokenization.

## 4. Training Hyperparameters
The model was fine-tuned using the following configuration:
- **Epochs:** 3
- **Learning Rate:** 2e-5
- **Optimizer:** AdamW
- **Batch Size:** 16
- **Max Sequence Length:** 128 tokens
- **Loss Function:** Cross-Entropy Loss

## 5. Performance Metrics
The model achieved the following on the test set:
- **Accuracy:** ~90%+ (on balanced validation data)
- **Export Format:** ONNX (Opset 17) for cross-platform compatibility.

## 6. Export to ONNX
To move from PyTorch to Ruby, the model was exported using:
```python
torch.onnx.export(
    model, 
    (input_ids, attention_mask), 
    "model.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['output'],
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'seq_len'}, ...}
)
```
This allows the Ruby `onnxruntime` gem to run the model without a Python environment.
