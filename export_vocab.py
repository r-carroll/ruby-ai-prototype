
from transformers import AutoTokenizer
import os

# Load the tokenizer we used
tokenizer_name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Ensure directory
output_dir = "ruby/app/models"
os.makedirs(output_dir, exist_ok=True)
vocab_path = os.path.join(output_dir, "vocab.txt")

# Save vocab
print(f"Saving vocab to {vocab_path}...")
tokenizer.save_vocabulary(output_dir)
print("Done.")
