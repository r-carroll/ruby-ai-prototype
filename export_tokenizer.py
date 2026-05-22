
from transformers import AutoTokenizer
import os
import json

# Load the tokenizer
tokenizer_name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Ensure directory
output_dir = "omikuji_app/vendor/models"
os.makedirs(output_dir, exist_ok=True)
json_path = os.path.join(output_dir, "tokenizer.json")

# For older or non-fast tokenizers, we can still use save_pretrained
# which generates a tokenizer.json if a fast version is available or requested.
# But for tohoku-nlp/bert-base-japanese-v3, let's try to force the fast export.
try:
    print(f"Exporting tokenizer to {json_path}...")
    # This specifically saves the fast tokenizer format
    tokenizer.save_pretrained(output_dir)
    # Check if tokenizer.json was created
    if os.path.exists(os.path.join(output_dir, "tokenizer.json")):
        print("Done: tokenizer.json created successfully.")
    else:
        print("Warning: save_pretrained didn't create tokenizer.json. Trying manual save...")
        # Fallback for older transformers
        if hasattr(tokenizer, 'tokenizer'):
            tokenizer.tokenizer.save(json_path)
        else:
             # Just save vocab if all else fails, but we want the JSON format
             tokenizer.save_vocabulary(output_dir)
except Exception as e:
    print(f"Failed to export: {e}")
