import json
import os
from textattack.datasets import Dataset
from transformers import AutoTokenizer

# Set environment variables
label_key = os.getenv('LABEL_KEY', 'label')
dataset_file = os.getenv('MY_DATASET_FILE', 'default_dataset.json')

# Load tokenizer to get the separator token
tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-large')

# Load JSON data
data = []
try:
    with open(dataset_file, 'r') as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print(f"JSON decoding error: {e}")
    print("Attempting to load line by line...")
    with open(dataset_file, 'r') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error on line {idx}: {e}")
                    continue

# Create label map
labels = sorted(set(item[label_key].lower() for item in data if label_key in item))
label_map = {label: i for i, label in enumerate(labels)}

# Create examples
examples = []
for item in data:
    if label_key in item and 'sentence1' in item and 'sentence2' in item:
        combined_text = item['sentence1'].strip() + ' ' + tokenizer.sep_token + ' ' + item['sentence2'].strip()
        label = label_map[item[label_key].lower()]
        examples.append((combined_text, label))
    else:
        print(f"Skipping item: {item}")

# Print debug information
print(f"First few examples: {examples[:2]}")

# Define the dataset variable
dataset = Dataset(examples)
