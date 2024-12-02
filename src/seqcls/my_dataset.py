import json
import os
from textattack.datasets import Dataset

label_key = os.getenv('LABEL_KEY', 'label')
dataset_file = os.getenv('MY_DATASET_FILE', 'default_dataset.json')

data = []
with open(dataset_file, 'r') as f:
    for line in f:
        try:
            data.append(json.loads(line.strip()))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line: {line.strip()}")
            continue

# Create label map
labels = sorted(set(item[label_key].lower() for item in data if label_key in item))
label_map = {label: i for i, label in enumerate(labels)}

# Create examples
examples = []
for item in data:
    if label_key in item and 'sentence1' in item and 'sentence2' in item:
        examples.append(((item['sentence1'], item['sentence2']), label_map[item[label_key].lower())))
    else:
        print(f"Skipping item: {item}")

# For debugging, print the first example
if examples:
    print(f"First example: {examples[0]}")
else:
    print("No valid examples found.")

dataset = Dataset(examples)
