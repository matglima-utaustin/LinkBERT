import json
import os
from textattack.datasets import Dataset

label_key = os.getenv('LABEL_KEY', 'label')
dataset_file = os.getenv('MY_DATASET_FILE', 'default_dataset.json')

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

labels = sorted(set(item[label_key].lower() for item in data if label_key in item))
label_map = {label: i for i, label in enumerate(labels)}

examples = []
for item in data:
    if label_key in item and 'sentence1' in item and 'sentence2' in item:
        examples.append(((item['sentence1'], item['sentence2']), label_map[item[label_key].lower()]))
    else:
        print(f"Skipping item: {item}")

dataset = Dataset(examples)
