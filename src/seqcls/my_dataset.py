import json
import os
from textattack.datasets import Dataset

# Define label mapping dynamically
label_key = os.getenv('LABEL_KEY', 'label')
dataset_file = os.getenv('MY_DATASET_FILE', 'default_dataset.json')

# Load JSON data
try:
    with open(dataset_file, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file {dataset_file} was not found.")
    exit(1)

# Extract unique labels and create label_map
labels = sorted(set(item[label_key].lower() for item in data))
label_map = {label: i for i, label in enumerate(labels)}

# Create list of examples: [((sentence1, sentence2), label)]
examples = [((item['sentence1'], item['sentence2']), label_map[item[label_key].lower()]) for item in data]

# Define the dataset variable
dataset = Dataset(examples)
