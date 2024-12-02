import json
import argparse
import textattack
from textattack.datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper

# Define command-line arguments
parser = argparse.ArgumentParser(description='Train a model using TextAttack.')
parser.add_argument('--train_path', type=str, required=True, help='Path to the training data JSON file.')
parser.add_argument('--eval_path', type=str, required=True, help='Path to the evaluation data JSON file.')
parser.add_argument('--model_name', type=str, required=True, help='HuggingFace model name.')
parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs.')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
parser.add_argument('--output_dir', type=str, default='./trained_model', help='Output directory for model and logs.')
args = parser.parse_args()

# Load tokenizer to get the separator token
tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-large')
# Label name
label_key='label'

def preprocess_function(examples):
        sentence1_key, sentence2_key = 'sentence1','sentence2'
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )

        result = tokenizer(*args, padding=True, max_length=512, truncation=True)
        # Create label map
        labels = ['yes','no']
        label_map = {label: i for i, label in enumerate(labels)}
        result["label"] = label_map[item[label_key].lower()]
        return result

def load_data(dataset_file):
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
    
    
    
    # Create examples
    examples = []
    for line in data:
        example=preprocess_function(line)
        examples.append(example)

    
    # Print debug information
    print(f"First few examples: {examples[:2]}")
    
    # Define the dataset variable
    dataset = Dataset(examples)
    return dataset
    
train_dataset = load_data(args.train_path)
eval_dataset = load_data(args.eval_path)

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, truncation=True, padding=True, max_length=512)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)

# Wrap the model for TextAttack
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Choose an attack recipe
attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)

# Set up Training Arguments
training_args = textattack.TrainingArgs(
    num_epochs=3,
    num_clean_epochs=1,
    num_train_adv_examples=1000,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    log_to_tb=True,
)

# Create the Trainer and train the model
trainer = textattack.Trainer(
    model_wrapper=model_wrapper,
    attack=attack,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    training_args=training_args
)

trainer.train()
