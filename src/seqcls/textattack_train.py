import json
import argparse
import textattack
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

# Function to load data from JSON file
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    label_map = {'yes': 0, 'no': 1, 'maybe': 2}
    dataset = [((example['sentence1'], example['sentence2']), label_map[example['label']]) for example in data]
    return dataset

# Load training and evaluation data
train_data = load_data(args.train_path)
eval_data = load_data(args.eval_path)

# Wrap datasets for TextAttack
train_dataset = textattack.datasets.Dataset(train_data)
eval_dataset = textattack.datasets.Dataset(eval_data)

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, truncation=True, max_length=512)
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
