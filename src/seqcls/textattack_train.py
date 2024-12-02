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

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def load_data(dataset_file):
    # Load JSON data
    data = []
    try:
        with open(dataset_file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e

    # Create examples in the format expected by TextAttack
    examples = []
    labels = {'yes': 1, 'no': 0}
    
    for item in data:
        # Combine sentences into a single text with separator
        text = f"{item['sentence1']} {tokenizer.sep_token} {item['sentence2']}"
        label = labels[item['label'].lower()]
        
        # TextAttack expects (text, label) pairs
        examples.append((text, label))
    
    return Dataset(examples)

# Load datasets
train_dataset = load_data(args.train_path)
eval_dataset = load_data(args.eval_path)

# Load the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

# Wrap the model for TextAttack
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Choose an attack recipe
attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)

# Set up Training Arguments
training_args = textattack.TrainingArgs(
    num_epochs=args.num_epochs,
    num_clean_epochs=1,
    num_train_adv_examples=1000,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    log_to_tb=True,
    output_dir=args.output_dir
)

# Create the Trainer and train the model
trainer = textattack.Trainer(
    model_wrapper=model_wrapper,
    attack=attack,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    training_args=training_args
)

try:
    # Train the model
    trainer.train()
    
    # Save the trained model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
except Exception as e:
    print(f"An error occurred during training: {str(e)}")
    raise e

print("Training completed successfully!")
