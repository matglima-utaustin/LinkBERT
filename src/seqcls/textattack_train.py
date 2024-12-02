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
tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-large')

def preprocess_function(example):
    # Tokenize the texts
    text_pair = (example['sentence1'], example['sentence2'])
    encoding = tokenizer(*text_pair, padding=True, max_length=512, truncation=True)
    
    # Create label map
    labels = ['yes', 'no']
    label_map = {label: i for i, label in enumerate(labels)}
    label = label_map[example['label'].lower()]
    
    return encoding, label

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
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        continue
    
    # Create examples in the format expected by TextAttack
    examples = []
    for item in data:
        encoding, label = preprocess_function(item)
        # TextAttack expects (text, label) pairs
        examples.append((
            (encoding['input_ids'], encoding['attention_mask']),
            label
        ))
    
    # Create TextAttack Dataset
    dataset = Dataset(examples)
    return dataset

# Load datasets
train_dataset = load_data(args.train_path)
eval_dataset = load_data(args.eval_path)

# Load the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)  # binary classification

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
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)
    
    # Save evaluation results
    with open(f"{args.output_dir}/eval_results.json", 'w') as f:
        json.dump(eval_results, f, indent=4)
        
except Exception as e:
    print(f"An error occurred during training: {str(e)}")
    raise e

# Optional: Add logging and metrics
def log_metrics(metrics, step):
    """Log metrics during training"""
    print(f"\nStep {step} metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

# Add a custom callback for logging
class MetricsCallback:
    def on_step_end(self, step, metrics):
        if step % 100 == 0:  # Log every 100 steps
            log_metrics(metrics, step)

# You can add the callback to the trainer if needed
# trainer.add_callback(MetricsCallback())

print("Training completed successfully!")

# Optional: Add inference example
def predict_example(text1, text2):
    """Make a prediction on a single example"""
    inputs = tokenizer(text1, text2, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    return predictions.item()

# Example usage of prediction
if __name__ == "__main__":
    # Add some example predictions
    example_texts = [
        ("Example sentence 1", "Example sentence 2"),
        ("Another sentence 1", "Another sentence 2")
    ]
    
    print("\nMaking predictions on example texts:")
    for text1, text2 in example_texts:
        prediction = predict_example(text1, text2)
        print(f"\nText 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"Prediction: {prediction} ({'yes' if prediction == 1 else 'no'})")

    # Save model configuration
    config = {
        "model_name": args.model_name,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": 512,
    }
    
    with open(f"{args.output_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\nModel and configuration saved to {args.output_dir}")
