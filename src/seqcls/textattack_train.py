import json
import argparse
import textattack
import torch
from textattack.datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, required=True)
parser.add_argument('--eval_path', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--output_dir', type=str, default='./trained_model')
parser.add_argument('--max_length', type=int, default=512)
args = parser.parse_args()

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

class CustomWrapper(HuggingFaceModelWrapper):
    def __call__(self, text_input_list):
        inputs = tokenizer(
            text_input_list,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        )
        outputs = self.model(**inputs)
        return outputs.logits

def load_data(dataset_file):
    data = []
    with open(dataset_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    examples = []
    labels = {'yes': 1, 'no': 0}
    
    for item in data:
        text = f"{item['sentence1']} {tokenizer.sep_token} {item['sentence2']}"
        label = labels[item['label'].lower()]
        examples.append((text, label))
    
    return Dataset(examples)

# Load datasets and create model wrapper
train_dataset = load_data(args.train_path)
eval_dataset = load_data(args.eval_path)
model_wrapper = CustomWrapper(model, tokenizer)

# Setup attack and training arguments
attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
training_args = textattack.TrainingArgs(
    num_epochs=args.num_epochs,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    output_dir=args.output_dir
)

# Create trainer and train
trainer = textattack.Trainer(
    model_wrapper=model_wrapper,
    attack=attack,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    training_args=training_args
)

try:
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training completed successfully!")
except Exception as e:
    print(f"An error occurred during training: {str(e)}")
    raise e
