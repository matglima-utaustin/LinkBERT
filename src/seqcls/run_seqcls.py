#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification.

Adapted from
https://github.com/huggingface/transformers/blob/72aee83ced5f31302c5e331d896412737287f976/examples/pytorch/text-classification/run_glue.py
"""
#!/usr/bin/env python
# coding=utf-8
"""
Refactored `run_seqcls.py` for fine-tuning sequence classification models with updated libraries.
Focuses only on functionality required for the bash script.
"""

import logging
import os
import numpy as np
from datasets import load_dataset
import evaluate  # For loading metrics
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Preprocessing function
def preprocess_function(examples):
    args = (examples["sentence"], examples.get("sentence2"))
    return tokenizer(*args, padding="max_length", truncation=True, max_length=max_seq_length)

# Compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)

# Main script function
def main():
    # Parse arguments
    parser = HfArgumentParser(TrainingArguments)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Load datasets
    raw_datasets = load_dataset("json", data_files={
        "train": data_args.train_file,
        "validation": data_args.validation_file,
        "test": data_args.test_file,
    })

    # Load model, tokenizer, and config
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=2)  # Update `num_labels` as needed
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)

    # Tokenize datasets
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    if training_args.do_train:
        trainer.train()
        trainer.save_model()

    # Evaluate
    if training_args.do_eval:
        metrics = trainer.evaluate()
        logger.info(metrics)

    # Predict
    if training_args.do_predict:
        predictions = trainer.predict(test_dataset=tokenized_datasets["test"]).predictions
        predictions = np.argmax(predictions, axis=1)
        output_path = os.path.join(training_args.output_dir, "test_predictions.txt")
        np.savetxt(output_path, predictions, fmt="%d")

if __name__ == "__main__":
    main()

