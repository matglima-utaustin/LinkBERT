#!/usr/bin/env python
# coding=utf-8

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import evaluate
import datasets
from datasets import load_dataset
import torch

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.integrations import TensorBoardCallback

# Update minimum version requirement
check_min_version("4.31.0")

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be truncated/padded to this length."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A json file containing the test data."}
    )
    metric_name: Optional[str] = field(
        default=None, metadata={"help": "The metric to use for evaluation."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Load datasets
    data_files = {
        "train": data_args.train_file,
        "validation": data_args.validation_file,
    }
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
    )

    # Convert labels to integers
    label_to_id = {"y": 1, "n": 0, "m": 2}
    num_labels = len(label_to_id)

    def convert_labels(examples):
        unique_labels = set(examples["label"])
        print("Unique labels in dataset:", unique_labels)
        
        label_mapping = {"y": "y", "n": "n", "m": "m", "s": "y", "e": "y"}
        
        unknown_labels = set()
        converted_labels = []
        for label in examples["label"]:
            mapped_label = label_mapping.get(label.lower())
            if mapped_label is None:
                unknown_labels.add(label)
                converted_labels.append(-1)
            else:
                converted_labels.append(label_to_id[mapped_label])
        
        if unknown_labels:
            print(f"Warning: Unknown labels found in the dataset: {unknown_labels}")
        
        examples["label"] = converted_labels
        return examples

    raw_datasets = raw_datasets.map(
        convert_labels,
        desc="Converting labels to ids",
    )

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="sequence-classification",
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Preprocessing the datasets
    padding = "max_length" if data_args.pad_to_max_length else False
    max_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        return tokenizer(
            text=examples["sentence1"],
            text_pair=examples["sentence2"],
            padding=padding,
            max_length=max_length,
            truncation=True,
            return_tensors=None,
        )

    # Process the datasets
    with training_args.main_process_first(desc="Dataset preprocessing"):
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
            num_proc=data_args.preprocessing_num_workers,
        )
    train_dataset = processed_datasets
    
    # Prepare datasets for training, validation, and testing
    if training_args.do_train:
        if "train" not in processed_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = processed_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    
    if training_args.do_eval:
        if "validation" not in processed_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = processed_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    
    if training_args.do_predict:
        if "test" not in processed_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = processed_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Define compute_metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = (predictions == labels).mean()
        
        # Calculate precision, recall, and F1 for each class
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=padding),
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions = trainer.predict(predict_dataset)
        
        # Save predictions
        output_predict_file = os.path.join(training_args.output_dir, "predictions.json")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as f:
                json.dump({
                    "predictions": predictions.predictions.tolist(),
                    "label_ids": predictions.label_ids.tolist(),
                    "metrics": predictions.metrics
                }, f)

    # Push to hub if specified
    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
        trainer.push_to_hub(**kwargs)

if __name__ == "__main__":
    main()
