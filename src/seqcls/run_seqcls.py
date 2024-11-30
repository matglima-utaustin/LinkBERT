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

def main():
    # Setup argument parser with modern defaults
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Load datasets
    data_files = {
        "train": data_args.train_file,
        "validation": data_args.validation_file,
        "test": data_args.test_file
    }
    
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
    )

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=2,  # Update based on your task
        finetuning_task="sequence-classification",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    # Enable gradient checkpointing for memory efficiency
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Preprocessing the datasets
    padding = "max_length" if data_args.pad_to_max_length else False
    max_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Handle single and paired sequences
        texts = (examples['text'] if 'text' in examples 
                else (examples['sentence1'], examples['sentence2']) if 'sentence2' in examples 
                else examples['sentence1'])
        
        # Using modern tokenizer features
        return tokenizer(
            texts,
            padding=padding,
            max_length=max_length,
            truncation=True,
            return_tensors=None,  # Return PyTorch tensors
        )

    # Process datasets using modern mapping
    with training_args.main_process_first(desc="Dataset preprocessing"):
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=training_args.dataloader_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        train_dataset = processed_datasets["train"]
        if data_args.max_train_samples:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        eval_dataset = processed_datasets["validation"]
        if data_args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        predict_dataset = processed_datasets["test"]
        if data_args.max_predict_samples:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Modern metric computation using evaluate library
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if data_args.metric_name == "accuracy":
            metric = evaluate.load("accuracy")
            return metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)
        elif data_args.metric_name == "f1":
            metric = evaluate.load("f1")
            return metric.compute(predictions=np.argmax(predictions, axis=1), references=labels, average="weighted")
        elif data_args.metric_name == "pearsonr":
            return {
                "pearsonr": float(np.corrcoef(predictions.squeeze(), labels)[0, 1])
            }
        else:
            raise ValueError(f"Metric {data_args.metric_name} not supported")

    # Initialize trainer with modern features
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="max_length" if data_args.pad_to_max_length else "longest",
            max_length=max_length,
        ),
        callbacks=[TensorBoardCallback()],
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint:
            checkpoint = training_args.resume_from_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # Save training metrics
        metrics = train_result.metrics
        trainer.save_model()
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
            import json
            with open(output_predict_file, "w") as f:
                json.dump({
                    "predictions": predictions.predictions.tolist(),
                    "label_ids": predictions.label_ids.tolist(),
                    "metrics": predictions.metrics
                }, f)

    # Push to Hub if specified
    if training_args.push_to_hub:
        trainer.push_to_hub(
            commit_message="End of training",
            tags=["sequence-classification", data_args.metric_name],
        )

if __name__ == "__main__":
    main()