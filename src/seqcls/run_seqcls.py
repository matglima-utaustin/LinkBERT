# run_seqcls.py

from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    SeqClsTrainer,  # Assuming custom trainer is imported
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset, load_metric
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory"})
    use_auth_token: bool = field(default=False, metadata={"help": "Use authentication token"})

@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(default=None, metadata={"help": "Task name"})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Dataset name"})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "Dataset config name"})
    max_seq_length: int = field(default=128, metadata={"help": "Maximum sequence length"})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite cache"})
    pad_to_max_length: bool = field(default=True, metadata={"help": "Pad to max length"})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Maximum train samples"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Maximum eval samples"})
    max_predict_samples: Optional[int] = field(default=None, metadata={"help": "Maximum predict samples"})
    train_file: Optional[str] = field(default=None, metadata={"help": "Train file"})
    validation_file: Optional[str] = field(default=None, metadata={"help": "Validation file"})
    test_file: Optional[str] = field(default=None, metadata={"help": "Test file"})

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if data_args.task_name is not None:
        raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    elif data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if not is_regression:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()
            num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
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

    padding = "max_length" if data_args.pad_to_max_length else False
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        args = (examples["sentence"],) if "sentence2" not in examples else (examples["sentence1"], examples["sentence2"])
        result = tokenizer(
            *args,
            padding=padding,
            max_length=max_seq_length,
            truncation=True,
        )
        if "label" in examples:
            result["label"] = examples["label"]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    def compute_metrics(p: EvalPrediction):
        if data_args.task_name is not None:
            metric = load_metric("glue", data_args.task_name)
            return metric.compute(predictions=p.predictions, references=p.label_ids)
        else:
            metric = load_metric("accuracy")
            return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = SeqClsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        predictions = trainer.predict(predict_dataset).predictions
        if is_regression:
            predictions = np.squeeze(predictions)
        else:
            predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "predictions.txt")
        with open(output_predict_file, "w") as writer:
            writer.write("\n".join(str(pred) for pred in predictions))

if __name__ == "__main__":
    main()
