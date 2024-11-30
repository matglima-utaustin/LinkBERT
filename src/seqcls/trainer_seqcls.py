# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
A subclass of `Trainer` specific to Question-Answering tasks
Customized Trainer for Sequence Classification tasks.
Streamlined for compatibility with updated libraries and necessary functionality.
"""

from transformers import Trainer
from transformers.trainer_utils import PredictionOutput

class SeqClsTrainer(Trainer):
    """
    Custom Trainer for sequence classification tasks, modified for metric computation and compatibility.
    """
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # Temporarily disable metrics computation during the loop
        compute_metrics = self.compute_metrics
        self.compute_metrics = None

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            ignore_keys=ignore_keys,
        )

        self.compute_metrics = compute_metrics

        # Compute metrics if a function is provided
        metrics = compute_metrics(output, eval_dataset) if compute_metrics else {}
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}  # Add prefix to metric keys

        self.log(metrics)
        return metrics

    def predict(self, predict_dataset, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metrics computation during the loop
        compute_metrics = self.compute_metrics
        self.compute_metrics = None

        output = self.evaluation_loop(
            predict_dataloader,
            description="Prediction",
            ignore_keys=ignore_keys,
        )

        self.compute_metrics = compute_metrics

        # Compute metrics if a function is provided
        metrics = compute_metrics(output, predict_dataset) if compute_metrics else {}
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}  # Add prefix to metric keys

        self.log(metrics)
        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=metrics)
