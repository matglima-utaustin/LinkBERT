import numpy as np
import random
import string
from typing import List, Dict
from dataclasses import dataclass
from datasets import Dataset

class DatasetCorruptor:
    def __init__(self, dataset):
        self.dataset = dataset

    def corrupt(self, method):
        corruption_method = getattr(self, f'_corrupt_{method}')
        corrupted_data = [corruption_method(item) for item in self.dataset]
        return Dataset.from_dict({k: [d[k] for d in corrupted_data] for k in corrupted_data[0].keys()})

    def _corrupt_random_question(self, item):      
        return {
            "id": item["id"],
            "sentence1": random.choice([d['sentence1'] for d in self.dataset]),
            "sentence2": item["sentence2"],
            "label": item["label"]
        }

    def _corrupt_random_passage(self, item):
        return {
            "id": item["id"],
            "sentence1": item["sentence1"],
            "sentence2": random.choice([d.sentence2 for d in self.dataset]),
            "label": item["label"]
        }

    def _corrupt_passage_with_gibberish(self, item):
        answer_candidates = self._extract_answer_candidates(item.sentence2)
        gibberish = self._generate_gibberish(len(item.sentence2))
        
        for candidate in answer_candidates:
            insert_position = random.randint(0, len(gibberish))
            gibberish = f"{gibberish[:insert_position]}{candidate}{gibberish[insert_position:]}"
        
        return DataItem(id=item.id, sentence1=item.sentence1, sentence2=gibberish, label=item.label)

    @staticmethod
    def _extract_answer_candidates(passage: str) -> List[str]:
        # This method should extract potential answer candidates from the passage
        # The implementation would depend on your specific dataset and task
        # For now, let's assume it splits the passage into words
        return passage.split()

    @staticmethod
    def _generate_gibberish(length: int) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + ' ', k=length))


LABELS = ['activating invasion and metastasis', 'avoiding immune destruction',
          'cellular energetics', 'enabling replicative immortality', 'evading growth suppressors',
          'genomic instability and mutation', 'inducing angiogenesis', 'resisting cell death',
          'sustaining proliferative signaling', 'tumor promoting inflammation']


def divide(x, y):
    return np.true_divide(x, y, out=np.zeros_like(x, dtype=np.float), where=y != 0)


def compute_p_r_f(preds, labels):
    TP = ((preds == labels) & (preds != 0)).astype(int).sum()
    P_total = (preds != 0).astype(int).sum()
    L_total = (labels != 0).astype(int).sum()
    P  = divide(TP, P_total).mean()
    R  = divide(TP, L_total).mean()
    F1 = divide(2 * P * R, (P + R)).mean()
    return P, R, F1


def eval_hoc(true_list, pred_list, id_list):
    data = {}

    assert len(true_list) == len(pred_list) == len(id_list), \
        f'Gold line no {len(true_list)} vs Prediction line no {len(pred_list)} vs Id line no {len(id_list)}'

    cat = len(LABELS)
    assert cat == len(true_list[0]) == len(pred_list[0])

    for i in range(len(true_list)):
        id = id_list[i]
        key = id.split('_')[0]
        if key not in data:
            data[key] = (set(), set())

        for j in range(cat):
            if true_list[i][j] == 1:
                data[key][0].add(j)
            if pred_list[i][j] == 1:
                data[key][1].add(j)

    print (f"There are {len(data)} documents in the data set")
    # print ('data', data)

    y_test = []
    y_pred = []
    for k, (true, pred) in data.items():
        t = [0] * len(LABELS)
        for i in true:
            t[i] = 1

        p = [0] * len(LABELS)
        for i in pred:
            p[i] = 1

        y_test.append(t)
        y_pred.append(p)

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    p, r, f1 = compute_p_r_f(y_pred, y_test)
    return {"precision": p, "recall": r, "F1": f1}
