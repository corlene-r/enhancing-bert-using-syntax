import os
import random
import logging

import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    results = dict()

    results["accuracy"] = accuracy_score(labels, preds)

    zero_division = 0

    results["macro_precision"], results["macro_recall"], results["macro_f1"], _ \
        = precision_recall_fscore_support(labels, preds, average="macro", zero_division=zero_division)

    results["micro_precision"], results["micro_recall"], results["micro_f1"], _ \
        = precision_recall_fscore_support(labels, preds, average="micro", zero_division=zero_division)

    results["weighted_precision"], results["weighted_recall"], results["weighted_f1"], _ \
        = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=zero_division)

    out_headers = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "micro_precision", "micro_recall", "micro_f1", "weighted_precision", "weighted_recall", "weighted_f1"]
    return (
        results,
        out_headers,
        [results[header] for header in out_headers]
    )
