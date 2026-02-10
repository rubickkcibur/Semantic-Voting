from datasets import load_dataset
import re
import os
import evaluate
from external_lib import mt5_model
import torch
import tqdm
import logging
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

DATA_PATH="alpaca_eval" # customize to local data path
COT_EXAMPLES_chat = []

COT_EXAMPLES_base = [
    d["content"]
    for d in COT_EXAMPLES_chat
]
COT_EXAMPLES_base = "".join(COT_EXAMPLES_base)

GROUND_KEYS = {"answer"}
REPORT_METRICS = {"placeholder_metric"}
logger = logging.getLogger("MainLogger.alpaca_eval")

def load_data(cot: bool = False):
    dataset = load_dataset("json", data_files=DATA_PATH)
    dataset["train"] = dataset["train"].map(lambda example: {
        "prompt": [
                dict(role="user", content="{}\nPlease answer with no more than 80 words.\n".format(example["instruction"]))
            ]
    })
    dataset["test"] = dataset["train"].map(lambda example: {
        "prompt": [
                dict(role="user", content="{}\nPlease answer with no more than 80 words.\n".format(example["instruction"]))
            ]
    })
    if "validation" in dataset:
        del dataset["validation"]
    return dataset


def metric(output_text, kwargs_list):
    length = len(output_text)
    return [{"placeholder_metric": 0}] * length