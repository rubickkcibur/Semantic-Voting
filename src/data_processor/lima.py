from datasets import load_dataset
import re
import os
import evaluate
from external_lib import mt5_model
import torch
import tqdm
import logging
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
MACLAB_NAS_NAME = os.environ["MACLAB_NAS_NAME"]

DATA_PATH="/mnt/{}/rubickjiang/public_dataset/lima".format(MACLAB_NAS_NAME)
COT_EXAMPLES_chat = []

COT_EXAMPLES_base = [
    d["content"]
    for d in COT_EXAMPLES_chat
]
COT_EXAMPLES_base = "".join(COT_EXAMPLES_base)

GROUND_KEYS = {"answer"}
REPORT_METRICS = {"placeholder_metric"}
logger = logging.getLogger("MainLogger.lima")

def load_data(cot: bool = False):
    dataset = load_dataset(DATA_PATH, trust_remote_code=True)
    dataset["train"] = dataset["test"].map(lambda example: {
        "prompt": [
                dict(role="user", content=example["conversations"][0])
            ]
    })
    dataset["test"] = dataset["test"].map(lambda example: {
        "prompt":
            [
                dict(role="user", content=example["conversations"][0])
            ]
    })
    if "validation" in dataset:
        del dataset["validation"]
    return dataset


def metric(output_text, kwargs_list):
    length = len(output_text)
    return [{"placeholder_metric": 0}] * length