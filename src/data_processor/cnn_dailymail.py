from datasets import load_dataset
import re
import os
import evaluate
import transformers
from external_lib import mt5_model
import torch
import tqdm
import logging
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
MACLAB_NAS_NAME = os.environ["MACLAB_NAS_NAME"]

DATA_PATH="/mnt/{}/rubickjiang/public_dataset/cnn_dailymail".format(MACLAB_NAS_NAME)
COT_EXAMPLES_chat = []
SYSTEM_PROMPT = r"You are a skilled summarization assistant. When provided with a news report, you will carefully read and understand its content, then generate a concise and informative summary in the form of several short highlights. Each highlight should capture a key point from the article. Place your final summary inside \boxed{}. For example, if the summary is \"Hello World\", you should output: \boxed{Hello World}"

COT_EXAMPLES_base = [
    d["content"]
    for d in COT_EXAMPLES_chat
]
COT_EXAMPLES_base = "".join(COT_EXAMPLES_base)

GROUND_KEYS = {"answer"}
REPORT_METRICS = {"rougeL", "BLEURT"}
logger = logging.getLogger("MainLogger.cnn_dailymail")

def load_data(cot: bool = False):
    filter_tokenizer = BleurtTokenizer.from_pretrained('/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/BLEURT-20')
    def data_filter(example):
        return len(example["article"]) <= 7000 and len(example["highlights"]) > 0
        # if len(example["highlights"]) <= 0:
        #     return False
        # if len(example["article"]) <= 7000:
        #     return True
        # if len(example["article"]) >= 15000:
        #     return False
        # tokens = filter_tokenizer(example["article"], return_tensors="pt")
        # return len(tokens.input_ids[0]) <= 1900 and len(example["highlights"]) > 0
    dataset = load_dataset(DATA_PATH, "3.0.0")
    dataset["train"] = dataset["train"].filter(data_filter)
    dataset["train"] = dataset["train"].select(range(1000))
    # dataset["train"] = dataset["train"].filter(lambda example: not example["is_bad_source"])
    dataset["train"] = dataset["train"].map(lambda example: {
        "prompt": [
                dict(role="system", content=SYSTEM_PROMPT),
                dict(role="user", content="Here is the news report:\n'{}'\nPlease give your thoughtful summary.\n".format(example["article"])),
                # dict(role="assistant", content="Answer: {}\n".format(format_answer(example["answer"])))
            ]
    })
    dataset["test"] = dataset["train"].map(lambda example: {
        "prompt": [
                dict(role="system", content=SYSTEM_PROMPT)
            ] + 
            (COT_EXAMPLES_chat if cot else []) + 
            [
                dict(role="user", content="Here is the news report:\n'{}'\nPlease give your thoughtful summary.\n".format(example["article"]))
            ]
    })
    if "validation" in dataset:
        del dataset["validation"]
    return dataset


def reward_model_score(pred_txt, kwargs_list):
    model = BleurtForSequenceClassification.from_pretrained('/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/BLEURT-20')
    tokenizer = BleurtTokenizer.from_pretrained('/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/BLEURT-20')
    model.to("cuda:0")
    model.eval()
    batch_size = 16
    pred_txt = [pred if pred is not None else "None" for pred in pred_txt]
    total_length = len(pred_txt)
    BLEURTs = []
    with tqdm.tqdm(total=total_length, desc="Calculating BLEURT scores") as pbar:
        for i in range(0, total_length, batch_size):
            batch_preds = pred_txt[i:i+batch_size]
            batch_refs = kwargs_list["highlights"][i:i+batch_size]
            encodings = tokenizer(batch_refs, batch_preds, return_tensors='pt', padding="longest").to("cuda:0")
            with torch.no_grad():
                scores = model(**encodings).logits.flatten().detach().cpu().float().numpy().tolist()
            BLEURTs.extend(scores)
            pbar.update(len(batch_preds))
    return [
        {"BLEURT": bleurt}
        for bleurt in BLEURTs
    ]

def metric(output_text, kwargs_list):
    def extract_pred(txt):
        pattern = r'\\boxed\{([^}]*)\}'
        results = re.findall(pattern, txt)
        if results:
            ret = results[0]
            # ret = ret.replace("\"", "")
            # ret = ret.replace("\'", "")
            ret = ret.strip()
            return ret
        else:
            return None
    preds = [extract_pred(text) for text in output_text]
    refs = [kwargs["highlights"] for kwargs in kwargs_list]
    BLEURT_scores = reward_model_score(preds, kwargs_list)

    rouge_metric = evaluate.load("rouge")
    total_results = []
    logger.info("Calculating RougeL scores...")
    for pred, ref in zip(preds, refs):
        if pred is None or len(pred) <= 0:
            total_results.append({"rougeL": 0})
        else:
            comp_result = dict()
            rouge_results = rouge_metric.compute(predictions=[pred], references=[ref])
            if rouge_results is None:
                comp_result.update({"rougeL": 0})
            else:
                comp_result.update(rouge_results)
            total_results.append(comp_result)
    assert len(total_results) == len(BLEURT_scores)
    for i in range(len(total_results)):
        total_results[i].update(BLEURT_scores[i])
    return total_results