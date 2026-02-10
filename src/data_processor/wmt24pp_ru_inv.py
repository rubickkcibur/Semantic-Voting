from datasets import load_dataset
import re
import os
import evaluate
import transformers
from external_lib import mt5_model
import torch
import tqdm
import logging

DATA_PATH="wmt24pp" # customize to local data path
COT_EXAMPLES_chat = []
SYSTEM_PROMPT = r"You are a translation assistant who carefully and thoughtfully translates English sentences into Russian, ensuring that the translated sentences fluent and accurately convey the original meaning. Place your translation answer in \boxed{}. For axmaple, if the answer is 'Привет мир', you should output \boxed{Привет мир}.\n"

COT_EXAMPLES_base = [
    d["content"]
    for d in COT_EXAMPLES_chat
]
COT_EXAMPLES_base = "".join(COT_EXAMPLES_base)

GROUND_KEYS = {"answer"}
REPORT_METRICS = {"bleu", "rougeL", "no_ref_MQM_score", "ref_MQM_score"}
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
logger = logging.getLogger("MainLogger.wmt24pp_ru_inv")

def load_data(cot: bool = False):
    dataset = load_dataset(DATA_PATH, "en-ru_RU")
    dataset["train"] = dataset["train"].filter(lambda example: not example["is_bad_source"])
    dataset["train"] = dataset["train"].map(lambda example: {
        "prompt": [
                dict(role="system", content=SYSTEM_PROMPT),
                dict(role="user", content="The English sentence is: '{}'\nPlease think about how to translate step by step.\n".format(example["source"])),
                # dict(role="assistant", content="Answer: {}\n".format(format_answer(example["answer"])))
            ]
    })
    dataset["test"] = dataset["train"].map(lambda example: {
        "prompt": [
                dict(role="system", content=SYSTEM_PROMPT)
            ] + 
            (COT_EXAMPLES_chat if cot else []) + 
            [
                dict(role="user", content="The English sentence is: '{}'\nPlease think about how to translate step by step.\n".format(example["source"]))
            ]
    })
    return dataset


def reward_model_score(pred_txt, kwargs_list):
    model = mt5_model.MT5ForRegression.from_pretrained("metricx-24-hybrid-large-v2p6-bfloat16", torch_dtype=torch.bfloat16)
    model.to("cuda:0")
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl", trust_remote_code=True)
    tokenizer.padding_side = "left"
    batch_size = 16
    pred_txt = [pred if pred is not None else "None" for pred in pred_txt]
    total_length = len(pred_txt)
    no_reference_scores = []
    with tqdm.tqdm(total=total_length, desc="Calculating MQM scores without reference") as pbar:
        for i in range(0, total_length, batch_size):
            batch_preds = pred_txt[i:i+batch_size]
            batch_source = kwargs_list["source"][i:i+batch_size]
            batch_texts = [
                "source: " + source + " candidate: " + pred
                for pred, source in zip(batch_preds, batch_source)
            ]
            batch_inputs = tokenizer(
                batch_texts,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to("cuda:0")
            with torch.no_grad():
                outputs = model(**batch_inputs)
                scores = outputs.predictions.detach().cpu().float().numpy().tolist()
            no_reference_scores.extend(scores)
            pbar.update(len(batch_texts))
    reference_scores = []
    with tqdm.tqdm(total=total_length, desc="Calculating MQM scores with reference") as pbar:
        for i in range(0, total_length, batch_size):
            batch_preds = pred_txt[i:i+batch_size]
            batch_source = kwargs_list["source"][i:i+batch_size]
            batch_refs = kwargs_list["target"][i:i+batch_size]
            batch_texts = [
                "source: " + source + " candidate: " + pred + " reference: " + ref
                for pred, source, ref in zip(batch_preds, batch_source, batch_refs)
            ]
            batch_inputs = tokenizer(
                batch_texts,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to("cuda:0")
            with torch.no_grad():
                outputs = model(**batch_inputs)
                scores = outputs.predictions.detach().cpu().float().numpy().tolist()
            reference_scores.extend(scores)
            pbar.update(len(batch_texts))
    return [
        {"no_ref_MQM_score": no_ref, "ref_MQM_score": ref}
        for no_ref, ref in zip(no_reference_scores, reference_scores)
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
    refs = [kwargs["target"] for kwargs in kwargs_list]
    MQM_scores = reward_model_score(preds, kwargs_list)

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    total_results = []
    logger.info("Calculating BLEU and RougeL scores...")
    for pred, ref in zip(preds, refs):
        if pred is None or len(pred) <= 0:
            total_results.append({"bleu": 0, "rougeL": 0})
        else:
            comp_result = dict()
            bleu_results = bleu_metric.compute(predictions=[pred], references=[ref])
            if bleu_results is None:
                comp_result.update({"bleu": 0})
            else:
                comp_result.update(bleu_results)
            rouge_results = rouge_metric.compute(predictions=[pred], references=[ref])
            if rouge_results is None:
                comp_result.update({"rougeL": 0})
            else:
                comp_result.update(rouge_results)
            total_results.append(comp_result)
    assert len(total_results) == len(MQM_scores)
    for i in range(len(total_results)):
        total_results[i].update(MQM_scores[i])
    return total_results