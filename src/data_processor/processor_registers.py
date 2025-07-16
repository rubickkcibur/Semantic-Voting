from data_processor import gsm8k, wmt24pp_zh, wmt24pp_de, wmt24pp_fr, wmt19_zh, wmt24pp_ru, wmt24pp_es, cnn_dailymail
from transformers import PreTrainedTokenizer

def load_custom_dataset(dataset_name: str, tokenizer: PreTrainedTokenizer = None, cot: bool = False, apply_chat_template: bool = False):
    dataset_registers = {
        "gsm8k": gsm8k.load_data,
        "wmt24pp_zh": wmt24pp_zh.load_data,
        "wmt24pp_de": wmt24pp_de.load_data,
        "wmt19_zh": wmt19_zh.load_data,
        "wmt24pp_fr": wmt24pp_fr.load_data,
        "wmt24pp_ru": wmt24pp_ru.load_data,
        "wmt24pp_es": wmt24pp_es.load_data,
        "cnn_dailymail": cnn_dailymail.load_data,
    }
    if dataset_name not in dataset_registers:
        raise ValueError(f"Dataset {dataset_name} is not registered.")
    dataset_loader = dataset_registers[dataset_name]
    dataset = dataset_loader(cot=cot)
    if apply_chat_template:
        def apply_chat_template_func(example):
            example["prompt"] = tokenizer.apply_chat_template(example["prompt"], tokenize=False)
            return example
        dataset = dataset.map(apply_chat_template_func)
    return dataset

def get_ground_keys(dataset_name: str) -> set[str]:
    ground_keys_registers = {
        "gsm8k": gsm8k.GROUND_KEYS,
    }
    if dataset_name not in ground_keys_registers:
        raise ValueError(f"Dataset {dataset_name} is not registered.")
    return ground_keys_registers[dataset_name]

def get_metrics(dataset_name: str):
    metrics_registers = {
        "gsm8k": gsm8k.metric,
        "wmt24pp_zh": wmt24pp_zh.metric,
        "wmt24pp_de": wmt24pp_de.metric,
        "wmt19_zh": wmt19_zh.metric,
        "wmt24pp_fr": wmt24pp_fr.metric,
        "wmt24pp_ru": wmt24pp_ru.metric,
        "wmt24pp_es": wmt24pp_es.metric,
        "cnn_dailymail": cnn_dailymail.metric,
    }
    if dataset_name not in metrics_registers:
        raise ValueError(f"Dataset {dataset_name} is not registered.")
    return metrics_registers[dataset_name]

REPORT_METRICS = {
    "gsm8k": gsm8k.REPORT_METRICS,
    "wmt24pp_zh": wmt24pp_zh.REPORT_METRICS,
    "wmt24pp_de": wmt24pp_de.REPORT_METRICS,
    "wmt19_zh": wmt19_zh.REPORT_METRICS,
    "wmt24pp_fr": wmt24pp_fr.REPORT_METRICS,
    "wmt24pp_ru": wmt24pp_ru.REPORT_METRICS,
    "wmt24pp_es": wmt24pp_es.REPORT_METRICS,
    "cnn_dailymail": cnn_dailymail.REPORT_METRICS,
}
