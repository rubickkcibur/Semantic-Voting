from openai import OpenAI
import jsonlines
import re
import os
from tqdm import tqdm
import random

def extract_pred(txt):
    pattern = r'boxed\{([^}]*)\}'
    results = re.findall(pattern, txt)
    if results:
        ret = results[0]
        ret = ret.strip()
        return ret
    else:
        return None

system_prompt = "Review an user's prompts and two candidate responses following five criteria:\n- Whether the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content.\n- Whether the response addresses a substantial portion of the user's question, but does not completely resolve the query or provide a direct answer.\n- Whether the response answers the basic elements of the user's question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.\n- Whether the response is clearly written from an AI Assistant's perspective, addressing the user's question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.\n- Whether the response is impeccably tailored to the user's question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.\n Pick the best candidate and enclose your answer with boxed{}. For example, if you think Response-A is the best, please output by 'boxed{A}'\n" 

system_prompt = "Review an user's prompts and two candidate responses.\nPick the best candidate and enclose your answer with boxed{}. For example, if you think Response-A is the best, please output by 'boxed{A}'\n" 


compare_pairs = "<User_prompt>\n{}\n</User_prompt>\n<Response-A>\n{}\n</Response-A>\n<Response-B>\n{}\n</Response-B>"

client = OpenAI(api_key="sk-79bfe079126242a683a6a54e40a6e857", base_url="https://api.deepseek.com")

for model in [
    "Llama-3.2-1B-Instruct",
    # "Llama-3.2-3B-Instruct",
    # "Meta-Llama-3-8B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    # "Qwen2.5-7B-Instruct",
]:
    base_alpaca = "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/{}/debug_alpaca_eval.jsonl".format(model)
    SV_alpaca = "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/alpaca_eval/models/{}-DPO-alpaca_eval-5-2/debug_alpaca_eval.jsonl".format(model)
    results = "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/alpaca_output/SV_base/{}/debug_alpaca-random.jsonl".format(model)
    os.makedirs(os.path.dirname(results), exist_ok=True)
    prompts = []
    A_cnt = 0
    B_cnt = 0
    None_cnt = 0
    with jsonlines.open(SV_alpaca) as f1, jsonlines.open(base_alpaca) as f2, jsonlines.open(results, mode='w') as writer, tqdm(total=805) as pbar:
        for item1, item2 in zip(f1, f2):
            assert item1["prompt"] == item2["prompt"]
            items = (item1["pred"], item2["pred"])
            A_id = random.randint(0, 1)
            prompt = compare_pairs.format(
                item1["prompt"], 
                items[A_id], 
                items[1 - A_id],
            )
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            A_choosed = False
            choice = response.choices[0].message.content
            choice = extract_pred(choice)
            if choice is None:
                choice = "None"
            if choice.lower() == "a":
                if A_id == 0:
                    A_cnt += 1
                    A_choosed = True
                if A_id == 1:
                    B_cnt += 1
                    A_choosed = False
            elif choice.lower() == "b":
                if A_id == 0:
                    B_cnt += 1
                    A_choosed = False
                if A_id == 1:
                    A_cnt += 1
                    A_choosed = True
            else:
                None_cnt += 1
            writer.write({
                "response_A-SV": item1["pred"],
                "response_B-base": item2["pred"],
                "best_choice": "None" if (choice.lower() not in ["a", "b"]) else ("A" if A_choosed else "B"),
            })
            pbar.update(1)
        print("A_cnt: {}, B_cnt: {}, None_cnt: {}".format(A_cnt, B_cnt, None_cnt))