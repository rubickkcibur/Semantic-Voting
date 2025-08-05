import transformers
import torch
import jsonlines
from external_lib import mt5_model
import umap
import re
import numpy as np
from sklearn.preprocessing import normalize
import plotly.graph_objects as go
import plotly.io as pio
import plotly

def draw_clusters(source, candidates):
    device = "cuda:0"

    model = mt5_model.MT5ForRegression.from_pretrained("/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/metricx-24-hybrid-large-v2p6-bfloat16", torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl", trust_remote_code=True)
    tokenizer.padding_side = "left"
    batch_texts = [
        "source: " + source + " candidate: " + pred
        for pred, source in zip(
            [source] * len(candidates), 
            [cand if cand is not None else "None" for cand in candidates])
    ]
    batch_inputs = tokenizer(
        batch_texts,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**batch_inputs)
        scores = outputs.predictions.detach().cpu().float().numpy().tolist()


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/unsup-simcse-bert-base-uncased",
        padding_side="left",
    )
    model = transformers.AutoModel.from_pretrained(
        "/mnt/maclabcv2/rubickjiang/proj_storage/huggingface_models/unsup-simcse-bert-base-uncased",
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    def is_invalid(txt):
        if txt is None or txt == "None":
            return True
        if not txt.isascii():
            return True
        return False

    none_idx = [i for i in range(len(candidates)) if is_invalid(candidates[i])]
    cands = [d for idx, d in enumerate(candidates) if idx not in none_idx]
    scores = [d for idx, d in enumerate(scores) if idx not in none_idx]
    encodings = tokenizer(
        cands,
        return_tensors="pt",
        padding='longest',
        truncation=True,
        max_length=512,).to(device)
    with torch.no_grad():
        output = model(**encodings)
        emb = torch.nn.functional.normalize(output.last_hidden_state[:, 0, :], p=2, dim=1)
        emb = emb.detach().cpu().float().numpy()
    reducer = umap.UMAP(n_neighbors=5, metric='cosine', n_components=3, random_state=42, min_dist = 0.5)
    emb_umap = reducer.fit_transform(emb)
    emb_umap = normalize(emb_umap, axis=1, norm='l2')
    trace = go.Scatter3d(
        x = emb_umap[:, 0],
        y = emb_umap[:, 1],
        z = emb_umap[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=scores,  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8,
            colorbar=dict(title='Scores')
        )
    )
    layout = go.Layout(title = "clusters")
    fig = go.Figure(data=[trace], layout=layout)
    # pio.write_image(fig, "./test.png")
    plotly.offline.plot(fig, filename='clusters.html')
    
    print(emb_umap.shape)
    print(len(scores))

def extract_data(obj):
    def extract_pred(txt):
        pattern = r'\\boxed\{([^}]*)\}'
        results = re.findall(pattern, txt)
        if results:
            ret = results[0]
            ret = ret.strip()
            return ret
        else:
            return None
    prompt = obj["prompt"]
    prompt = prompt.split("sentence is: '")[-1].split("'\nPlease think about how to translate step by step.")[0].strip()
    candidates = obj["outputs"]
    candidates = [extract_pred(cand) for cand in candidates]
    return prompt, candidates

with jsonlines.open("/mnt/maclabcv2/rubickjiang/codes/open-r1/data/SR_candidates/Qwen2.5-1.5B-Instruct/wmt24pp_zh_output_64.jsonl", "r") as reader:
    objs = [obj for obj in reader]
    prompt, candidates = extract_data(objs[0])
    draw_clusters(prompt, candidates)

