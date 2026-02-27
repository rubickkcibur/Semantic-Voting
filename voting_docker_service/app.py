from fastapi import FastAPI
from pydantic import BaseModel
import threading
import os
import torch
import transformers

from torch import device
from SV import compute_scores

app = FastAPI()

model = None
model_lock = threading.Lock()


class LoadRequest(BaseModel):
    model_path: str
    device: str = "cuda:0"  # default to CPU, can be set to "cuda" for GPU


class PredictRequest(BaseModel):
    candidates: list[str]
    clustering: str = "whole"  # whole or HDBSCAN
    batch_size: int = 32,
    HDBSCAN_min_cluster_size: int = 5
    HDBSCAN_min_samples: int = 2


@app.post("/load_model")
def load_model(req: LoadRequest):
    global emb_model
    global emb_tokenizer

    with model_lock:
        model_path = os.path.join("/models", req.model_path)
        if not os.path.exists(model_path):
            return {"status": "The model path does not exist, please make sure the model is under local the directory which is bound to '/models'"}
        device = req.device
        try:
            emb_tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_path,
                padding_side="left",
            )
            emb_model = transformers.AutoModel.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
            emb_model.eval()
        except Exception as e:
            return {"status": f"Failed to load model because of following exception:\n {str(e)}"}

    return {"status": "model loaded"}


@app.post("/predict")
def predict(req: PredictRequest):
    global emb_model
    global emb_tokenizer

    if emb_model is None:
        return {"status": "Please first load the model by calling /load_model endpoint"}
    cands = req.candidates
    if cands is None or len(cands) == 0:
        return {"status": "No candidates provided"}
    if any([not isinstance(c, str) for c in cands]):
        return {"status": "All candidates should be strings"}
    if req.clustering not in ["whole", "HDBSCAN"]:
        return {"status": "Invalid clustering method, please choose either 'whole' or 'HDBSCAN'"}
    if req.clustering == "HDBSCAN" and (not isinstance(req.HDBSCAN_min_cluster_size, int) or req.HDBSCAN_min_cluster_size <= 0 or not isinstance(req.HDBSCAN_min_samples, int) or req.HDBSCAN_min_samples <= 0 or req.HDBSCAN_min_samples > req.HDBSCAN_min_cluster_size):
        return {"status": "Invalid HDBSCAN parameters, min_cluster_size and min_samples should be positive integers, and min_samples should be less than or equal to min_cluster_size"}
    if req.batch_size and req.batch_size <= 0:
        return {"status": "Invalid batch size, it should be a positive integer"}
    try:
        scores = compute_scores(cands, emb_model, emb_tokenizer, batch_size=req.batch_size, clustering_method=req.clustering, min_cluster_size=req.HDBSCAN_min_cluster_size, min_samples=req.HDBSCAN_min_samples)
    except Exception as e:
        return {"status": f"Failed to compute scores because of following exception:\n {str(e)}"}

    return {"scores": scores, "status": "Done"}