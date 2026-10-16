"""FastAPI backend for the WildlifeReID web interface."""

import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from src.data.dataset import get_eval_transform
from src.models.embedding_model import EmbeddingModel
from src.retrieval.hnsw import load_index as load_hnsw
from src.retrieval.hnsw import search as hnsw_search


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"

config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
DATA_CFG = config["data"]
MODEL_CFG = config["model"]
RETRIEVAL_CFG = config["retrieval"]

EMBEDDING_MODE = DATA_CFG.get("embedding_mode", "batch_hard")
EMBEDDING_DIM = int(MODEL_CFG.get("embedding_dim", 2048))
IMAGE_SIZE = int(DATA_CFG.get("image_size", 224))
CHECKPOINT_PATH = PROJECT_ROOT / MODEL_CFG["checkpoint"]

GALLERY_DIR = PROJECT_ROOT / "artifacts" / "indexes" / EMBEDDING_MODE / "gallery_100"
HNSW_PATH = GALLERY_DIR / "hnsw" / "hnsw_index.bin"
GALLERY_METADATA_PATH = GALLERY_DIR / "exact" / "gallery_metadata.csv"
RESULTS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "results"
    / EMBEDDING_MODE
    / f"{EMBEDDING_MODE}_evaluation.csv"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = get_eval_transform(IMAGE_SIZE)


def check_required_files():
    required = [
        CHECKPOINT_PATH,
        HNSW_PATH,
        GALLERY_METADATA_PATH,
        RESULTS_PATH,
    ]

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required inference files are missing:\n" + "\n".join(missing)
        )


def load_model():
    model = EmbeddingModel(
        normalize_embeddings=bool(MODEL_CFG.get("normalize_embeddings", True))
    )

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


def load_threshold():
    results = pd.read_csv(RESULTS_PATH)

    selected = results[
        (results["method"].str.upper() == "HNSW")
        & np.isclose(results["gallery_fraction"].astype(float), 1.0)
    ]

    if selected.empty:
        raise RuntimeError(
            "Could not find the 100% HNSW threshold in the evaluation results."
        )

    return float(selected.iloc[0]["threshold"])


check_required_files()

MODEL = load_model()
GALLERY_METADATA = pd.read_csv(GALLERY_METADATA_PATH)
THRESHOLD = load_threshold()

HNSW_CFG = RETRIEVAL_CFG["hnsw"]
HNSW_INDEX = load_hnsw(
    HNSW_PATH,
    EMBEDDING_DIM,
    HNSW_CFG.get("space", "cosine"),
)
HNSW_INDEX.set_ef(int(HNSW_CFG.get("ef_search", 100)))


app = FastAPI(title="Wildlife Re-ID API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embedding_mode": EMBEDDING_MODE,
        "retrieval_method": "hnsw",
        "gallery_size": int(len(GALLERY_METADATA)),
        "device": str(DEVICE),
    }


@app.post("/reidentify")
async def reidentify(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    if top_k < 1 or top_k > 5:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 5.")

    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    start = time.perf_counter()

    image_tensor = TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        query_embedding = MODEL(image_tensor)[0].cpu().numpy().astype(np.float32)

    indices, similarities, _ = hnsw_search(
        HNSW_INDEX,
        query_embedding,
        top_k,
    )

    matches = []
    for rank, (gallery_index, similarity) in enumerate(
        zip(indices, similarities),
        start=1,
    ):
        row = GALLERY_METADATA.iloc[int(gallery_index)]
        matches.append(
            {
                "rank": rank,
                "identity": str(row["identity"]),
                "imageId": f"gallery_{int(gallery_index)}",
                "similarity": float(similarity),
                "imageUrl": None,
            }
        )

    best_similarity = float(similarities[0]) if len(similarities) else None
    is_known = best_similarity is not None and best_similarity >= THRESHOLD
    predicted_identity = matches[0]["identity"] if is_known and matches else None

    latency_ms = (time.perf_counter() - start) * 1000.0

    return {
        "decision": "KNOWN" if is_known else "UNKNOWN",
        "predictedIdentity": predicted_identity,
        "similarity": best_similarity,
        "threshold": THRESHOLD,
        "matches": matches,
        "inference": {
            "embeddingMode": EMBEDDING_MODE,
            "modelVersion": CHECKPOINT_PATH.name,
            "retrievalMethod": "hnsw",
            "galleryVersion": "gallery_100",
            "thresholdVersion": RESULTS_PATH.name,
        },
        "latencyMs": latency_ms,
        "isDemo": False,
    }
