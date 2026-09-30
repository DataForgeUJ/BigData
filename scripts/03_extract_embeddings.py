"""Extract embeddings using frozen or fine-tuned ResNet-50."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from src.data.dataset import load_dataset, get_eval_transform, WildlifeDataset


# ============================================================
# Configuration
# ============================================================

CONFIG_PATH = Path("configs/base.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Model
# ============================================================

def create_model(device, mode, model_cfg):
    print(f"\nCreating ResNet-50 | Mode: {mode}")

    if mode == "frozen":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)

    elif mode == "finetuned":
        checkpoint = Path(model_cfg["checkpoint"])

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"\nFine-tuned checkpoint not found:\n{checkpoint}"
            )

        print(f"Loading checkpoint: {checkpoint}")
        model = resnet50(weights=None)

    else:
        raise ValueError("Mode must be 'frozen' or 'finetuned'.")

    model.fc = torch.nn.Identity()

    if mode == "finetuned":
        state_dict = torch.load(
            checkpoint,
            map_location=device,
            weights_only=True,
        )
        state_dict = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict)
        print("Fine-tuned weights loaded.")

    model = model.to(device).eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


# ============================================================
# Extraction
# ============================================================

def extract_embeddings(model, loader, device, normalize=True):
    embeddings, metadata = [], []

    with torch.no_grad():
        for batch_number, batch in enumerate(loader, 1):
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            batch_embeddings = model(images)

            if normalize:
                batch_embeddings = torch.nn.functional.normalize(
                    batch_embeddings,
                    p=2,
                    dim=1,
                )

            embeddings.append(
                batch_embeddings.cpu().numpy()
            )

            metadata.extend(
                {
                    "identity": batch["identity"][i],
                    "species": batch["species"][i],
                    "dataset": batch["dataset"][i],
                }
                for i in range(len(batch["identity"]))
            )

            if batch_number % 100 == 0:
                print(
                    f"  Batches: {batch_number}/{len(loader)}"
                )

    return np.concatenate(embeddings, axis=0), metadata


# ============================================================
# Save
# ============================================================

def save_embeddings(embeddings, metadata, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_file = output_dir / f"{name}_embeddings.npy"
    metadata_file = output_dir / f"{name}_metadata.csv"

    np.save(embedding_file, embeddings)
    pd.DataFrame(metadata).to_csv(
        metadata_file,
        index=False,
    )

    print(f"  Embeddings: {embedding_file}")
    print(f"  Metadata:   {metadata_file}")
    print(f"  Shape:      {embeddings.shape}")


# ============================================================
# Main
# ============================================================

def main():
    config = load_config()

    data_cfg = config["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]

    parser = argparse.ArgumentParser(
        description="Extract ResNet-50 embeddings."
    )
    parser.add_argument(
        "--mode",
        choices=["frozen", "finetuned"],
        default=data_cfg.get("embedding_mode", "finetuned"),
    )
    args = parser.parse_args()

    mode = args.mode
    batch_size = training_cfg["batch_size"]
    embedding_dim = model_cfg["embedding_dim"]
    normalize = model_cfg.get("normalize_embeddings", True)

    data_root = Path(data_cfg["root"])
    processed_dir = Path(data_cfg["processed_dir"])
    output_dir = Path(data_cfg["embeddings_dir"]) / mode

    datasets = [
        ("train", processed_dir / "train.csv"),
        ("validation", processed_dir / "validation.csv"),
        ("test", processed_dir / "test.csv"),
    ]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n" + "=" * 55)
    print("RESNET-50 EMBEDDING EXTRACTION")
    print("=" * 55)
    print(f"Mode:        {mode}")
    print(f"Device:      {device}")
    print(f"Batch size:  {batch_size}")
    print(f"Dimension:   {embedding_dim}")
    print(f"Normalize:   {normalize}")
    print(f"Output:      {output_dir}")

    if device.type == "cuda":
        print(f"GPU:         {torch.cuda.get_device_name(0)}")

    model = create_model(
        device,
        mode,
        model_cfg,
    )

    transform = get_eval_transform()

    for name, metadata_file in datasets:
        print(f"\n{'-' * 55}\nProcessing: {name}\n{'-' * 55}")

        records = load_dataset(
            metadata_file,
            data_root,
            name,
        )

        print(f"Images: {len(records):,}")

        dataset = WildlifeDataset(
            records,
            transform,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        embeddings, metadata = extract_embeddings(
            model,
            loader,
            device,
            normalize,
        )

        if embeddings.shape[1] != embedding_dim:
            raise ValueError(
                f"Expected {embedding_dim}-D embeddings, "
                f"got {embeddings.shape[1]}-D."
            )

        save_embeddings(
            embeddings,
            metadata,
            output_dir,
            name,
        )

    print("\n" + "=" * 55)
    print("EMBEDDING EXTRACTION COMPLETE")
    print("=" * 55)
    print(f"Mode:   {mode}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()