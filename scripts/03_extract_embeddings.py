"""Extract 2048-D embeddings using a frozen ImageNet-pretrained ResNet-50."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from src.data.dataset import load_dataset, get_transform, WildlifeDataset


BATCH_SIZE = 16


def create_model(device):
    """Create a frozen ImageNet-pretrained ResNet-50."""

    print("Loading ImageNet-pretrained ResNet-50...")

    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Remove the classification layer.
    model.fc = torch.nn.Identity()

    model = model.to(device)

    # Freeze all model parameters.
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model


def extract_embeddings(model, data_loader, device):
    """Extract embeddings for all images in a DataLoader."""

    model.eval()

    embeddings = []
    metadata = []

    with torch.no_grad():
        for batch_number, batch in enumerate(data_loader, start=1):

            images = batch["image"].to(device)

            # Generate 2048-D embeddings.
            batch_embeddings = model(images)

            # Move embeddings to CPU.
            batch_embeddings = batch_embeddings.cpu().numpy()

            embeddings.append(batch_embeddings)

            # Store metadata.
            for i in range(len(batch["identity"])):
                metadata.append({
                    "identity": batch["identity"][i],
                    "species": batch["species"][i],
                    "dataset": batch["dataset"][i]
                })

            if batch_number % 100 == 0:
                print("Processed batches:", batch_number)

    embeddings = np.concatenate(embeddings, axis=0)

    return embeddings, metadata


def save_embeddings(embeddings, metadata, output_dir, name):
    """Save embeddings and metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_file = output_dir / f"{name}_embeddings.npy"
    metadata_file = output_dir / f"{name}_metadata.csv"

    np.save(embedding_file, embeddings)

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(metadata_file, index=False)

    print("Saved:", embedding_file)
    print("Saved:", metadata_file)
    print("Shape:", embeddings.shape)


def main():

    train_file = Path("data/processed/train.csv")
    validation_file = Path("data/processed/validation.csv")
    test_file = Path("data/processed/test.csv")

    dataset_dir = Path("data/raw")
    output_dir = Path("data/embeddings")

    # Use GPU if available, otherwise CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print()
    print("---- Embedding Extraction ----")
    print("Device:", device)
    print("Batch size:", BATCH_SIZE)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load frozen ResNet-50.
    model = create_model(device)

    print("Embedding dimension: 2048")

    transform = get_transform()

    datasets = [
        ("train", train_file),
        ("validation", validation_file),
        ("test", test_file)
    ]

    for name, metadata_file in datasets:

        print()
        print("------------------------------")
        print("Processing:", name)
        print("------------------------------")

        records = load_dataset(
            metadata_file,
            dataset_dir,
            name
        )

        print("Images:", len(records))

        dataset = WildlifeDataset(
            records,
            transform
        )

        data_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        embeddings, metadata = extract_embeddings(
            model,
            data_loader,
            device
        )

        save_embeddings(
            embeddings,
            metadata,
            output_dir,
            name
        )

    print()
    print("---- Embedding extraction complete ----")


if __name__ == "__main__":
    main()