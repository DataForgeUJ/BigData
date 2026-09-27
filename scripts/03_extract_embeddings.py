"""Extract embeddings using frozen or fine-tuned ResNet-50."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from src.data.dataset import (
    load_dataset,
    get_eval_transform,
    WildlifeDataset
)


# Configuration
BATCH_SIZE = 16

CHECKPOINT_PATH = Path(
    "artifacts/checkpoints/best_resnet50.pth"
)


# Model creation
def create_model(device, mode):
    """Create the ResNet-50 embedding model."""

    print()
    print("Creating ResNet-50...")
    print("Mode:", mode)

    # Frozen ImageNet model
    if mode == "frozen":

        print("Using ImageNet-pretrained weights.")

        model = resnet50(
            weights=ResNet50_Weights.DEFAULT
        )

    # Fine-tuned model
    elif mode == "finetuned":

        if not CHECKPOINT_PATH.exists():

            raise FileNotFoundError(
                "\nFine-tuned checkpoint was not found:\n"
                f"{CHECKPOINT_PATH}\n\n"
                "Make sure best_resnet50.pth exists "
                "before running fine-tuned extraction."
            )

        print(
            "Loading fine-tuned checkpoint:"
        )

        print(CHECKPOINT_PATH)

        model = resnet50(
            weights=None
        )

    else:

        raise ValueError(
            "Mode must be 'frozen' or 'finetuned'."
        )

    # Remove classification layer
    model.fc = torch.nn.Identity()

    # Load fine-tuned weights
    if mode == "finetuned":

        state_dict = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=True
        )

        state_dict = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
        }

        model.load_state_dict(
            state_dict
        )

        print(
            "Fine-tuned weights loaded successfully."
        )

    # Prepare model
    model = model.to(device)

    # Embedding extraction does not require gradients.
    for parameter in model.parameters():

        parameter.requires_grad = False

    model.eval()

    return model


# Embedding extraction
def extract_embeddings(
    model,
    data_loader,
    device
):
    """Extract normalized 2048-dimensional embeddings."""

    model.eval()

    embeddings = []
    metadata = []

    with torch.no_grad():

        for batch_number, batch in enumerate(
            data_loader,
            start=1
        ):

            images = batch["image"].to(
                device,
                non_blocking=True
            )

            # Generate the 2048-dimensional ResNet-50 feature vector.
            batch_embeddings = model(
                images
            )

            # Explicit L2 normalization.
            batch_embeddings = (
                torch.nn.functional.normalize(
                    batch_embeddings,
                    p=2,
                    dim=1
                )
            )

            # Move embeddings from GPU to CPU before converting them to NumPy.
            batch_embeddings = (
                batch_embeddings
                .cpu()
                .numpy()
            )

            embeddings.append(
                batch_embeddings
            )

            # Store metadata in exactly the same order as the embeddings.
            for i in range(
                len(batch["identity"])
            ):

                metadata.append({
                    "identity": batch["identity"][i],
                    "species": batch["species"][i],
                    "dataset": batch["dataset"][i]
                })

            # Progress information.
            if batch_number % 100 == 0:

                print(
                    "Processed batches:",
                    batch_number,
                    "/",
                    len(data_loader)
                )

    # Combine all batches into one matrix.
    embeddings = np.concatenate(
        embeddings,
        axis=0
    )

    return embeddings, metadata


# Save embeddings
def save_embeddings(
    embeddings,
    metadata,
    output_dir,
    name
):
    """Save embeddings and corresponding metadata."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    embedding_file = (
        output_dir
        / f"{name}_embeddings.npy"
    )

    metadata_file = (
        output_dir
        / f"{name}_metadata.csv"
    )

    # Save NumPy embeddings.
    np.save(
        embedding_file,
        embeddings
    )

    # Save corresponding metadata.
    metadata_df = pd.DataFrame(
        metadata
    )

    metadata_df.to_csv(
        metadata_file,
        index=False
    )

    print()
    print("Saved embeddings:")
    print(embedding_file)

    print("Saved metadata:")
    print(metadata_file)

    print("Embedding shape:")
    print(embeddings.shape)


# Command-line arguments
def parse_arguments():
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract ResNet-50 embeddings "
            "using frozen or fine-tuned weights."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "frozen",
            "finetuned"
        ],
        default="finetuned",
        help=(
            "Embedding model to use: "
            "frozen or finetuned."
        )
    )

    return parser.parse_args()


# Main
def main():

    args = parse_arguments()

    mode = args.mode

    # Paths
    train_file = Path(
        "data/processed/train.csv"
    )

    validation_file = Path(
        "data/processed/validation.csv"
    )

    test_file = Path(
        "data/processed/test.csv"
    )

    dataset_dir = Path(
        "data/raw"
    )

    # Keep frozen and fine-tuned embeddings completely separate.
    output_dir = (
        Path("data/embeddings")
        / mode
    )

    # Device
    if torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    # Header
    print()
    print("========================================")
    print("ResNet-50 Embedding Extraction")
    print("========================================")

    print("Mode:", mode)
    print("Device:", device)
    print("Batch size:", BATCH_SIZE)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "Embedding dimension:",
        2048
    )

    print(
        "Output directory:",
        output_dir
    )

    # Create model
    model = create_model(
        device,
        mode
    )

    # Evaluation preprocessing
    transform = get_eval_transform()

    # Process datasets
    datasets = [
        (
            "train",
            train_file
        ),
        (
            "validation",
            validation_file
        ),
        (
            "test",
            test_file
        )
    ]

    for name, metadata_file in datasets:

        print()
        print("----------------------------------------")
        print("Processing:", name)
        print("----------------------------------------")

        # Load image records.
        records = load_dataset(
            metadata_file,
            dataset_dir,
            name
        )

        print(
            "Images:",
            len(records)
        )

        # Create dataset.
        dataset = WildlifeDataset(
            records,
            transform
        )

        # Create data loader.
        data_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(
                device.type == "cuda"
            )
        )

        # Extract embeddings.
        embeddings, metadata = (
            extract_embeddings(
                model,
                data_loader,
                device
            )
        )

        # Save embeddings and metadata.
        save_embeddings(
            embeddings,
            metadata,
            output_dir,
            name
        )

    # Complete
    print()
    print("========================================")
    print("Embedding extraction complete")
    print("========================================")

    print("Mode:", mode)

    print(
        "Output directory:",
        output_dir
    )


if __name__ == "__main__":
    main()
