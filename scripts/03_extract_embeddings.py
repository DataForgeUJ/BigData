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

from src.data.dataset import load_dataset, get_transform, WildlifeDataset


BATCH_SIZE = 16

CHECKPOINT_PATH = Path(
    "artifacts/checkpoints/best_resnet50.pth"
)


def create_model(device, mode):
    """Create the ResNet-50 model."""

    print()
    print("Creating ResNet-50...")
    print("Mode:", mode)

    if mode == "frozen":

        print("Using ImageNet-pretrained weights.")

        model = resnet50(
            weights=ResNet50_Weights.DEFAULT
        )

    elif mode == "finetuned":

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                "\nFine-tuned checkpoint was not found:\n"
                f"{CHECKPOINT_PATH}\n\n"
                "Run scripts/02_train_model.py first "
                "to create best_resnet50.pth."
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

    # Remove the classification layer.
    model.fc = torch.nn.Identity()

    # Load fine-tuned weights if required.
    if mode == "finetuned":

        state_dict = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(state_dict)

        print("Fine-tuned weights loaded successfully.")

    model = model.to(device)

    # Embedding extraction does not require gradients.
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model


def extract_embeddings(
    model,
    data_loader,
    device
):
    """Extract 2048-dimensional embeddings."""

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

            # Generate 2048-dimensional embeddings.
            batch_embeddings = model(images)

            # Explicit L2 normalization.
            # This makes cosine similarity equivalent
            # to the dot product.
            batch_embeddings = torch.nn.functional.normalize(
                batch_embeddings,
                p=2,
                dim=1
            )

            batch_embeddings = (
                batch_embeddings
                .cpu()
                .numpy()
            )

            embeddings.append(batch_embeddings)

            # Store metadata.
            for i in range(len(batch["identity"])):

                metadata.append({
                    "identity": batch["identity"][i],
                    "species": batch["species"][i],
                    "dataset": batch["dataset"][i]
                })

            if batch_number % 100 == 0:

                print(
                    "Processed batches:",
                    batch_number,
                    "/",
                    len(data_loader)
                )

    embeddings = np.concatenate(
        embeddings,
        axis=0
    )

    return embeddings, metadata


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
        output_dir /
        f"{name}_embeddings.npy"
    )

    metadata_file = (
        output_dir /
        f"{name}_metadata.csv"
    )

    np.save(
        embedding_file,
        embeddings
    )

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
        default="frozen",
        help=(
            "Embedding model to use: "
            "frozen or finetuned."
        )
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    mode = args.mode

    # ----------------------------------------
    # Paths
    # ----------------------------------------

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

    # Store frozen and fine-tuned embeddings
    # separately.
    output_dir = (
        Path("data/embeddings") /
        mode
    )

    # ----------------------------------------
    # Device
    # ----------------------------------------

    if torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

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

    print("Embedding dimension: 2048")

    # ----------------------------------------
    # Create model
    # ----------------------------------------

    model = create_model(
        device,
        mode
    )

    # ----------------------------------------
    # Image transformation
    # ----------------------------------------

    transform = get_transform()

    # ----------------------------------------
    # Process datasets
    # ----------------------------------------

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

        records = load_dataset(
            metadata_file,
            dataset_dir,
            name
        )

        print(
            "Images:",
            len(records)
        )

        dataset = WildlifeDataset(
            records,
            transform
        )

        data_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda")
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
    print("========================================")
    print("Embedding extraction complete")
    print("========================================")

    print("Mode:", mode)
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()

