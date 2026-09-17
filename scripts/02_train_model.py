# Fine-tunes ResNet-50 using triplet loss

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import load_dataset, get_transform, TripletDataset
from src.models.embedding_model import EmbeddingModel


SEED = 12345

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.0001
MARGIN = 0.3

# Limit the number of training batches per epoch.
# This makes training practical on Colab while still
# sampling new triplets randomly.
MAX_TRAIN_BATCHES = 2000

# Validation uses the complete validation dataset.
MAX_VALIDATION_BATCHES = None

# Number of CPU workers used to load images.
NUM_WORKERS = 2


def train_epoch(
    model,
    data_loader,
    loss_function,
    optimizer,
    device,
    scaler,
    max_batches=None
):
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    batches_processed = 0

    for batch_number, (anchor, positive, negative) in enumerate(
        data_loader,
        start=1
    ):

        # Stop after the configured number of batches.
        if max_batches is not None and batch_number > max_batches:
            break

        anchor = anchor.to(device, non_blocking=True)
        positive = positive.to(device, non_blocking=True)
        negative = negative.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Use mixed precision on CUDA.
        if device.type == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                anchor_embedding = model(anchor)
                positive_embedding = model(positive)
                negative_embedding = model(negative)

                loss = loss_function(
                    anchor_embedding,
                    positive_embedding,
                    negative_embedding
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            loss = loss_function(
                anchor_embedding,
                positive_embedding,
                negative_embedding
            )

            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        batches_processed += 1

        if batch_number % 250 == 0:
            print(
                "Training batch:",
                batch_number,
                "/",
                max_batches if max_batches else len(data_loader),
                "- Loss:",
                round(loss.item(), 4)
            )

    if batches_processed == 0:
        return 0.0

    return total_loss / batches_processed


def validate_model(
    model,
    data_loader,
    loss_function,
    device,
    max_batches=None
):
    """Validate the model."""

    model.eval()

    total_loss = 0.0
    batches_processed = 0

    with torch.no_grad():

        for batch_number, (anchor, positive, negative) in enumerate(
            data_loader,
            start=1
        ):

            if max_batches is not None and batch_number > max_batches:
                break

            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):
                    anchor_embedding = model(anchor)
                    positive_embedding = model(positive)
                    negative_embedding = model(negative)

                    loss = loss_function(
                        anchor_embedding,
                        positive_embedding,
                        negative_embedding
                    )

            else:
                anchor_embedding = model(anchor)
                positive_embedding = model(positive)
                negative_embedding = model(negative)

                loss = loss_function(
                    anchor_embedding,
                    positive_embedding,
                    negative_embedding
                )

            total_loss += loss.item()
            batches_processed += 1

    if batches_processed == 0:
        return 0.0

    return total_loss / batches_processed


def main():

    # Reproducibility.
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True

    # Paths.
    train_file = Path("data/processed/train.csv")
    validation_file = Path("data/processed/validation.csv")

    dataset_dir = Path("data/raw")
    checkpoint_dir = Path("artifacts/checkpoints")

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_file = checkpoint_dir / "training_checkpoint.pth"
    best_model_file = checkpoint_dir / "best_resnet50.pth"

    # Select device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print()
    print("========================================")
    print("ResNet-50 Fine-Tuning")
    print("========================================")

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )

    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Triplet margin:", MARGIN)
    print("Maximum training batches:", MAX_TRAIN_BATCHES)

    # ----------------------------------------
    # Load data
    # ----------------------------------------

    print()
    print("Loading datasets...")

    train_records = load_dataset(
        train_file,
        dataset_dir,
        "train"
    )

    validation_records = load_dataset(
        validation_file,
        dataset_dir,
        "validation"
    )

    print("Train records:", len(train_records))
    print("Validation records:", len(validation_records))

    # ----------------------------------------
    # Create datasets
    # ----------------------------------------

    transform = get_transform()

    train_dataset = TripletDataset(
        train_records,
        transform
    )

    validation_dataset = TripletDataset(
        validation_records,
        transform
    )

    print("Train triplets:", len(train_dataset))
    print("Validation triplets:", len(validation_dataset))

    # ----------------------------------------
    # Create data loaders
    # ----------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    # ----------------------------------------
    # Create model
    # ----------------------------------------

    print()
    print("Loading ImageNet-pretrained ResNet-50...")

    model = EmbeddingModel().to(device)

    print("Model loaded.")
    print("Embedding dimension: 2048")

    # ----------------------------------------
    # Loss and optimizer
    # ----------------------------------------

    loss_function = nn.TripletMarginLoss(
        margin=MARGIN,
        p=2
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Mixed-precision gradient scaler.
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    # ----------------------------------------
    # Resume checkpoint
    # ----------------------------------------

    start_epoch = 0
    best_validation_loss = float("inf")

    if checkpoint_file.exists():

        print()
        print("Checkpoint found.")
        print("Loading previous training state...")

        checkpoint = torch.load(
            checkpoint_file,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model_state"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

        start_epoch = checkpoint["epoch"]

        best_validation_loss = checkpoint[
            "best_validation_loss"
        ]

        print(
            "Resuming from epoch:",
            start_epoch + 1
        )

        print(
            "Previous best validation loss:",
            round(best_validation_loss, 4)
        )

    # ----------------------------------------
    # Training
    # ----------------------------------------

    print()
    print("========================================")
    print("Training started")
    print("========================================")

    for epoch in range(start_epoch, EPOCHS):

        print()
        print(
            "Epoch:",
            epoch + 1,
            "/",
            EPOCHS
        )

        # Train.
        train_loss = train_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            max_batches=MAX_TRAIN_BATCHES
        )

        print()
        print("Running validation...")

        # Validate.
        validation_loss = validate_model(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            device=device,
            max_batches=MAX_VALIDATION_BATCHES
        )

        print()
        print("Train loss:", round(train_loss, 4))
        print(
            "Validation loss:",
            round(validation_loss, 4)
        )

        # ----------------------------------------
        # Save best model
        # ----------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            torch.save(
                model.state_dict(),
                best_model_file
            )

            print()
            print("Best model saved:")
            print(best_model_file)

        # ----------------------------------------
        # Save checkpoint
        # ----------------------------------------

        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "train_loss": train_loss,
            "validation_loss": validation_loss
        }

        torch.save(
            checkpoint,
            checkpoint_file
        )

        print("Training checkpoint saved.")

    # ----------------------------------------
    # Complete
    # ----------------------------------------

    print()
    print("========================================")
    print("Training complete")
    print("========================================")

    print(
        "Best validation loss:",
        round(best_validation_loss, 4)
    )

    print(
        "Best model:",
        best_model_file
    )

    print(
        "Checkpoint:",
        checkpoint_file
    )


if __name__ == "__main__":
    main()

