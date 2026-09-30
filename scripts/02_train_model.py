"""Fine-tune ResNet-50 using triplet loss."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import load_dataset, get_transform, TripletDataset
from src.models.embedding_model import EmbeddingModel


CONFIG_PATH = Path("configs/base.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, loss_fn, optimizer, device, scaler, max_batches=None):
    model.train()
    total_loss = 0.0
    batches = 0

    for batch_number, (anchor, positive, negative) in enumerate(loader, 1):
        if max_batches is not None and batch_number > max_batches:
            break

        anchor = anchor.to(device, non_blocking=True)
        positive = positive.to(device, non_blocking=True)
        negative = negative.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = loss_fn(model(anchor), model(positive), model(negative))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model(anchor), model(positive), model(negative))
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        batches += 1

        if batch_number % 250 == 0:
            limit = max_batches if max_batches else len(loader)
            print(f"  Batch: {batch_number}/{limit} | Loss: {loss.item():.4f}")

    return total_loss / batches if batches else 0.0


def validate_model(model, loader, loss_fn, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for batch_number, (anchor, positive, negative) in enumerate(loader, 1):
            if max_batches is not None and batch_number > max_batches:
                break

            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = loss_fn(model(anchor), model(positive), model(negative))
            else:
                loss = loss_fn(model(anchor), model(positive), model(negative))

            total_loss += loss.item()
            batches += 1

    return total_loss / batches if batches else 0.0


def main():
    config = load_config()
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    seed = config["seed"]
    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]
    learning_rate = train_cfg["learning_rate"]
    margin = train_cfg["margin"]

    max_train_batches = train_cfg.get("max_train_batches", 2000)
    max_validation_batches = train_cfg.get("max_validation_batches")
    num_workers = train_cfg.get("num_workers", 2)

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    train_file = Path(data_cfg["processed_dir"]) / "train.csv"
    validation_file = Path(data_cfg["processed_dir"]) / "validation.csv"
    dataset_dir = Path(data_cfg["root"])

    checkpoint_dir = Path("artifacts/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / "training_checkpoint.pth"
    best_model_file = Path(model_cfg.get(
        "checkpoint",
        checkpoint_dir / "best_resnet50.pth"
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 55)
    print("RESNET-50 FINE-TUNING")
    print("=" * 55)
    print(f"Device:          {device}")
    print(f"Seed:            {seed}")
    print(f"Batch size:      {batch_size}")
    print(f"Epochs:          {epochs}")
    print(f"Learning rate:   {learning_rate}")
    print(f"Triplet margin:  {margin}")
    print(f"Max train:       {max_train_batches}")
    print(f"Workers:         {num_workers}")

    if device.type == "cuda":
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory:      "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    print("\nLoading datasets...")

    train_records = load_dataset(train_file, dataset_dir, "train")
    validation_records = load_dataset(
        validation_file, dataset_dir, "validation"
    )

    print(f"Train records:       {len(train_records):,}")
    print(f"Validation records:  {len(validation_records):,}")

    transform = get_transform()

    train_dataset = TripletDataset(train_records, transform)
    validation_dataset = TripletDataset(validation_records, transform)

    print(f"Train triplets:      {len(train_dataset):,}")
    print(f"Validation triplets: {len(validation_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("\nLoading ImageNet-pretrained ResNet-50...")

    model = EmbeddingModel().to(device)

    print("Model loaded.")
    print(f"Embedding dimension: {model_cfg['embedding_dim']}")

    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 0
    best_validation_loss = float("inf")

    if checkpoint_file.exists():
        print("\nCheckpoint found. Resuming training...")

        checkpoint = torch.load(
            checkpoint_file,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        start_epoch = checkpoint["epoch"]
        best_validation_loss = checkpoint["best_validation_loss"]

        print(f"Resuming from epoch: {start_epoch + 1}")
        print(f"Previous best validation loss: {best_validation_loss:.4f}")

    print("\n" + "=" * 55)
    print("TRAINING STARTED")
    print("=" * 55)

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch: {epoch + 1}/{epochs}")

        train_loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scaler,
            max_train_batches,
        )

        print("\nRunning validation...")

        validation_loss = validate_model(
            model,
            validation_loader,
            loss_fn,
            device,
            max_validation_batches,
        )

        print(f"\nTrain loss:       {train_loss:.4f}")
        print(f"Validation loss:  {validation_loss:.4f}")

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), best_model_file)
            print(f"Best model saved: {best_model_file}")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        }

        torch.save(checkpoint, checkpoint_file)
        print("Training checkpoint saved.")

    print("\n" + "=" * 55)
    print("TRAINING COMPLETE")
    print("=" * 55)
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Best model:          {best_model_file}")
    print(f"Checkpoint:          {checkpoint_file}")


if __name__ == "__main__":
    main()