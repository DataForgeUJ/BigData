"""Refine the current ResNet-50 using same-species batch-hard triplet loss."""

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Sampler

from src.data.dataset import (
    WildlifeDataset,
    get_eval_transform,
    get_train_transform,
    load_dataset,
)
from src.models.embedding_model import EmbeddingModel


BASE_CONFIG_PATH = Path("configs/base.yaml")
EXPERIMENT_CONFIG_PATH = Path("configs/species_batch_hard.yaml")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class SameSpeciesBatchSampler(Sampler):
    def __init__(
        self,
        records,
        identities_per_batch,
        images_per_identity,
        batches_per_epoch,
        seed,
        shuffle=True,
    ):
        self.identities_per_batch = identities_per_batch
        self.images_per_identity = images_per_identity
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

        species_identities = {}

        for index, record in enumerate(records):
            species = record["species"]
            identity = record["identity"]

            if species not in species_identities:
                species_identities[species] = {}

            if identity not in species_identities[species]:
                species_identities[species][identity] = []

            species_identities[species][identity].append(index)

        self.species_identities = {}

        for species, identities in species_identities.items():
            valid_identities = {}

            for identity, indexes in identities.items():
                if len(indexes) >= images_per_identity:
                    valid_identities[identity] = indexes

            if len(valid_identities) >= identities_per_batch:
                self.species_identities[species] = valid_identities

        self.valid_species = list(self.species_identities.keys())

        if not self.valid_species:
            raise ValueError(
                "No species has enough identities and images for the requested batch."
            )

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
        else:
            rng = random.Random(self.seed)

        for _ in range(self.batches_per_epoch):
            species = rng.choice(self.valid_species)

            identity_names = list(self.species_identities[species].keys())
            chosen_identities = rng.sample(
                identity_names,
                self.identities_per_batch,
            )

            batch_indexes = []

            for identity in chosen_identities:
                image_indexes = self.species_identities[species][identity]
                chosen_images = rng.sample(
                    image_indexes,
                    self.images_per_identity,
                )
                batch_indexes.extend(chosen_images)

            if self.shuffle:
                rng.shuffle(batch_indexes)

            yield batch_indexes

        if self.shuffle:
            self.epoch += 1


def make_labels(identity_names, device):
    identity_to_number = {}
    labels = []

    for identity in identity_names:
        if identity not in identity_to_number:
            identity_to_number[identity] = len(identity_to_number)

        labels.append(identity_to_number[identity])

    return torch.tensor(labels, dtype=torch.long, device=device)


def batch_hard_loss(embeddings, labels, margin):
    distances = torch.cdist(
        embeddings.float(),
        embeddings.float(),
        p=2,
    )

    same_identity = labels.unsqueeze(0) == labels.unsqueeze(1)
    diagonal = torch.eye(
        len(labels),
        dtype=torch.bool,
        device=labels.device,
    )

    positive_mask = same_identity & ~diagonal
    negative_mask = ~same_identity

    hardest_positive = distances.masked_fill(
        ~positive_mask,
        -1.0,
    ).max(dim=1).values

    hardest_negative = distances.masked_fill(
        ~negative_mask,
        float("inf"),
    ).min(dim=1).values

    losses = F.relu(
        hardest_positive - hardest_negative + margin
    )

    active_triplets = (losses > 0).float().mean().item()

    return losses.mean(), active_triplets, distances


def batch_top1(distances, labels):
    distances = distances.clone()

    diagonal = torch.eye(
        len(labels),
        dtype=torch.bool,
        device=labels.device,
    )

    distances[diagonal] = float("inf")
    nearest = distances.argmin(dim=1)

    correct = labels[nearest] == labels
    return correct.float().mean().item()


def train_epoch(model, loader, optimizer, device, scaler, margin):
    model.train()

    total_loss = 0.0
    total_active = 0.0
    batches = 0

    for batch_number, batch in enumerate(loader, 1):
        images = batch["image"].to(device, non_blocking=True)
        labels = make_labels(batch["identity"], device)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                embeddings = model(images)
                loss, active_triplets, _ = batch_hard_loss(
                    embeddings,
                    labels,
                    margin,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings = model(images)
            loss, active_triplets, _ = batch_hard_loss(
                embeddings,
                labels,
                margin,
            )
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_active += active_triplets
        batches += 1

        if batch_number % 100 == 0:
            print(
                f"  Batch: {batch_number}/{len(loader)}"
                f" | Loss: {loss.item():.4f}"
                f" | Active: {active_triplets * 100:.1f}%"
            )

    return (
        total_loss / batches,
        total_active / batches,
    )


def validate_model(model, loader, device, margin):
    model.eval()

    total_loss = 0.0
    total_active = 0.0
    total_top1 = 0.0
    batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = make_labels(batch["identity"], device)

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    embeddings = model(images)
                    loss, active_triplets, distances = batch_hard_loss(
                        embeddings,
                        labels,
                        margin,
                    )
            else:
                embeddings = model(images)
                loss, active_triplets, distances = batch_hard_loss(
                    embeddings,
                    labels,
                    margin,
                )

            top1 = batch_top1(distances, labels)

            total_loss += loss.item()
            total_active += active_triplets
            total_top1 += top1
            batches += 1

    return (
        total_loss / batches,
        total_active / batches,
        total_top1 / batches,
    )


def save_history(history_file, rows):
    fieldnames = [
        "epoch",
        "train_loss",
        "validation_loss",
        "train_active_triplets",
        "validation_active_triplets",
        "validation_same_species_top1",
    ]

    with open(history_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    base_config = load_yaml(BASE_CONFIG_PATH)
    experiment = load_yaml(EXPERIMENT_CONFIG_PATH)

    data_cfg = base_config["data"]

    seed = base_config["seed"]
    identities_per_batch = experiment["identities_per_batch"]
    images_per_identity = experiment["images_per_identity"]
    batches_per_epoch = experiment["batches_per_epoch"]
    validation_batches = experiment["validation_batches"]
    epochs = experiment["epochs"]
    learning_rate = experiment["learning_rate"]
    margin = experiment["margin"]
    num_workers = experiment["num_workers"]

    batch_size = identities_per_batch * images_per_identity

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    splits_dir = Path(data_cfg["splits_dir"])
    dataset_dir = Path(data_cfg["root"])

    train_records = load_dataset(
        splits_dir / "train.csv",
        dataset_dir,
        "train",
    )

    validation_records = load_dataset(
        splits_dir / "validation.csv",
        dataset_dir,
        "validation",
    )

    output_dir = Path(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_file = output_dir / "best_species_batch_hard.pth"
    checkpoint_file = output_dir / "training_checkpoint.pth"
    history_file = output_dir / "training_history.csv"

    train_dataset = WildlifeDataset(
        train_records,
        get_train_transform(data_cfg["image_size"]),
    )

    validation_dataset = WildlifeDataset(
        validation_records,
        get_eval_transform(data_cfg["image_size"]),
    )

    train_sampler = SameSpeciesBatchSampler(
        train_records,
        identities_per_batch,
        images_per_identity,
        batches_per_epoch,
        seed,
        shuffle=True,
    )

    validation_sampler = SameSpeciesBatchSampler(
        validation_records,
        identities_per_batch,
        images_per_identity,
        validation_batches,
        seed,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("\n" + "=" * 60)
    print("SAME-SPECIES BATCH-HARD REFINEMENT")
    print("=" * 60)
    print(f"Device:                  {device}")
    print(f"Train records:           {len(train_records):,}")
    print(f"Validation records:      {len(validation_records):,}")
    print(f"Identities per batch:    {identities_per_batch}")
    print(f"Images per identity:     {images_per_identity}")
    print(f"Batch size:              {batch_size}")
    print(f"Train species available: {len(train_sampler.valid_species)}")
    print(f"Val species available:   {len(validation_sampler.valid_species)}")
    print(f"Batches per epoch:       {batches_per_epoch}")
    print(f"Validation batches:      {validation_batches}")
    print(f"Epochs:                  {epochs}")
    print(f"Learning rate:           {learning_rate}")
    print(f"Margin:                  {margin}")

    if device.type == "cuda":
        print(f"GPU:                     {torch.cuda.get_device_name(0)}")

    model = EmbeddingModel().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda"
        else None
    )

    start_epoch = 0
    best_validation_top1 = -1.0
    best_validation_loss = float("inf")
    history = []

    if checkpoint_file.exists():
        print("\nResuming same-species training...")

        checkpoint = torch.load(
            checkpoint_file,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        start_epoch = checkpoint["epoch"]
        best_validation_top1 = checkpoint["best_validation_top1"]
        best_validation_loss = checkpoint["best_validation_loss"]
        history = checkpoint.get("history", [])

        print(f"Resuming from epoch {start_epoch + 1}")

    else:
        initial_checkpoint = Path(
            experiment["initial_checkpoint"]
        )

        if not initial_checkpoint.exists():
            raise FileNotFoundError(
                f"Initial checkpoint not found: {initial_checkpoint}"
            )

        print(f"\nLoading initial model: {initial_checkpoint}")

        state_dict = torch.load(
            initial_checkpoint,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(state_dict)
        print("Initial batch-hard model loaded.")

    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_active = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            margin,
        )

        print("\nRunning validation...")

        val_loss, val_active, val_top1 = validate_model(
            model,
            validation_loader,
            device,
            margin,
        )

        print(f"Train loss:                 {train_loss:.4f}")
        print(f"Validation loss:            {val_loss:.4f}")
        print(f"Train active triplets:      {train_active * 100:.1f}%")
        print(f"Validation active triplets: {val_active * 100:.1f}%")
        print(f"Validation same-species T1: {val_top1 * 100:.2f}%")

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_active_triplets": train_active,
            "validation_active_triplets": val_active,
            "validation_same_species_top1": val_top1,
        }

        history.append(row)
        save_history(history_file, history)

        better_top1 = val_top1 > best_validation_top1
        same_top1 = abs(val_top1 - best_validation_top1) < 1e-12
        lower_loss = val_loss < best_validation_loss

        if better_top1 or (same_top1 and lower_loss):
            best_validation_top1 = val_top1
            best_validation_loss = val_loss

            torch.save(
                model.state_dict(),
                best_model_file,
            )

            print(f"Best model saved: {best_model_file}")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_validation_top1": best_validation_top1,
            "best_validation_loss": best_validation_loss,
            "history": history,
        }

        torch.save(checkpoint, checkpoint_file)
        print("Training checkpoint saved.")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(
        f"Best validation same-species Top-1: "
        f"{best_validation_top1 * 100:.2f}%"
    )
    print(f"Best model: {best_model_file}")
    print(f"History:    {history_file}")


if __name__ == "__main__":
    main()
