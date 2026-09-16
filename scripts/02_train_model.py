# Fine-tunes ResNet-50 using triplet loss
import random
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from src.data.dataset import load_dataset, get_transform, TripletDataset
from src.models.embedding_model import EmbeddingModel


SEED = 12345
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.0001
MARGIN = 0.3


# Trains the model for one epoch
def train_epoch(model, data_loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0

    for anchor, positive, negative in data_loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()       # Clear gradients

        # Create embeddings
        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)

        loss = loss_function(anchor_embedding, positive_embedding, negative_embedding)

        # Update the model
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)

    return average_loss


# Validates the model using the validation dataset
def validate_model(model, data_loader, loss_function, device):
    model.eval()
    total_loss = 0

    # Use the same validation triplets each time
    random.seed(SEED)

    with torch.no_grad():
        for anchor, positive, negative in data_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # Create embeddings
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            loss = loss_function(anchor_embedding, positive_embedding, negative_embedding)
            total_loss += loss.item()

    average_loss = total_loss / len(data_loader)

    return average_loss


# TESTING
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Paths
    train_file = Path("data/processed/train.csv")
    validation_file = Path("data/processed/validation.csv")
    dataset_dir = Path("data/raw")
    checkpoint_dir = Path("artifacts/checkpoints")

    # Create checkpoint folder
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / "training_checkpoint.pth"
    best_model_file = checkpoint_dir / "best_resnet50.pth"

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print()
    print("---- Training information ----")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load the data
    train_records = load_dataset(train_file, dataset_dir, "train")
    validation_records = load_dataset(validation_file, dataset_dir, "validation")

    print("Train records:", len(train_records))
    print("Validation records:", len(validation_records))

    # Image preprocessing
    transform = get_transform()

    # Create triplet datasets
    train_dataset = TripletDataset(train_records, transform)
    validation_dataset = TripletDataset(validation_records, transform)

    print("Train triplets:", len(train_dataset))
    print("Validation triplets:", len(validation_dataset))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create the model
    model = EmbeddingModel().to(device)

    # Training settings
    loss_function = nn.TripletMarginLoss(margin=MARGIN, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 0
    best_validation_loss = float("inf")

    # Continue previous training
    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        start_epoch = checkpoint["epoch"]
        best_validation_loss = checkpoint["best_validation_loss"]

        print()
        print("Checkpoint found")
        print("Continuing from epoch:", start_epoch + 1)

    print()
    print("---- Training started ----")

    for epoch in range(start_epoch, EPOCHS):
        print()
        print("epoch:", epoch + 1, "/", EPOCHS)

        # Train and validate
        train_loss = train_epoch(model, train_loader, loss_function, optimizer, device)

        print("Running validation...")

        validation_loss = validate_model(model, validation_loader, loss_function, device)

        print("Train loss:", round(train_loss, 4))
        print("Validation loss:", round(validation_loss, 4))

        # Save the best model
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), best_model_file)

            print("Best model saved")

        # Save training progress
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "train_loss": train_loss,
            "validation_loss": validation_loss
        }

        torch.save(checkpoint, checkpoint_file)

        print("Training checkpoint saved")

    print()
    print("---- Training complete ----")
    print("Best validation loss:", round(best_validation_loss, 4))


if __name__ == "__main__":
    main()