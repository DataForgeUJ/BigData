# Prepares the dataset and creates the train, validation and test splits
import random
import pandas as pd
from pathlib import Path


SEED = 12345

# Splits the dataset into training, validation and test sets
def prepare_data():
    # Paths
    metadata_file = Path("data/raw/metadata.csv")
    processed_dir = Path("data/processed")

    # Create the processed folder
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Read the metadata file
    metadata = pd.read_csv(metadata_file, low_memory=False)

    # Get the train and test data
    train_data = metadata[metadata["split"] == "train"].copy()
    test_data = metadata[metadata["split"] == "test"].copy()

    # Count how many images each identity has
    identity_counts = train_data["identity"].value_counts()

    # Get identities that appear in the test data
    test_ids = set(test_data["identity"])

    validation_candidates = []

    for identity in train_data["identity"].unique():

        # Avoid identities that also appear in the test data
        if identity not in test_ids:

            # Identity must have at least two images
            if identity_counts[identity] >= 2:
                validation_candidates.append(identity)

    random.seed(SEED)
    random.shuffle(validation_candidates)

    # Use 20% of the identities for validation
    validation_count = int(len(validation_candidates) * 0.20)
    validation_ids = validation_candidates[:validation_count]

    # Separate validation from training
    validation_data = train_data[
        train_data["identity"].isin(validation_ids)
    ].copy()

    train_data = train_data[
        ~train_data["identity"].isin(validation_ids)
    ].copy()

    # Update the split names
    train_data["split"] = "train"
    validation_data["split"] = "validation"
    test_data["split"] = "test"

    # Save data
    train_data.to_csv(processed_dir / "train.csv", index=False)
    validation_data.to_csv(processed_dir / "validation.csv", index=False)
    test_data.to_csv(processed_dir / "test.csv", index=False)

    # Display results
    print()
    print("---- Data information ----")
    print("Total images:", len(metadata))
    print("Train images:", len(train_data))
    print("Validation images:", len(validation_data))
    print("Test images:", len(test_data))
    print()
    print("---- Identity information ----")
    print("Train identities:", train_data["identity"].nunique())
    print("Validation identities:", validation_data["identity"].nunique())
    print("Test identities:", test_data["identity"].nunique())



# TESTING
def main():
    prepare_data()


if __name__ == "__main__":
    main()