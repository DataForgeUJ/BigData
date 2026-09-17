# Prepares the dataset and creates the train, validation and test splits
import random
import yaml
import pandas as pd
from pathlib import Path


config_path = Path("configs/base.yaml")

def load_config():
    with open(config_path,"r",encoding ="utf-8") as file:
        return yaml.safe_load(file)

# Splits the dataset into training, validation and test sets
def prepare_data():

    config = load_config()
    seed = config["seed"]
    # Paths
    metadata_file = Path(config["data"]["metadata"])

    splits_dir = Path(
        config["data"]["splits_dir"]
    )

    # Create the splits folder
    splits_dir.mkdir(parents=True, exist_ok=True)

    #checks that metadata exists
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}"
        )

    # Read the metadata file
    metadata = pd.read_csv(metadata_file, low_memory=False)

    #validate metadata
    required_columns = {
        "path",
        "identity",
        "species",
        "dataset",
        "split"
    }
    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing_columns)}"
        )

    print()
    print("Original metadata information")
    print("Metadata columns:", metadata.columns.tolist())
    print()
    print("Available splits:")
    print(metadata["split"].value_counts(dropna=False))


    # Get the train and test data
    train_data = metadata[metadata["split"] == "train"].copy()
    test_data = metadata[metadata["split"] == "test"].copy()

    if train_data.empty:
        raise ValueError(
            "No rows with split = 'train' were found."
        )
    if test_data.empty:
        raise ValueError(
            "No rows with split = 'test' were found."
        )
    # Count how many images each identity has
    identity_counts = train_data["identity"].value_counts()

    # Get identities that appear in the test data
    test_ids = set(test_data["identity"])

    #create validation candidtes
    # TODO:
    # This currently creates a custom identity-level
    # validation split.
    #
    # Before final experiments, verify that this is
    # compatible with the official WildlifeReID-10k
    # evaluation protocol.

    validation_candidates = []

    for identity in train_data["identity"].unique():

        # Avoid identities that also appear in the test data
        if identity not in test_ids:

            # Identity must have at least two images
            if identity_counts[identity] >= 2:
                validation_candidates.append(identity)

    random.seed(seed)
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

    #check that training and validation do not overlap (sanity check)
    train_ids = set(train_data["identity"])
    validation_ids_set = set(validation_data["identity"])

    train_validation_overlap = (
        train_ids.intersection(validation_ids_set)
    )

    if train_validation_overlap:
        raise ValueError(
            "training and validation identities overlap"
        )

    validation_test_overlap =(
        validation_ids_set.intersection(test_ids)
    )

    # Update the split names
    train_data["split"] = "train"
    validation_data["split"] = "validation"
    test_data["split"] = "test"

    # Save data
    train_data.to_csv(splits_dir / "train.csv", index=False)
    validation_data.to_csv(splits_dir / "validation.csv", index=False)
    test_data.to_csv(splits_dir / "test.csv", index=False)

    # Display results
    print()
    print("---- Data information ----")
    print("Total images:", len(metadata))
    print("Train images:", len(train_data))
    print("Validation images:", len(validation_data))
    print("Test images:", len(test_data))
    print()
    print(" Identity information ")
    print("Train identities:", train_data["identity"].nunique())
    print("Validation identities:", validation_data["identity"].nunique())
    print("Test identities:", test_data["identity"].nunique())

    print()
    print("Split checks")

    print(
        "Train/validation identity overlap:",
        len(train_validation_overlap)
    )

    print(
        "Validation/test identity overlap:",
        len(validation_test_overlap)
    )

    print()
    print(
        f"Split files saved to: {splits_dir}"
    )

# TESTING
def main():
    prepare_data()


if __name__ == "__main__":
    main()