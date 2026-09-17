import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Reading the metadata file and creating image records
def load_dataset(file_name, dataset_dir, split):
    metadata = pd.read_csv(file_name, low_memory=False)

    # Filter metadata based on split
    split_data = metadata[metadata["split"] == split]

    records = []

    for _, row in split_data.iterrows():
        image_path = dataset_dir / row["path"]

        if image_path.exists():
            records.append({
                "path": image_path,
                "identity": row["identity"],
                "species": row["species"],
                "dataset": row["dataset"]
            })

    return records


# Image preprocessing for ResNet-50
#separate training and evaluation transforms
def get_train_transform(image_size = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale =(0.8,1.0)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness = 0.2,
            contrast =0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_eval_transform(image_size = 224):
    resize_size = int(image_size *232/224)

    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]           
        )
    ])



# Dataset used for image loading
class WildlifeDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image = Image.open(record["path"]).convert("RGB")

        # Apply image preprocessing
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "identity": record["identity"],
            "species": record["species"],
            "dataset": record["dataset"]
        }


# Dataset used for loss training
class TripletDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

        # Store image indexes 
        self.identity_images = {}
        for index, record in enumerate(records):
            identity = record["identity"]

            if identity not in self.identity_images:
                self.identity_images[identity] = []

            self.identity_images[identity].append(index)


        # Find identities with at least two images
        self.anchor_identities = []
        for identity in self.identity_images:
            if len(self.identity_images[identity]) >= 2:
                self.anchor_identities.append(identity)


        # Find images that can be used as anchors
        self.anchor_indexes = []
        for index, record in enumerate(records):
            if record["identity"] in self.anchor_identities:
                self.anchor_indexes.append(index)

        # Use all identities for negative sampling
        self.identities = list(self.identity_images.keys())

        if len(self.identities)< 2:
            raise ValueError(
                "TripletDataset requires at least two identities."
            )
        if len(self.anchor_indexes) ==0:
            raise ValueError(
                "TripletDataset requires at least one identity"
                "with two or more images."
            )

    def __len__(self):
        return len(self.anchor_indexes)

    def __getitem__(self, index):
        # Get the anchor image
        anchor_index = self.anchor_indexes[index]
        anchor_record = self.records[anchor_index]
        anchor_identity = anchor_record["identity"]

        # Get another image of the same identity
        positive_indexes = self.identity_images[anchor_identity].copy()
        positive_indexes.remove(anchor_index)

        positive_index = random.choice(positive_indexes)
        positive_record = self.records[positive_index]

        # Get an image from a different identity
        negative_identity = random.choice(self.identities)

        while negative_identity == anchor_identity:
            negative_identity = random.choice(self.identities)

        negative_index = random.choice(self.identity_images[negative_identity])
        negative_record = self.records[negative_index]

        # Open images
        anchor_image = Image.open(anchor_record["path"]).convert("RGB")
        positive_image = Image.open(positive_record["path"]).convert("RGB")
        negative_image = Image.open(negative_record["path"]).convert("RGB")

        # Apply image preprocessing
        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image