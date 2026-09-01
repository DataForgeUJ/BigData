"""WildlifeReID-10k dataset loading and official split handling."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class WildlifeReIDDataset(Dataset):
    def __init__(self, dataframe, image_root, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = self.image_root / row["image_path"] #illustrative name
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "identity": row["identity"],#illustrative name
            "image_path": str(image_path),
        }