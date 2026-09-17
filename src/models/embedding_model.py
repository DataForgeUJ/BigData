import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class EmbeddingModel(nn.Module):
    def __init__(self,normalize_embeddings = True):
        super().__init__()

        self.normalize_embeddings = normalize_embeddings
        # Load ResNet-50
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove the classification layer
        self.model.fc = nn.Identity()

    def forward(self, images):
        embeddings = self.model(images)

        # Normalize embeddings for cosine similarity
        if self.normalize_embeddings:
            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1
            )

        return embeddings