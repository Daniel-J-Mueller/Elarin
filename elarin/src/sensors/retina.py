import torch
from typing import Iterable
from PIL.Image import Image
from transformers import CLIPProcessor, CLIPModel

class Retina:
    """Camera frame encoder using the vision portion of CLIP.

    Raw images are mapped to 512-dimensional embeddings. The text branch and
    classification head of CLIP are ignored so pixel data is immediately
    discarded after encoding.
    """

    def __init__(self, model_dir: str):
        self.processor = CLIPProcessor.from_pretrained(model_dir)
        self.model = CLIPModel.from_pretrained(model_dir)
        self.model.eval()

    @torch.no_grad()
    def encode(self, images: Iterable[Image]) -> torch.Tensor:
        """Return image embeddings for the provided ``images``."""
        inputs = self.processor(images=list(images), return_tensors="pt")
        features = self.model.get_image_features(**inputs)
        return features

if __name__ == "__main__":
    from PIL import Image
    retina = Retina("../../models/clip-vit-b32")
    img = Image.new("RGB", (224, 224), color="white")
    emb = retina.encode([img])
    print(emb.shape)
