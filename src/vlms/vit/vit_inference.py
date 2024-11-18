import torch

from PIL import Image
from transformers import ViTForImageClassification, AutoImageProcessor

class VitInference:
    """Class for calling ViT for image classification"""
    def __init__(self, model_path = "google/vit-base-patch16-224-in21k"):
        self.model = ViTForImageClassification.from_pretrained(model_path)
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_score(self, image_path):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        logits = self.model(**inputs).logits
        probs = logits.softmax(
            dim=1
        )  # take the softmax to get the label probabilities
        return probs[0][1].item()