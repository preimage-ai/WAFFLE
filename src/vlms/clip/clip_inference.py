import json
import os
import torch

from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class ClipInference:
    """Class for calling CLIP"""

    def __init__(self, model_path="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def call_clip(self, image_path, categories_list):
        """
        Calls CLIP with categories for each image in a directory

        Args:
            image_path: str, path to image
            categories_list: list of strings, categories to test

        Returns:
            dict, mapping from category to score
        """
        image = Image.open(image_path)

        inputs = self.processor(
            text=categories_list,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        logits_per_image = (
            outputs.logits_per_image
        )  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)
        return {
            key: value
            for key, value in zip(
                categories_list, probs.cpu().detach().numpy().flatten().tolist()
            )
        }

    @torch.no_grad()
    def store_clip_scores(self, image_path, categories_list, output_path):
        """
        Calls CLIP with categories for each image in a directory and stores the scores as JSON files

        Args:
            image_path: str, path to image
            categories_list: list of strings, categories to test

        Returns:
            a path to the JSON file containing a mapping from category to score
        """
        if os.path.exists(output_path):
            print(f"File {output_path} already exists, skipping")
            return output_path
        categories_scores = self.call_clip(image_path, categories_list)
        with open(output_path, "w") as json_file:
            json.dump(categories_scores, json_file)
