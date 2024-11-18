from argparse import ArgumentParser
import torch
from vlms.clipseg.v2.dataset import DEFAULT_SIZE
from vlms.clipseg.v2.sizing_utils import resize_image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import requests
from io import BytesIO

def load_image_from_url(img_url):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    return img


class ClipSegInference:
    def __init__(self, model_path):
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_path)
        self.model.to("cuda")

    def call_model(self, model, img, labels):
        if isinstance(img, str) and img.startswith("http"):
            img = load_image_from_url(img).convert("RGB")
        img = resize_image(img, DEFAULT_SIZE)
        inp = self.processor(
            images=[img] * len(labels), text=labels, padding=True, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            res = model(**inp).logits
        return res.unsqueeze(1).sigmoid() if len(labels) > 1 else res.sigmoid()

    def call(self, img, labels):
        return self.call_model(self.model, img, labels)
