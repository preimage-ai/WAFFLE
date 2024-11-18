from argparse import ArgumentParser
import os
import pickle
import torch
from vlms.clipseg.v2. plot_utils import plot_base_vs_ft
from vlms.clipseg.v2.dataset import DEFAULT_SIZE
from vlms.clipseg.v2.sizing_utils import resize_image
from vlms.clipseg.v2.data import CLIPSegDatum, CLIPSegGT
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from PIL import Image
import requests
from io import BytesIO


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--test_dataset_fn",
        type=str,
        default="data/for_segmentation/churches_dataset_test.pkl",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="CIDAS/clipseg-rd64-refined",
    )
    parser.add_argument(
        "--ft_model_path",
        type=str,
        default="checkpoints/segmentation/churches_10_epochs",
    )

    return parser.parse_args()


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


if __name__ == "__main__":
    args = get_args()
    ft_inference = ClipSegInference(args.ft_model_path)
    original_inference = ClipSegInference(args.base_model_path)
    test_ds = pickle.load(open(args.test_dataset_fn, "rb"))
    for datum in test_ds:
        plots_dir = os.path.join(args.ft_model_path, "test_images")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        for gt in datum.gts:
            for label in gt.pos_labels_to_boxes.keys():
                base_res = original_inference.call(datum.img, [label])
                ft_res = ft_inference.call(datum.img, [label])
                plot_base_vs_ft(
                    datum.img,
                    gt,
                    base_res,
                    ft_res,
                    label,
                    f"{plots_dir}/{datum.df_row.page_id}_{label}.png",
                )
