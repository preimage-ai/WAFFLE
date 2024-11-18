import argparse
import json
import matplotlib.pyplot as plt
import torch
from itertools import groupby
from PIL import Image
from torchvision.ops.boxes import nms as torch_nms
from typing import List, Tuple
from transformers import DetrImageProcessor
from vlms.detr.detr import Detr

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="lightning_logs/camera_ready/model",
        # default="lightning_logs/all_with_scale_more_iters/model",
    )
    parser.add_argument(
        "--id2label_path",
        type=str,
        default="lightning_logs/camera_ready/id2label.json",
    )
    return parser.parse_args()


def filter_results(
    scores: List[float],
    labels: List[int],
    boxes: List[Tuple[int, int, int, int]],
    iou_threshold=0.5,
):
    """
    Filter overlapping boxes.

    Parameters:
    scores (List[float]): The scores of the boxes.
    labels (List[int]): The labels of the boxes.
    boxes (List[Tuple[int, int, int, int]]): The boxes to filter.
    """
    # Combine the scores, labels and boxes into a single list
    combined = list(zip(scores, labels, boxes))
    # Sort the list by score in descending order
    combined.sort(key=lambda x: x[0], reverse=True)
    # Sort the list by label
    combined.sort(key=lambda x: x[1])
    # group by label
    grouped_lists = [list(group) for _, group in groupby(combined, key=lambda x: x[1])]
    filtered_boxes = []
    for labeled_boxes in grouped_lists:
        filtered_indices = torch_nms(
            torch.tensor([box for _, _, box in labeled_boxes]),
            torch.tensor([score for score, _, _ in labeled_boxes]),
            iou_threshold,
        )
        filtered_boxes += [labeled_boxes[i] for i in filtered_indices]
    if filtered_boxes:
        return zip(*filtered_boxes)
    return [], [], []


def plot_results(image_path, results, id2label, output_path="a.png"):
    image = Image.open(image_path)
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    ax = plt.gca()
    colors = COLORS * 100
    filtered_scores, filtered_labels, filtered_boxes = filter_results(
        results["scores"].tolist(),
        results["labels"].tolist(),
        results["boxes"].tolist(),
    )
    for score, label, (xmin, ymin, xmax, ymax), c in zip(
        filtered_scores, filtered_labels, filtered_boxes, colors
    ):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3
            )
        )
        text = f"{id2label[label]}: {score:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    if (
        len(filtered_scores) == 0
        or len(filtered_labels) == 0
        or len(filtered_boxes) == 0
    ):
        return
    plt.axis("off")
    plt.show()
    plt.savefig(output_path)
    plt.close()


class DETRInference:
    def __init__(self, model_path, id2label_path):
        id2label_json = json.load(open(id2label_path))
        self.id2label = {int(k): v for k, v in id2label_json.items()}
        model = Detr(
            lr=1e-4,
            lr_backbone=1e-5,
            weight_decay=1e-4,
            num_labels=len(self.id2label),
            model_path=model_path,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.processor = DetrImageProcessor.from_pretrained(model_path)

    def infer(self, image_path, threshold=0.5):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].to(self.device)
        with torch.no_grad():
            # forward pass to get class logits and bounding boxes
            outputs = self.model(pixel_values=pixel_values, pixel_mask=None)
        width, height = image.size
        postprocessed_outputs = self.processor.post_process_object_detection(
            outputs, target_sizes=[(height, width)], threshold=threshold
        )
        return postprocessed_outputs[0]


if __name__ == "__main__":
    args = get_args()
    detr_inf = DETRInference(args.model_path, args.id2label_path)
    results = detr_inf.infer("castle_Penrhyn_Castle_Grundriss.png")
    plot_results("castle_Penrhyn_Castle_Grundriss.png", results, detr_inf.id2label)
