from PIL import Image, ImageDraw
from torchvision.ops.boxes import nms as torch_nms
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple
from itertools import groupby

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def intersection_over_min_area(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the ratio of the intersection area to the area of the smaller box
    minArea = min(boxAArea, boxBArea)
    ratio = interArea / float(minArea)

    # Return the intersection over minimum area ratio
    return ratio

def filter_boxes(boxes, threshold):
    chosen_indices = set()
    filtered_indices = set()
    i = 0
    while i < len(boxes):
        if i in filtered_indices:
            i += 1
            continue
        chosen_indices.add(i)
        j = i + 1
        while j < len(boxes):
            if j in filtered_indices:
                j += 1
                continue
            if intersection_over_min_area(boxes[i], boxes[j]) > threshold:
                filtered_indices.add(j)
            j += 1
        i += 1
    return chosen_indices


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
        # filtered_indices = filter_boxes([box for _, _, box in labeled_boxes], iou_threshold)
        filtered_boxes += [labeled_boxes[i] for i in filtered_indices]
    if filtered_boxes:
        return zip(*filtered_boxes)
    return [], [], []

def plot_results(pil_img, scores, labels, boxes, id2label, output_path="a.png"):
    plt.figure(figsize=(12, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    filtered_scores, filtered_labels, filtered_boxes = filter_results(
        scores.tolist(), labels.tolist(), boxes.tolist()
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
    if len(scores) == 0 or len(labels) == 0 or len(boxes) == 0:
        return
    plt.axis("off")
    plt.show()
    plt.savefig(output_path)
    plt.close()

def filter_results_max(
    scores: List[float],
    labels: List[int],
    boxes: List[Tuple[int, int, int, int]],
    iou_threshold=0.5,
):
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
        filtered_indices = [0]
        filtered_boxes += [labeled_boxes[i] for i in filtered_indices]
    if filtered_boxes:
        return zip(*filtered_boxes)
    return [], [], []

def plot_results_max(pil_img, scores, labels, boxes, id2label, output_path="a.png"):
    plt.figure(figsize=(12, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    filtered_scores, filtered_labels, filtered_boxes = filter_results_max(
        scores.tolist(), labels.tolist(), boxes.tolist()
    )
    for score, label, (xmin, ymin, xmax, ymax), c in zip(
        filtered_scores, filtered_labels, filtered_boxes, colors
    ):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3
            )
        )
        # text = f"{id2label[label]}: {score:0.2f}"
        text = f"{id2label[label]}: {score:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    if len(scores) == 0 or len(labels) == 0 or len(boxes) == 0:
        return
    plt.axis("off")
    plt.show()
    plt.savefig(output_path)
    plt.close()

def test_image(model, processor, device, image_path, id2label, output_path="a.png"):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    encoding = processor(images=image, return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(device)
    with torch.no_grad():
        # forward pass to get class logits and bounding boxes
        outputs = model(pixel_values=pixel_values, pixel_mask=None)
    width, height = image.size
    postprocessed_outputs = processor.post_process_object_detection(
        outputs, target_sizes=[(height, width)], threshold=0.3
    )
    results = postprocessed_outputs[0]
    plot_results(
        image,
        results["scores"],
        results["labels"],
        results["boxes"],
        id2label,
        output_path=output_path,
    )


def get_ordered_bbox(bbox):
    x0, y0, x1, y1 = bbox
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

def visualize_gt(image, annotations, id2label, output_path="a.png"):
    if type(image) != Image.Image:
        image = Image.fromarray(image)
    draw = ImageDraw.Draw(image, "RGBA")
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'cyan', 'magenta']
    for annotation in annotations:
        box = annotation['bbox']
        class_idx = annotation['category_id']
        x,y,w,h = tuple(box)
        draw.rectangle((x,y,x+w,y+h), outline=colors[class_idx], width=4)
        draw.text((x, y), id2label[class_idx], fill='black')

    image.save(output_path)
