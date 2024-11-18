from torchvision.ops.boxes import nms as torch_nms
from torchvision.ops.boxes import box_iou
import os
import torch
import json

from ocr.models.gcp.ocr_texts import OCRTexts

id2label = json.load(open("data/for_object_detection/WAFFLE_all/id2label.json"))
label2id = {v: int(k) for k, v in id2label.items()}
estimated_thresholds_per_label = {
    "floorplan": 0.5,
    "legend": 0.4,
}


class DetrResult:
    def __init__(self, result_dir, threshold=0.4, nms_threshold=0.5):
        scores = torch.load(os.path.join(result_dir, "scores.pt"))
        labels = torch.load(os.path.join(result_dir, "labels.pt"))
        boxes = torch.load(os.path.join(result_dir, "boxes.pt"))
        if nms_threshold:
            filtered_indices = torch_nms(boxes, scores, nms_threshold)
            scores = scores[filtered_indices]
            labels = labels[filtered_indices]
            boxes = boxes[filtered_indices]
        self.results = {
            "scores": scores[scores > threshold],
            "labels": labels[scores > threshold],
            "boxes": boxes[scores > threshold],
        }

    def get_boxes(self, label):
        if label not in label2id:
            raise ValueError(f"Label {label} not found in the dataset")
        label_id = label2id[label]
        mask = self.results["labels"] == label_id
        if label in estimated_thresholds_per_label:
            mask &= self.results["scores"] > estimated_thresholds_per_label[label]
        return self.results["boxes"][mask]

    def get_ocr_blocks(self, ocr_fn, label=None, negative_labels=[], iou_threshold=0.5):
        """
        Returns the OCR blocks that match the boxes of the label. We return the
        OCR blocks that have an IOU greater than iou_threshold with the boxes of the label,
        and have an IOU less than iou_threshold with the boxes of the negative_labels.
        """
        positive_boxes = self.get_boxes(label) if label else []
        negative_boxes = [
            box
            for negative_label in negative_labels
            for box in self.get_boxes(negative_label)
        ]
        ocr_blocks = OCRTexts(ocr_fn).get_ocr_texts(level="block")
        candidates = []
        for block in ocr_blocks:
            if block in candidates:
                continue
            if (
                len(positive_boxes) == 0
                and maximal_intersection_by_min_area(
                    get_box_from_block(block), negative_boxes
                )
                < iou_threshold
            ):
                candidates.append(block)
                continue
            else:
                for box in positive_boxes:
                    if (
                        intersection_by_min_area(box.cpu(), get_box_from_block(block))
                        > iou_threshold
                    ) and maximal_intersection_by_min_area(
                        get_box_from_block(block), negative_boxes
                    ) < iou_threshold:
                        candidates.append(block)
                        break
        return candidates


def get_box_from_block(block):
    """
    Returns the bounding box of the block in the format (xmin, ymin, xmax, ymax)
    """
    xs = [vertice.x for vertice in block.bounding_box.vertices]
    ys = [vertice.y for vertice in block.bounding_box.vertices]
    return [min(xs), min(ys), max(xs), max(ys)]


def maximal_intersection_by_min_area(box, boxes_to_compare):
    """
    Returns the maximal intersection over the minimum area of the box with all the boxes_to_compare.
    """
    return max(
        [
            intersection_by_min_area(box, box_to_compare)
            for box_to_compare in boxes_to_compare
        ],
        default=0,
    )


def intersection_by_min_area(box1, box2):
    """
    Returns the intersection over the minimum area of the two boxes.
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin_inter = max(xmin1, xmin2)
    ymin_inter = max(ymin1, ymin2)
    xmax_inter = min(xmax1, xmax2)
    ymax_inter = min(ymax1, ymax2)
    intersection = max(0, xmax_inter - xmin_inter) * max(0, ymax_inter - ymin_inter)

    # denominator is maxed with 1 to avoid division by zero
    return intersection / max(min(area1, area2), 1)


if __name__ == "__main__":
    page_id = 11578291
    detr_result = DetrResult(
        f"data/outputs/od_outputs/all_with_scale_more_iters/{page_id}"
    )
    block_candidates = detr_result.get_ocr_blocks(
        f"data/outputs/ocr_outputs_v2/{page_id}.json",
        label="floorplan",
        negative_labels=["legend"],
    )
    pass
