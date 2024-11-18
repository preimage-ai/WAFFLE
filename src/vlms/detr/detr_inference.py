from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image


class DetrInference:
    def __init__(self, model_path="facebook/detr-resnet-50"):
        self.model = DetrForObjectDetection.from_pretrained(model_path)
        self.processor = DetrImageProcessor.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_score(self, image_path, threshold=0.5):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {self.model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )


if __name__ == "__main__":
    table_detr = DetrInference("TahaDouaji/detr-doc-table-detection")
    image_path = "data/for_object_detection/WAFFLE/20_of_'(Hereford_Cathedral.)_Ward_and_Lock's_Illustrated_Historical_Handbook_to_Hereford_Cathedral,_etc'_(11204965424).jpg"
    table_detr.get_score(image_path)
    pass
