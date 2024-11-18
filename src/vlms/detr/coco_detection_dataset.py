import torchvision
import os
import albumentations as A
import cv2
import copy
import numpy as np
from vlms.detr.plotting_utils import visualize_gt

DATASET_DIR = "/storage/kerenganon/waffle/data/for_object_detection/WAFFLE"
JSON_FN = "WAFFLE.json"


def split_test_train(ann_file):
    import json
    from sklearn.model_selection import train_test_split

    with open(ann_file) as f:
        data = json.load(f)
    train, test = train_test_split(data["images"], test_size=0.2, random_state=42)
    train_data = {
        "images": train,
        "categories": data["categories"],
        "annotations": data["annotations"],
    }
    test_data = {
        "images": test,
        "categories": data["categories"],
        "annotations": data["annotations"],
    }
    with open("train.json", "w") as f:
        json.dump(train_data, f)
    with open("test.json", "w") as f:
        json.dump(test_data, f)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, processor, dataset_dir, json_fn, augment=False):
        ann_file = os.path.join(dataset_dir, json_fn)
        super(CocoDetection, self).__init__(dataset_dir, ann_file)
        self.processor = processor
        self.id2label = {k: v["name"] for k, v in self.coco.cats.items()}
        self.transform = None
        if augment:
            self.transform = A.Compose(
                [
                    A.GaussNoise(p=0.5),
                    A.ISONoise(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                ],
                bbox_params=A.BboxParams(
                    format="coco",
                    min_visibility=0.2,
                ),
            )

    def __getitem__(self, idx):
        img, target = self.maybe_perform_transformation(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target

    def maybe_perform_transformation(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)
        # visualize_gt(img, target, self.id2label, output_path="a.png")
        if self.transform is None:
            return img, target
        # we add the id as the last item in each bbox so we can retrieve its location later.
        # anything after index 4 is ignored by albumentations
        id_bboxes = [
            list(annotation["bbox"]) + list([annotation["id"]]) for annotation in target
        ]
        try:
            transformed = self.transform(
                image=np.array(img),
                bboxes=id_bboxes,
            )
        except Exception as e:
            print(f"Exception thrown while augmenting. Error: {e}")
            return img, target
        id_to_transformed_bbox = {
            transformed_bbox[4]: transformed_bbox[:4]
            for transformed_bbox in transformed["bboxes"]
        }
        transformed_img = transformed["image"]
        transformed_annotations = []
        for annotation in target:
            if annotation["id"] in id_to_transformed_bbox.keys():
                transformed_annotation = copy.deepcopy(annotation)
                transformed_annotation["bbox"] = id_to_transformed_bbox[annotation["id"]]
                transformed_annotations.append(transformed_annotation)
        # visualize_gt(img, target, self.id2label, output_path="b.png")
        return transformed_img, transformed_annotations
