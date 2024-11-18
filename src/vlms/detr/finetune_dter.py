import pandas as pd
from tqdm.auto import tqdm
from transformers import DetrImageProcessor
from PIL import Image
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from transformers import DetrForObjectDetection
import torch
from vlms.detr.plotting_utils import plot_results
from vlms.detr.coco_detection_dataset import CocoDetection
from vlms.detr.detr import Detr
import argparse
import json

Image.MAX_IMAGE_PIXELS = None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_dir",
        type=str,
        default="data/for_object_detection/WAFFLE_all",
        help="path to directory that contains the dataset",
    )
    parser.add_argument(
        "--split",
        type=bool,
        default=False,
        help="whether or not to split the dataset into train and test",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="TahaDouaji/detr-doc-table-detection",
        help="the base model to use for finetuning",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1300,
        help="number of steps to train the model for",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="lightning_logs/camera_ready/model",
        help="where to store the FT model",
    )
    parser.add_argument(
        "--images_output_dir",
        type=str,
        default="data/od_images/first_iter",
        help="the directory to store the tested images",
    )
    parser.add_argument(
        "--test_ft",
        type=bool,
        default=False,
        help="whether or not to test the finetuned model",
    )
    return parser.parse_args()


def test_image(
    model, processor, device, image_path, id2label, threshold, output_path="a.png"
):
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
        outputs, target_sizes=[(height, width)], threshold=threshold
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


def get_floorplan_max_box(
    model, processor, device, image_path, id2label, threshold, output_path="a.png"
):
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
        outputs, target_sizes=[(height, width)], threshold=threshold
    )
    results = postprocessed_outputs[0]
    combined = list(zip(results["scores"], results["labels"], results["boxes"]))
    # Sort the list by score in descending order
    combined.sort(key=lambda x: x[0], reverse=True)

    label2id = {v: k for k, v in id2label.items()}
    floorplan_label = label2id["floorplan"]
    # filter for floorplans only
    combined_floorplans = [x for x in combined if x[1] == floorplan_label]


def test_model(model_path, processor, id2label, output_dir, threshold=0.35):
    model = Detr(
        lr=1e-4,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        num_labels=len(id2label),
        model_path=model_path,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tqdm.pandas()
    df = pd.read_csv("data/csvs_v2/clean/dataset.csv")
    df_with_legend = df[df.has_ocr_legend_v3 == "Y"]
    df_with_legend.progress_apply(
        lambda row: test_image(
            model,
            processor,
            device,
            row["img_path"],
            id2label,
            threshold,
            os.path.join(output_dir, f"{row['page_id']}.png"),
        ),
        axis=1,
    )


def get_floorplan_max_boxes(
    model_path, processor, id2label, output_dir, threshold=0.35
):
    model = Detr(
        lr=1e-4,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        num_labels=len(id2label),
        model_path=model_path,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tqdm.pandas()
    df = pd.read_csv("data/csvs_v2/clean/dataset.csv")
    # df_with_legend = df[df.has_ocr_legend_v3 == "Y"]
    df["floorplan_max_box"] = df.progress_apply(
        lambda row: test_image(
            model,
            processor,
            device,
            row["img_path"],
            id2label,
            threshold,
            os.path.join(output_dir, f"{row['page_id']}.png"),
        ),
        axis=1,
    )


def split_test_train(ann_dir, ann_fn):
    from sklearn.model_selection import train_test_split

    with open(os.path.join(ann_dir, ann_fn)) as f:
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
    with open(os.path.join(ann_dir, "train.json"), "w") as f:
        json.dump(train_data, f)
    with open(os.path.join(ann_dir, "test.json"), "w") as f:
        json.dump(test_data, f)


def save_tensor_as_image(tensor, filename):
    """
    Save a tensor as an image.

    Parameters:
    tensor (torch.Tensor): The tensor to save as an image.
    filename (str): The name of the file to save the image as.
    """
    # Convert the tensor to a PIL Image
    image = transforms.ToPILImage()(tensor)

    # Save the image
    image.save(filename)


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


if __name__ == "__main__":
    args = get_args()
    if args.split:
        coco_fn = os.path.basename(args.ds_dir) + ".json"
        split_test_train(ann_dir=args.ds_dir, ann_fn=coco_fn)
    processor = DetrImageProcessor.from_pretrained(args.model_path)
    train_dataset = CocoDetection(
        processor=processor, dataset_dir=args.ds_dir, json_fn="train.json", augment=True
    )
    val_dataset = CocoDetection(
        processor=processor, dataset_dir=args.ds_dir, json_fn="test.json"
    )
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))
    train_dataloader = DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=4, num_workers=16, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)
    id2label = train_dataset.id2label

    model = Detr(
        lr=1e-4,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        num_labels=len(id2label),
        train_dl=train_dataloader,
        val_dl=val_dataloader,
        model_path=args.model_path,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trainer = Trainer(max_steps=args.steps, gradient_clip_val=0.1)
    trainer.fit(model)
    print("Done training")
    print(f"Saving model to {args.output_path}...")

    model.model.save_pretrained(args.output_path)
    processor.save_pretrained(args.output_path.replace("model", "processor"))
    processor.save_pretrained(args.output_path)
    print("Model saved")
    if args.test_ft:
        print("Testing model...")
        test_model(
            model_path=args.output_path,
            processor=processor,
            id2label=id2label,
            output_dir=args.images_output_dir,
            threshold=0.5,
        )
        print("Done testing")
