# taken from https://huggingface.co/blog/fine-tune-vit

import argparse
import os
import pandas as pd
import torch
import numpy as np

from datasets import load_dataset, load_metric
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import (
    TrainingArguments,
    Trainer,
    ViTImageProcessor,
    ViTForImageClassification,
)
from vlms.vit.vit_inference import VitInference

BUILDING_TOPICS = ["A", "B"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the training data directory",
        default="data/for_classifier",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to use",
        default="google/vit-base-patch16-224-in21k",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        help="Path to directory where the finetuned model will be stored",
        default="checkpoints/vit-base-floorplan-classifier",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train for",
        default=5,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        help="Number of steps between saving checkpoints",
        default=100,
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        help="Number of steps between evaluating on the validation set",
        default=100,
    )
    parser.add_argument(
         "--df_path",
        type=str,
        default="data/all_data.csv",
        help="path to df that contains the initial dataset",
    )
    parser.add_argument(
         "--large_dataset_path",
        type=str,
        default="data/large_dataset.csv",
        help="the path we'll store the filtered large dataset",
    )
        
    return parser.parse_args()


def transform(example_batch, processor):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor(
        images=[read_img(x) for x in example_batch["path"]], return_tensors="pt"
    )

    # Don't forget to include the labels!
    inputs["label"] = example_batch["label"]
    return inputs


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }


def compute_metrics(p, metric):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )


def read_img(path):
    img = Image.open(path)
    return img.convert("RGB")


def finetune(args):
    ds = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(args.data_dir, "train.csv"),
            "validation": os.path.join(args.data_dir, "valid.csv"),
            "test": os.path.join(args.data_dir, "test.csv"),
        },
    )

    processor = ViTImageProcessor.from_pretrained(args.model_name)

    prepared_ds = ds.with_transform(transform=lambda x: transform(x, processor))
    metric = load_metric("accuracy")

    labels = set(ds["train"]["label"])
    id2label = {
        c: "A floorplan" if c == 1 else "Not a floorplan" for _, c in enumerate(labels)
    }
    label2id = {v: k for k, v in id2label.items()}

    model = ViTForImageClassification.from_pretrained(
        args.model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )
    training_args = TrainingArguments(
        output_dir=args.ckpt_dir,
        per_device_train_batch_size=4,
        evaluation_strategy="steps",
        num_train_epochs=args.epochs,
        fp16=True,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=1,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=lambda x: compute_metrics(x, metric),
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["validation"],
        tokenizer=processor,
    )
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    metrics = trainer.evaluate(prepared_ds["validation"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def add_classifier_scores_to_df(args):
    df = pd.read_csv(args.df_path)
    vit_inference = VitInference(args.ckpt_dir)
    df["classifier_score"] = df["img_path"].progress_apply(vit_inference.get_score)
    df.to_csv(args.df_path, index=False)
    

def save_large_dataset(args):
    '''
    The large dataset is created with the following filters:
    1. The depicted topic is a building, and
    2.1 The image has a wide CLIP score > 0.5, or
    2.2 The image has a classifier score > 0.005
    '''
    df = pd.read_csv(args.df_path)
    topic_building_mask = df.topic_answer.isin(BUILDING_TOPICS)
    clip_mask = df.wide_clip_score > 0.5
    classifier_mask = df.classifier_score > 0.005
    large_dataset = df[topic_building_mask & (clip_mask | classifier_mask)]
    large_dataset.to_csv(args.large_dataset_path, index=False)


if __name__ == "__main__":
    args = get_args()
    finetune(args)
    add_classifier_scores_to_df(args)
    save_large_dataset(args)
