import math
import pickle
from tqdm.auto import tqdm, trange
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from argparse import ArgumentParser
import os
from vlms.clipseg.v2.plot_utils import plot_losses
from vlms.clipseg.v2.dataset import CLIPSegDataset
from vlms.clipseg.v2.data import CLIPSegDatum, CLIPSegGT

import statistics
import torch
import json


def get_opts():
    parser = ArgumentParser()
    parser.add_argument(
        "--train_dataset_fn",
        "-d",
        type=str,
        default="data/for_segmentation/churches_dataset_train.pkl",
    )
    parser.add_argument(
        "--validation_dataset_fn",
        "-v",
        type=str,
        default="data/for_segmentation/churches_dataset_test.pkl",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        default="CIDAS/clipseg-rd64-refined",
    )
    parser.add_argument(
        "--with_pseudo_labels",
        "-p",
        type=bool,
        default=False,
        help="whether or not the dataset is a pseudo labels one",
    )
    parser.add_argument("--epochs", "-e", type=int, default=4, help="epochs")
    parser.add_argument("--lr", "-l", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--augment_data",
        "--a",
        type=bool,
        default=True,
        help="whether or not to augment the images and boxes in the dataset",
    )
    parser.add_argument(
        "--grad_acc_steps",
        "-g",
        type=int,
        default=1,
        help="gradient accumulation steps",
    )
    parser.add_argument(
        "--lambda_loss",
        type=float,
        default=0.5,
        help="lambda for loss function - how much weight to give the main loss (GT vs pred). (1-lambda)  will be the weight for L2 loss)",
    )
    parser.add_argument(
        "--save_every",
        "-s",
        type=int,
        default=5,
        help="save model every s epochs",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="checkpoints/segmentation/churches",
        help="output checkpoint directory",
    )

    return parser.parse_args()


def save_model(
    model,
    output_dir,
    losses,
    losses_gt,
    losses_l2,
    losses_entropy,
    valitation,
    val_losses,
    val_losses_gt,
    val_losses_l2,
    val_losses_entropy,
):
    print("Saving trained model to:", output_dir)
    if os.path.exists(output_dir):
        print(f"Warning: {output_dir} exists; overwriting")

    model.save_pretrained(output_dir)
    loss_path = f"{output_dir}/loss.json"
    print("Saving losses to:", loss_path)
    if os.path.exists(loss_path):
        print(f"Warning: {loss_path} exists; overwriting")
    losses_obj = {
        "losses": losses,
        "losses_gt": losses_gt,
        "losses_l2": losses_l2,
        "losses_entropy": losses_entropy,
    }
    if valitation:
        losses_obj["val_losses"] = val_losses
        losses_obj["val_losses_gt"] = val_losses_gt
        losses_obj["val_losses_l2"] = val_losses_l2
        losses_obj["val_losses_entropy"] = val_losses_entropy
    with open(loss_path, "w") as f:
        json.dump(losses_obj, f, indent=4)
    plot_losses(output_dir)


def main():
    args = get_opts()

    print("Loading models...")
    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained(args.model_path)
    model.to("cuda")

    print("Models loaded")

    valitation = os.path.exists(args.validation_dataset_fn)
    with_pseudo_labels = args.with_pseudo_labels

    print("Loading data...")
    ds = CLIPSegDataset(pckl_path=args.train_dataset_fn, augment=args.augment_data)
    dl = DataLoader(ds, shuffle=True, num_workers=0, collate_fn=lambda x: x[0])

    if valitation:
        val_ds = CLIPSegDataset(args.validation_dataset_fn, augment=False)
        val_dl = DataLoader(
            val_ds, shuffle=True, num_workers=0, collate_fn=lambda x: x[0]
        )
    print("Data loaded")

    print("Starting training...")

    loss_fn = nn.BCEWithLogitsLoss()

    epochs = args.epochs
    lr = args.lr
    grad_acc_steps = args.grad_acc_steps
    lambda_loss = args.lambda_loss
    save_every = args.save_every

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    losses_gt = []
    losses_l2 = []
    losses_entropy = []

    val_losses = []
    val_losses_gt = []
    val_losses_l2 = []
    val_losses_entropy = []
    for i in trange(epochs):
        if i > 0 and i % save_every == 0:
            save_model(
                model,
                os.path.join(args.output, f"epoch_{i}"),
                losses,
                losses_gt,
                losses_l2,
                losses_entropy,
                valitation,
                val_losses,
                val_losses_gt,
                val_losses_l2,
                val_losses_entropy,
            )
        for j, item in enumerate(tqdm(dl)):
            if j % grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            inp = processor(
                images=item.img, text=item.label, padding=True, return_tensors="pt"
            ).to("cuda")
            out = model(**inp).decoder_output.logits
            scores = torch.tensor(item.scores).to("cuda")
            mask_torch = torch.tensor(item.mask).to("cuda")
            mask_flat = mask_torch.ravel() > 0
            loss = loss_fn(out.ravel()[mask_flat], scores.ravel()[mask_flat])
            loss = loss * lambda_loss
            losses_gt.append(loss.item())
            
            entropy_loss = loss_fn(out.ravel(), out.ravel().sigmoid())
            lambda_entropy_loss = 1.0
            loss += entropy_loss * lambda_entropy_loss
            losses_entropy.append(entropy_loss.item())
            

            l2_loss = out.sigmoid().mean()
            losses_l2.append(l2_loss.item())
            loss += l2_loss * (1 - lambda_loss)
            if math.isnan(loss.item()):
                pass
            losses.append(loss.item())
            loss.backward()

        if valitation:
            # Evaluate on validation set
            print(f"Evaluating on validation set for epoch {i}")
            for item in tqdm(val_dl):
                inp = processor(
                    images=item.img, text=item.label, padding=True, return_tensors="pt"
                ).to("cuda")
                out = model(**inp).decoder_output.logits
                scores = torch.tensor(item.scores).to("cuda")
                mask_torch = torch.tensor(item.mask).to("cuda")
                mask_flat = mask_torch.ravel() > 0
                loss = loss_fn(out.ravel()[mask_flat], scores.ravel()[mask_flat])
                loss = loss * lambda_loss
                val_losses_gt.append(loss.item())
                
                entropy_loss = loss_fn(out.ravel(), out.ravel().sigmoid())
                lambda_entropy_loss = 1.0
                loss += entropy_loss * lambda_entropy_loss
                val_losses_entropy.append(entropy_loss.item())

                l2_loss = out.sigmoid().mean()
                val_losses_l2.append(l2_loss.item())
                loss += l2_loss * (1 - lambda_loss)

                val_losses.append(loss.item())

            print(
                f"Validation loss for this epoch: {statistics.mean(val_losses[i*len(val_dl):])}"
            )
        print(f"Loss for this epoch: {statistics.mean(losses[i*len(dl):])}")

    print("Training finished")

    save_model(
        model,
        args.output,
        losses,
        losses_gt,
        losses_l2,
        valitation,
        val_losses,
        val_losses_gt,
        val_losses_l2,
    )

    print("Saving model fine tuning params...")
    params_path = f"{args.output}/params.json"
    if os.path.exists(params_path):
        print(f"Warning: {params_path} exists; overwriting")
    params_obj = {
        "data_path": args.train_dataset_fn,
        "model_path": args.model_path,
        "data_size": len(ds),
        "data_augmentation": args.augment_data,
        "with_pseudo_labels": with_pseudo_labels,
        "epochs": epochs,
        "lr": lr,
        "grad_acc_steps": grad_acc_steps,
        "lambda_loss": lambda_loss,
    }
    if valitation:
        params_obj["validation_data_path"] = args.validation_dataset_fn
        params_obj["validation_data_size"] = len(val_ds)
    with open(params_path, "w") as f:
        json.dump(params_obj, f, indent=4)

    print("done")


if __name__ == "__main__":
    main()
