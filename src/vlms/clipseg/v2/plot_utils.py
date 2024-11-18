import textwrap
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np


COLORS = {
    "red": np.array([1, 0, 0, 0.6]),
    "green": np.array([0, 1, 0, 0.6]),
    "gray": np.array([128 / 255, 128 / 255, 128 / 255, 0.6]),
}

CHOSEN_GREEN_RGB = np.array([60, 179, 113])


def show_pred(pred, ax):
    C = np.zeros((256, 4), dtype=np.float32)
    C[:, 0] = np.linspace(1, 0, 256)  # Green component
    C[:, 1] = np.linspace(0, 1, 256)  # Red component
    C[:, -1] = np.linspace(0, 1, 256)  # Alpha component
    cmap_ = ListedColormap(C)
    ax.imshow(pred, cmap=cmap_)


def show_boolean_mask(mask, ax, color="green"):
    mask = mask.astype(bool)
    try:
        color = COLORS[color]
    except KeyError:
        print(f"Color {color} not found. Using green instead.")
        color = COLORS["green"]
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plot_gt(img, clipseg_gt, ax=None, output_path=None, show_box_labels=False):
    if not ax:
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
    ax.imshow(img)
    show_boolean_mask(clipseg_gt.scores, ax, color="green")
    neg_mask = clipseg_gt.mask.astype(bool) & ~clipseg_gt.scores.astype(bool)
    show_boolean_mask(neg_mask, ax, color="red")
    if show_box_labels:
        for label, boxes in {**clipseg_gt.pos_labels_to_boxes, **clipseg_gt.neg_labels_to_boxes}.items():
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                ax.add_patch(
                    plt.Rectangle(
                        (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color="yellow", linewidth=1
                    )
                )
                text = f"{label}"
                ax.text(xmin, ymin, text, fontsize=12, bbox=dict(facecolor="yellow", alpha=0.5))
    if output_path:
        plt.savefig(output_path)
        plt.close()


def plot_datum(clipseg_datum, output_path):
    '''
    for every gt in the datum, plot the gt on the image
    '''
    n = len(clipseg_datum.gts)
    nrows = int(np.ceil(np.sqrt(n)))
    ncols = int(np.ceil(n / nrows))

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))
    
    # If axs is not a 2D array, convert it into a 2D array
    if not isinstance(axs, np.ndarray):
        axs = np.array([[axs]])


    for i, gt in enumerate(clipseg_datum.gts):
        row = i // ncols
        col = i % ncols
        if np.ndim(axs) == 1:
            ax = axs[i]
        else:
            ax = axs[row, col]
        plot_gt(clipseg_datum.img, gt, ax=ax, show_box_labels=False)
        ax.set_title(f"label: {list(gt.pos_labels_to_boxes.keys())}")
    
        # Remove empty subplots
    for i in range(n, nrows*ncols):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    

def plot_loss(loss, loss_key="losses", output_path="loss.png", window_size=100):
    import pandas as pd

    loss_df = pd.DataFrame(loss[loss_key], columns=["Loss"])

    # Compute the rolling mean
    loss_df["Rolling Mean"] = loss_df["Loss"].rolling(window=window_size).mean()

    plt.plot(loss_df["Rolling Mean"])
    plt.title("Loss curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(output_path)
    plt.close()

def plot_losses(model_path, window_size=100):
    import json

    loss = json.load(open(f"{model_path}/loss.json"))
    for key in loss.keys():
        plot_loss(
            loss,
            loss_key=key,
            output_path=f"{model_path}/{key}.png",
            window_size=window_size,
        )


def plot_res_on_ax(img, gt, res, label, ax):
    plot_gt(img, gt, ax=ax[0])
    ax[0].text(
        0, -15, "\n".join(textwrap.wrap(f"GT: {label}", 30)), fontsize=20
    )  # Wrap the text
    res = res.cpu()
    ax[1].imshow(res, cmap="RdYlGn")
    ax[1].text(
        0, -15, "\n".join(textwrap.wrap(f"Pred: {label}", 30)), fontsize=20
    )  # Wrap the text


def plot_base_vs_ft(img, gt, base_res, ft_res, label, output_path):
    _, ax = plt.subplots(2, 3, figsize=(20, 10))
    [a.axis("off") for a in ax.flatten()]

    ax[0, 0].text(0.5, 0.5, "Base", fontsize=30, va="center", ha="center")
    
    plot_res_on_ax(img, gt, base_res, label, ax[0, 1:])

    ax[1, 0].text(0.5, 0.5, "FT", fontsize=30, va="center", ha="center")
    plot_res_on_ax(img, gt, ft_res, label, ax[1, 1:])

    # Adjust the margins
    plt.subplots_adjust(left=0.005, right=0.97, top=0.95, bottom=0.01, hspace=0.1)
    plt.savefig(output_path)
    plt.close()
    