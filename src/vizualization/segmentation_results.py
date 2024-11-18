import argparse
import json
import os
import pickle
import textwrap
import cv2
from matplotlib import pyplot as plt
from vlms.clipseg.clipseg_dataset import CLIPSegDatum, CLIPSegGT, create_mask

from legends.legend_utils import get_ocr_found_legend_item_to_ocr_boxes
from vlms.clipseg.plot_utils import center_crop_res, show_pred
from vlms.clipseg.sizing_utils import resize_array
from vizualization.vizualization_utils import print_img
from tqdm.auto import tqdm


HTMLS_DIR = "src/vizualization/htmls"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_fn",
        type=str,
        default="data/for_segmentation/churches_dataset_gt_for_metrics_evaluation_5_epochs.pkl",
        # default="data/for_segmentation/castles_dataset_gt_for_metrics_evaluation.pkl",
        help="path to df that contains our dataset",
    )
    parser.add_argument("--html_fn", type=str, default="churches_segmentation_5_epochs.html")
    # parser.add_argument("--html_fn", type=str, default="castles_segmentation.html")
    parser.add_argument(
        "--html_title", type=str, default="Results from churches segmentation"
    )
    return parser.parse_args()

def crop_and_resize_res(img, res):
    cropped_res = center_crop_res(img, res)
    # Resize cropped_res to the size of img
    return cv2.resize(
        cropped_res,
        (img.size[0], img.size[1]),
        interpolation=cv2.INTER_LINEAR,
    )

def plot_res(img, res, label, output_path):
    _, ax = plt.subplots(1, 2, figsize=(20, 10))
    [a.axis("off") for a in ax.flatten()]
    res = crop_and_resize_res(img, res)
    ax[0].imshow(res, cmap="RdYlGn")
    ax[0].text(
        0, -15, "\n".join(textwrap.wrap(f"Pred: {label}", 30)), fontsize=20
    )  # Wrap the text
    ax[1].imshow(img)
    ax[1].text(
        0,
        -15,
        "\n".join(textwrap.wrap(f"Pred mask for label: {label}", 30)),
        fontsize=20,
    )  # Wrap the text
    show_pred(res, ax[1])
    plt.savefig(output_path)
    plt.close()


def print_metadata(df_row, html_out):
    print("<h4>Image info</h4>", file=html_out)
    print(f"<b>Building type:</b> {df_row['building_type']}<br>", file=html_out)
    print(f"<b>Country:</b> {df_row['country']}<br>", file=html_out)
    print(f"<br><b></b><br>", file=html_out)


def print_legend(legend_fn, title, html_out):
    with open(legend_fn, "r") as legend_file:
        legend = json.load(legend_file)
    print(
        f"<h4>{title}</h4>",
        file=html_out,
    )
    for key, value in legend.items():
        print(f"<b>{key}</b>: {value}<br>", file=html_out)
    print(f"<br><b></b><br>", file=html_out)


def print_found_legend_in_ocr(df_row, html_out):
    found_legend = get_ocr_found_legend_item_to_ocr_boxes(
        df_row["simplified_legend_fn"], df_row["ocr_fn"]
    )
    print(f"<h4>Legend found in OCR</h4>", file=html_out)
    for key, value in found_legend.items():
        print(f"<b>{key}</b>: {value}<br>", file=html_out)
    print(f"<br><b></b><br>", file=html_out)


def print_data(data, html_out):
    print(f"<h3> {data.df_row['fn']}: </h3>", file=html_out)
    print(
        f"<h4>Image:</h4>",
        file=html_out,
    )
    print_img(data.df_row["img_url"], html_out)
    print(
        f"<h4>Inpainted image:</h4>",
        file=html_out,
    )
    inpainted_path = f"data/for_segmentation/inpainted_images/{data.df_row['page_id']}.png"
    if not os.path.exists(inpainted_path):
        data.img.save(inpainted_path)
    print_img(inpainted_path, html_out)
    print(
        f"<h4>Predictions:</h4>",
        file=html_out,
    )
    for i, gt in enumerate(data.gts):
        plot_path = (
            f"data/for_segmentation/pred_plots/gt_{data.df_row['page_id']}_{i}.png"
        )
        # if not os.path.exists(plot_path):
        plot_res(data.img, resize_array(gt.gt_mask_, 512), gt.labels[0], plot_path)
        print(
            f"<h5>GT:</h5>",
            file=html_out,
        )
        print_img(plot_path, html_out)
        
        plot_path = (
            f"data/for_segmentation/pred_plots/base_{data.df_row['page_id']}_{i}.png"
        )
        # if not os.path.exists(plot_path):
        plot_res(data.img, gt.base_res_, gt.labels[0], plot_path)
        print(
            f"<h5>Base:</h5>",
            file=html_out,
        )
        print_img(plot_path, html_out)
        
        plot_path = (
            f"data/for_segmentation/pred_plots/ft_{data.df_row['page_id']}_{i}.png"
        )
        # if not os.path.exists(plot_path):
        plot_res(data.img, gt.ft_res_, gt.labels[0], plot_path)
        print(
            f"<h5>FT:</h5>",
            file=html_out,
        )
        print_img(plot_path, html_out)
    print_metadata(data.df_row, html_out)


def print_dataset(dataset, html_out):
    print(f"<h2>Images ({len(dataset)} in total)</h2>", file=html_out)
    for data in tqdm(dataset):
        print_data(data, html_out)


def data_visualization_html():
    args = get_args()
    html_out = open(
        os.path.join(HTMLS_DIR, args.html_fn),
        "w",
        encoding="utf-8",
    )
    print('<head><meta charset="UTF-8"></head>', file=html_out)
    print(
        "<style>table, th, td {border:1px solid black;} "
        + "h3 {background-color: #C6F4D9;} h4 {background-color: #FFC0CB;} h5 {background-color: #FFFF00;}</style>",
        file=html_out,
    )

    with open(args.dataset_fn, "rb") as f:
        dataset = pickle.load(f)

    print(f"<h1>{args.html_title}</h1>", file=html_out)
    # print(f"<h1>Castles data for segmentation with legends</h1>", file=html_out)
    print_dataset(dataset, html_out)

    print("<hr>", file=html_out)
    html_out.close()


if __name__ == "__main__":
    data_visualization_html()
