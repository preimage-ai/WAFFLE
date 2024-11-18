import argparse
import json
import os
import pickle
import textwrap
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from legends.legend_utils import get_ocr_found_legend_item_to_ocr_boxes
from googletrans import Translator
from depr_plot_utils import plot_data
from vizualization.vizualization_utils import print_img
from vlms.clipseg.seg_data import CLIPSegData, CLIPSegGT
from tqdm.auto import tqdm


HTMLS_DIR = "src/vizualization/htmls"

translator = Translator()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_fn",
        type=str,
        default="data/for_segmentation/churches_dataset_v3.pkl",
        help="path to df that contains our dataset",
    )
    parser.add_argument("--html_fn", type=str, default="data_for_segmentation.html")
    parser.add_argument(
        "--html_title", type=str, default="Data for segmentation with legends"
    )
    return parser.parse_args()


def plot_with_gt(data, output_path):
    if os.path.exists(output_path):
        return
    _, ax = plt.subplots(len(data.gts), 2, figsize=(10, 5 * len(data.gts)))
    [a.axis("off") for a in ax.flatten()]
    if len(data.gts) == 1:
        ax = ax.reshape(1, 2)
    for i, gt in enumerate(data.gts):
        ax[i][0].imshow(data.img)
        gt_combined = np.zeros((data.img.size[1], data.img.size[0]))
        gt_combined[gt.gt > 0] = 1
        gt_combined[gt.gt_neg > 0] = -1
        
        # Create a custom color map
        cmap = mcolors.ListedColormap(['red', 'black', 'green'])
        bounds = [-1., 0., 1., 2.]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Use the custom color map to plot gt_combined
        ax[i][1].imshow(gt_combined, cmap=cmap, norm=norm)
        ax[i][1].text(0, -15, '\n'.join(textwrap.wrap(f"GT: {gt.labels}", 30)))  # Wrap the text
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


def print_row(df_row, html_out):
    print(f"<h3> {df_row['fn']}: </h3>", file=html_out)
    print(
        f"<h4>Image:</h4>",
        file=html_out,
    )
    print_img(df_row["img_url"], html_out)
    print(
        f"<h4>Legend marked image:</h4>",
        file=html_out,
    )
    print_img(df_row["legend_img_path"], html_out)
    print(
        f"<h4>GT:</h4>",
        file=html_out,
    )
    print_img(f"data/for_segmentation/gt_plots/{df_row['page_id']}.png", html_out)
    print_metadata(df_row, html_out)
    print_legend(df_row["simplified_legend_fn"], "Formatted legend", html_out)
    print_found_legend_in_ocr(df_row, html_out)


def print_data(dataset, html_out):
    print(f"<h2>Images ({len(dataset)} in total)</h2>", file=html_out)
    for data in tqdm(dataset):
        plot_data(data, f"data/for_segmentation/gt_plots/{data.df_row['page_id']}.png")
    df_rows = [data.df_row for data in dataset]

    for row in tqdm(df_rows):
        print_row(row, html_out)


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
    print_data(dataset, html_out)

    print("<hr>", file=html_out)
    html_out.close()


if __name__ == "__main__":
    data_visualization_html()
