import argparse
import json
import os
import pickle
import textwrap
from matplotlib import pyplot as plt
from vlms.clipseg.clipseg_dataset import CLIPSegDatum, CLIPSegGT, create_mask

from legends.legend_utils import get_ocr_found_legend_item_to_ocr_boxes
from vlms.clipseg.plot_utils import show_boolean_mask
from vizualization.vizualization_utils import print_img
from tqdm.auto import tqdm


HTMLS_DIR = "src/vizualization/htmls"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_fn",
        type=str,
        default="data/for_segmentation/churches_dataset_dynamic_boxes.pkl",
        help="path to df that contains our dataset",
    )
    parser.add_argument("--html_fn", type=str, default="data_for_segmentation_new.html")
    parser.add_argument(
        "--html_title", type=str, default="Data for segmentation with legends"
    )
    return parser.parse_args()



def plot_gt(img, gt, output_path):
    if os.path.exists(output_path):
        return
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    ax.text(0, -15, '\n'.join(textwrap.wrap(f"GT: {gt.labels}", 30)))
    gt_mask = create_mask(img, gt.boxes)
    mask = create_mask(img, gt.boxes + gt.neg_boxes)
    show_boolean_mask(gt_mask, ax, color="green")
    neg_mask = mask.astype(bool) & ~gt_mask.astype(bool)
    show_boolean_mask(neg_mask, ax, color="red")
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
        f"<h4>Legend marked image:</h4>",
        file=html_out,
    )
    print_img(data.df_row["legend_img_path"], html_out)
    print_legend(data.df_row["simplified_legend_fn"], "Formatted legend", html_out)
    print_found_legend_in_ocr(data.df_row, html_out)
    print(
        f"<h4>GT:</h4>",
        file=html_out,
    )
    # for i, gt in enumerate(data.gts):
    #     plot_path = f"data/for_segmentation/gt_plots/{data.df_row['page_id']}_{i}.png"
    #     plot_gt(data.img, gt, plot_path)
    #     print_img(plot_path, html_out)
    print_metadata(data.df_row, html_out)


def print_dataset(dataset, html_out):
    print(f"<h2>Images ({len(dataset)} in total)</h2>", file=html_out)
    # for data in tqdm(dataset):
    #     plot_data(data, f"data/for_segmentation/gt_plots/{data.df_row['page_id']}.png")
    # df_rows = [data.df_row for data in dataset]

    for data in tqdm(dataset):
        print_data(data, html_out)


def data_visualization_html():
    args = get_args()
    html_out = open(
        # os.path.join(HTMLS_DIR, args.html_fn),
        os.path.join(HTMLS_DIR, "castle_data_for_segmentation.html"),
        "w",
        encoding="utf-8",
    )
    print('<head><meta charset="UTF-8"></head>', file=html_out)
    print(
        "<style>table, th, td {border:1px solid black;} "
        + "h3 {background-color: #C6F4D9;} h4 {background-color: #FFC0CB;} h5 {background-color: #FFFF00;}</style>",
        file=html_out,
    )

    # with open(args.dataset_fn, "rb") as f:
    with open('data/for_segmentation/castles_dataset_dynamic_boxes.pkl', "rb") as f:
        dataset = pickle.load(f)

    # print(f"<h1>{args.html_title}</h1>", file=html_out)
    print(f"<h1>Castles data for segmentation with legends</h1>", file=html_out)
    print_dataset(dataset, html_out)

    print("<hr>", file=html_out)
    html_out.close()


if __name__ == "__main__":
    data_visualization_html()
