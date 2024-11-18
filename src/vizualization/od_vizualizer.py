import argparse
import os
import pandas as pd

from vizualization.vizualization_utils import print_img
from llms.extract_text_methods import get_page_content_table_value
from tqdm.auto import tqdm


HTMLS_DIR = "src/vizualization/htmls"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--df_path",
        type=str,
        default="data/clean_csv/dataset.csv",
        help="path to df that contains our dataset",
    )
    parser.add_argument("--html_fn", type=str, default="data_with_od_and_aug.html")
    parser.add_argument("--html_title", type=str, default="Data with object detection with augmentation")
    parser.add_argument("--od_images_dir", type=str, default="data/od_images/all_with_scale_more_iters")
    return parser.parse_args()


def print_metadata(df_row, html_out):
    print("<h4>Wiki Commons description (caption)</h4>", file=html_out)
    description = get_page_content_table_value(df_row["page_content"], "Description")
    print(description.replace("\n", "<br>"), file=html_out)
    print(f"<br><b></b><br>", file=html_out)


def print_row(df_row, html_out):
    print(f"<h3> {df_row['fn']}: </h3>", file=html_out)
    print(
        f"<h4>OD marked image:</h4>",
        file=html_out,
    )
    print_img(df_row["od_fn"], html_out)


def print_data(df, html_out, sample_size=None):
    print(f"<h2>Images ({len(df)} in total)</h2>", file=html_out)
    if sample_size:
        df = df.sample(sample_size)

    for _, row in tqdm(df.iterrows(), total=len(df)):
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
    tqdm.pandas()
    df = pd.read_csv(args.df_path)

    od_imgs_dir = args.od_images_dir

    df["od_fn"] = df.progress_apply(
        lambda row: os.path.join(od_imgs_dir, f'{row["page_id"]}.png')
        if os.path.exists(os.path.join(od_imgs_dir, f'{row["page_id"]}.png'))
        else pd.NA,
        axis=1,
    )

    df_with_od = df[~df["od_fn"].isna()]

    print(f"<h1>{args.html_title}</h1>", file=html_out)
    print_data(df_with_od, html_out, sample_size=100)

    print("<hr>", file=html_out)
    html_out.close()


if __name__ == "__main__":
    data_visualization_html()
