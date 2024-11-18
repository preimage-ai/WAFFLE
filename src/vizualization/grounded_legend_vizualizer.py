import argparse
import json
import os
import pickle
import pandas as pd

from legends.legend_utils import format_keys
from ocr.models.gcp.ocr_texts import OCRTexts
from googletrans import Translator
from vizualization.vizualization_utils import print_img
from llms.extract_text_methods import get_page_content_table_value
from tqdm.auto import tqdm


HTMLS_DIR = "src/vizualization/htmls"

translator = Translator()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--df_path",
        type=str,
        default="data/clean_csv/dataset.csv",
        help="path to df that contains our dataset",
    )
    parser.add_argument(
        "--html_fn", type=str, default="data_with_grounded_legends.html"
    )
    parser.add_argument("--html_title", type=str, default="Data with grounded legends")
    return parser.parse_args()


def print_legend(legend, html_out):
    for key, value in legend.items():
        print(f"<b>{key}</b>: {value}<br>", file=html_out)
    print(f"<br><b></b><br>", file=html_out)


def get_legend_source(legend_fn):
    if "ocr" in legend_fn:
        return "ocr"
    if "metadata" in legend_fn:
        return "metadata"
    return "wiki"


def print_legend_row(df_row, html_out):
    print(f"<h3> {df_row['fn']}: </h3>", file=html_out)
    print(
        f"<h4>OD Image:</h4>",
        file=html_out,
    )

    print_img(df_row["od_img_path"], html_out)
    grounded_path = f"data/legend_images/grounded_legends/{df_row['page_id']}.png"
    legend_source = get_legend_source(df_row["legend_fn_v2"])
    if os.path.exists(grounded_path):
        print(
            f"<h4>Grounded image:</h4>",
            file=html_out,
        )
        print_img(grounded_path, html_out)
        print(
            f"<h4>Grounded legend (source: {legend_source}):</h4>",
            file=html_out,
        )
        grounded_legend = pickle.load(open(df_row["grounded_legend_fn"], "rb"))
        key_val_legend = {
            key: value.value for key, value in grounded_legend.legend.items()
        }
        print_legend(key_val_legend, html_out)
    # print(
    #     f"<h4>LLM output formatted legend ({legend_source}):</h4>",
    #     file=html_out,
    # )
    with open(df_row["legend_fn_v2"], "r") as legend_file:
        legend = json.load(legend_file)
        print_legend(legend, html_out)


def print_and_sample(df, html_out, print_row_func, samples=50):
    sampled = df.sample(samples)
    for _, row in tqdm(sampled.iterrows(), total=len(sampled)):
        print_row_func(row, html_out)


def print_grounded_legends(df, html_out):
    grounded = df[df["grounded_legend_fn"].notna()]
    print(f"<h2>Grounded legends Images ({len(grounded)} in total)</h2>", file=html_out)
    print_and_sample(grounded, print_row_func=print_legend_row, html_out=html_out)


def print_ungrounded_legends(df, html_out):
    ungrounded = df[df["grounded_legend_fn"].isna()]
    print(f"<h2>Ungrounded legend Images ({len(df)} in total)</h2>", file=html_out)
    print_and_sample(ungrounded, print_row_func=print_legend_row, html_out=html_out)


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

    df_with_legend = df[df["legend_fn_v2"].notna()]
    df_with_legend["od_img_path"] = df["page_id"].progress_apply(
        lambda id: f"data/od_images/all_with_scale_more_iters/{id}.png"
    )

    print(f"<h1>{args.html_title}</h1>", file=html_out)
    print_grounded_legends(df_with_legend, html_out)
    print_ungrounded_legends(df_with_legend, html_out)

    print("<hr>", file=html_out)
    html_out.close()


if __name__ == "__main__":
    data_visualization_html()
