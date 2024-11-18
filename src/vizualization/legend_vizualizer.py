import argparse
import json
import os
import pandas as pd

from legends.legend_image_renderer import COL_NAME_TO_LEGEND_DIR
from legends.legend_utils import format_keys
from ocr.models.gcp.ocr_texts import OCRTexts
from googletrans import Translator
from vizualization.vizualization_utils import (
    print_img,
    print_ocr_texts,
)
from llms.extract_text_methods import get_page_content_table_value
from tqdm.auto import tqdm


HTMLS_DIR = "src/vizualization/htmls"

translator = Translator()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--df_path",
        type=str,
        default="data/csvs_v2/clean/dataset.csv",
        help="path to df that contains our dataset",
    )
    parser.add_argument(
        "--legends_col",
        type=str,
        default="legend_fn",
        help="the column containing the legend file names",
    )
    parser.add_argument("--html_fn", type=str, default="data_with_legends.html")
    parser.add_argument("--html_title", type=str, default="Data with legends")
    return parser.parse_args()


def print_metadata(df_row, html_out):
    print("<h4>Wiki Commons description (caption)</h4>", file=html_out)
    description = get_page_content_table_value(df_row["page_content"], "Description")
    print(description.replace("\n", "<br>"), file=html_out)
    print(f"<br><b></b><br>", file=html_out)


def print_legend(legend_fn, title, html_out):
    with open(legend_fn, "r") as legend_file:
        if legend_fn.endswith(".json"):
            print(
                f"<h4>{title}</h4>",
                file=html_out,
            )
            legend = json.load(legend_file)
            for key, value in legend.items():
                print(f"<b>{key}</b>: {value}<br>", file=html_out)
        print(f"<br><b></b><br>", file=html_out)


def print_found_legend_in_ocr(df_row, html_out):
    with open(df_row["legend_fn"], "r") as legend_file:
        legend = json.load(legend_file)
    ocr_texts = OCRTexts(df_row["ocr_fn"])
    found_keys = ocr_texts.find_texts(format_keys(legend.keys()))
    print(f"<h4>Legend found in OCR</h4>", file=html_out)
    found_legend = {}
    for key in [key.text for key in found_keys]:
        if key in legend:
            found_legend[key] = legend[key]
        elif key.lower() in legend:
            found_legend[key] = legend[key.lower()]
        elif key.upper() in legend:
            found_legend[key] = legend[key.upper()]
    for key, value in found_legend.items():
        print(f"<b>{key}</b>: {value}<br>", file=html_out)
    print(f"<br><b></b><br>", file=html_out)
    print(f"<h4>Legend found in OCR (translated)</h4>", file=html_out)
    for key, value in found_legend.items():
        print(
            f"<b>{key}</b>: {translator.translate(value, dest='en').text}<br>",
            file=html_out,
        )
    print(f"<br><b></b><br>", file=html_out)


def print_features(features_fn, html_out):
    with open(features_fn, "r") as features_file:
        print(
            f"<h4>Architectural features</h4>",
            file=html_out,
        )
        features = features_file.read()
        print(features.replace("\n", "<br>"), file=html_out)
        print(f"<br><b></b><br>", file=html_out)


def print_captions(captions_fn, building_type, html_out):
    with open(captions_fn, "r") as captions_file:
        print(
            f"<h4>Captions for building type: {building_type}</h4>",
            file=html_out,
        )
        captions = captions_file.read()
        print(captions.replace("\n", "<br>"), file=html_out)
        print(f"<br><b></b><br>", file=html_out)


def print_row(df_row, html_out):
    print(f"<h3> {df_row['fn']}: </h3>", file=html_out)
    print(
        f"<h4>Has legend according to llm: {df_row['has_ocr_legend_v3']}</h4>",
        file=html_out,
    )
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
    # print_metadata(df_row, html_out)
    print_legend(df_row["legend_fn"], "Formatted legend", html_out)
    print_found_legend_in_ocr(df_row, html_out)
    # captions_fn = df_row["captions_fn"]
    # if os.path.exists(captions_fn):
    #     print_captions(captions_fn, df_row["building_type"], html_out)
    # if isinstance(df_row["wiki_features_fn"], str):
    #     print_features(df_row["wiki_features_fn"], html_out)
    # print_ocr_texts(df_row, html_out, confidence_threshold=0)


def print_data(df, html_out):
    print(f"<h2>Images ({len(df)} in total)</h2>", file=html_out)

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

    df["legend_fn"] = df.progress_apply(
        lambda row: f'data/outputs/legends_outputs/from_ocr/{row["page_id"]}.json'
        if os.path.exists(
            f'data/outputs/legends_outputs/from_ocr/{row["page_id"]}.json'
        )
        else pd.NA,
        axis=1,
    )
    df["legend_img_path"] = df.progress_apply(
        lambda row: f'data/legend_images/from_ocr/{row["page_id"]}.png'
        if os.path.exists(f'data/legend_images/from_ocr/{row["page_id"]}.png')
        else pd.NA,
        axis=1,
    )

    col_name = args.legends_col
    df_with_legend = df[(~df[col_name].isna()) & ~(df["legend_img_path"].isna())]
    # df_with_legend["formatted_legend_fn"] = df_with_legend[col_name].apply(
    #     lambda x: x.replace(".txt", ".json")
    # )
    # df_with_legend["marked_legend_img_path"] = df_with_legend["page_id"].apply(
    #     lambda x: os.path.join(COL_NAME_TO_LEGEND_DIR[col_name], f"{x}.png")
    # )
    # df_with_legend["captions_fn"] = df_with_legend["page_id"].apply(
    #     lambda x: f"data/outputs/captions_outputs/{x}.txt"
    # )
    # df_with_legend = df_with_legend[
    #     (df_with_legend.building_type.str.contains("castle"))
    #     | (df_with_legend.building_type.str.contains("palace"))
    # ]
    # df_with_legend = df_with_legend[~df_with_legend.legend_fn.str.contains("from_ocr")]
    df_with_legend = df_with_legend.sample(80)

    print(f"<h1>{args.html_title}</h1>", file=html_out)
    print_data(df_with_legend, html_out)

    print("<hr>", file=html_out)
    html_out.close()


if __name__ == "__main__":
    data_visualization_html()
