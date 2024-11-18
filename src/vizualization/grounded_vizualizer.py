import argparse
import json
import os
import pickle
import pandas as pd
from grounded_legend_vizualizer import print_and_sample, print_grounded_legends, print_legend_row

from legends.legend_utils import format_keys
from ocr.models.gcp.ocr_texts import OCRTexts
from googletrans import Translator
from vizualization.vizualization_utils import (
    print_img,
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
        default="data/clean_csv/dataset.csv",
        help="path to df that contains our dataset",
    )
    parser.add_argument("--html_fn", type=str, default="data_with_grounded_info.html")
    parser.add_argument(
        "--html_title",
        type=str,
        default="Data with grounded legends and architectural features",
    )
    return parser.parse_args()


def print_arc_feats(simplified_arc_feats_fn, html_out):
    grounded_arc_feats = json.load(open(simplified_arc_feats_fn, "rb"))
    print(f"<h4>LLM output grounded arc feats:</h4>", file=html_out)
    print(f"<b>{', '.join(grounded_arc_feats.keys())}</b><br>", file=html_out)
    
    print(f"<br><b></b><br>", file=html_out)



def print_arc_feats_row(df_row, html_out):
    print(f"<h3> {df_row['fn']}: </h3>", file=html_out)
    if os.path.exists(df_row["od_img_path"]):
        print(
            f"<h4>OD Image:</h4>",
            file=html_out,
        )
        print_img(df_row["od_img_path"], html_out)
    
    grounded_path = f"data/legend_images/arc_feats_grounded/{df_row['page_id']}.png"
    print(
        f"<h4>Grounded image:</h4>",
        file=html_out,
    )
    print_img(grounded_path, html_out)
    print_arc_feats(df_row["simplified_arc_feats_fn"], html_out)

def print_all_grounded_row(df_row, html_out):
    if isinstance(df_row.grounded_legend_fn, str) and os.path.exists(df_row.grounded_legend_fn):
        print_legend_row(df_row, html_out)
    else:
        print_arc_feats_row(df_row, html_out)


def print_grounded_arc_feats(df, html_out):
    print(f"<h2>Grounded Images ({len(df)} in total)</h2>", file=html_out)
    print_and_sample(df, print_row_func=print_arc_feats_row, html_out=html_out)
    
    
def print_all_grounded_by_building_type(df, html_out):
    print(f"<h2>Grounded Images ({len(df)} in total)</h2>", file=html_out)
    counts = df['building_type'].value_counts()[:20]
    for value, count in counts.items():
        print(f"<h2>Building type: {value} ({count} in total) </h2>", file=html_out)
        print_and_sample(df[df.building_type == value], print_row_func=print_all_grounded_row, samples=5, html_out=html_out)
        
    


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
        + "h2 {background-color: #FFFF00;} h3 {background-color: #C6F4D9;} h4 {background-color: #FFC0CB;} h5 {background-color: #FFFF00;}</style>",
        file=html_out,
    )
    tqdm.pandas()
    df = pd.read_csv(args.df_path)
    df["od_img_path"] = df["page_id"].progress_apply(
        lambda id: f"data/od_images/all_with_scale_more_iters/{id}.png"
    )
    df_grounded = df[df["grounded_legend_fn"].notna() | df["simplified_arc_feats_fn"].notna()]

    print(f"<h1>{args.html_title}, total of {len(df_grounded)} grounded images</h1>", file=html_out)
    print_all_grounded_by_building_type(df_grounded, html_out)

    print("<hr>", file=html_out)
    html_out.close()


if __name__ == "__main__":
    data_visualization_html()
