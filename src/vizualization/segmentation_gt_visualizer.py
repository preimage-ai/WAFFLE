import argparse
import json
import os
import pickle
import pandas as pd

from vlms.clipseg.v2.data import CLIPSegDatum, CLIPSegGT
from vlms.clipseg.v2.plot_utils import plot_gt, plot_datum
from googletrans import Translator
from vizualization.vizualization_utils import print_img
import random
from tqdm.auto import tqdm


HTMLS_DIR = "src/vizualization/htmls"

translator = Translator()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/for_segmentation/churches_dataset.pkl",
        help="path to segmentation dataset",
    )
    parser.add_argument(
        "--html_fn", type=str, default="churches_segmentation_dataset.html"
    )
    parser.add_argument("--html_title", type=str, default="Churches Segmentation Dataset")
    return parser.parse_args()


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
    print(f"<h1>{args.html_title}</h1>", file=html_out)
    
    test_dataset = pickle.load(open(args.dataset_path.replace(".pkl", "_test.pkl"), "rb"))
    train_dataset = pickle.load(open(args.dataset_path.replace(".pkl", "_train.pkl"), "rb"))
    dataset = test_dataset + train_dataset
    print(f"<h2>Churches segmentation GT ({dataset} in total)</h2>", file=html_out)
    
    sample = random.sample(dataset, 50)
    for datum in tqdm(sample):
        print(f"<h3> {datum.df_row['fn']}: </h3>", file=html_out)
        print_img(datum.df_row["img_path"], html_out)
        print(
            f"<h4>GTs:</h4>",
            file=html_out,
        )
        plot_datum(datum, "a.png")
        print_img("a.png", html_out)
        

    

    print("<hr>", file=html_out)
    html_out.close()


if __name__ == "__main__":
    data_visualization_html()
