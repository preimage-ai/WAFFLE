import argparse
import os
import pandas as pd
import json
import pickle

from tqdm.auto import tqdm
from PIL import Image
from legends.legend_utils import format_keys
from ocr.models.gcp.rendering_helper import draw_box_with_text, draw_boxes_with_texts
from ocr.models.gcp.ocr_texts import OCRTexts
from legends.grounded_legend import GroundedLegend, GroundedLegendValue
from google.cloud.vision_v1.types import BoundingPoly


COL_NAME_TO_LEGEND_DIR = {
    "caption_legend_fn": "data/legend_images/from_caption",
    "wiki_legend_fn": "data/legend_images/from_wiki",
    "ocr_legend_v3_fn": "data/legend_images/from_ocr",
    "legend_fn": "data/legend_images/unified",
    "simplified_legend_fn": "data/legend_images/simplified",
}


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
        default="caption_legend_fn",
        help="the column containing the legend file names",
    )
    return parser.parse_args()


def format_texts(texts):
    formatted_texts = set([text.strip() for text in texts])
    for text in texts:
        if text.isdigit():
            # sometimes number keys can have a leading 0
            formatted_texts.add(str(int(text)))
        if text.isalpha():
            # sometimes text keys can be mislead for upper or lower case
            formatted_texts.add(text.upper())
            formatted_texts.add(text.lower())
    return formatted_texts


def render_image_with_legend(df_row, legend_col_name, output_path):
    # legend_json_path = df_row[legend_col_name].replace(".txt", ".json")
    legend_json_path = f'data/outputs/legends_outputs/from_ocr/{df_row["page_id"]}.json'
    if not os.path.exists(legend_json_path):
        # print(f"Legend json not found for {legend_json_path}")
        return pd.NA
    if not isinstance(df_row["ocr_fn"], str):
        print(f"OCR not found for {df_row['ocr_fn']}")
        return pd.NA
    with open(legend_json_path) as f:
        legend = json.load(f)
    draw_boxes_with_texts(
        texts=format_texts(legend.keys()),
        document_fn=df_row["ocr_fn"],
        img_path=df_row["img_path"],
        output_path=output_path,
    )
    return output_path


def draw_grounded_legend(grounded_legend, img_path, output_path):
    image = Image.open(img_path)
    image = image.convert("RGB")
    for key, value in grounded_legend.legend.items():
        for box in value.ocr_boxes:
            image = draw_box_with_text(
                image,
                bound=box,
                text=key,
                box_color="red",
                text_color="black",
                text_background_color="cyan",
            )
    image.save(output_path)
    return output_path


def draw_grounded_arc_feats(grounded_arc_feats, img_path, output_path):
    image = Image.open(img_path)
    image = image.convert("RGB")
    for value in grounded_arc_feats.values():
        for ocr_text in value['ocr_texts']:
            image = draw_box_with_text(
                image,
                bound=ocr_text['ocr_box'],
                text=ocr_text['text'],
                box_color="green",
                text_color="black",
                text_background_color="cyan",
            )
    image.save(output_path)
    return output_path


def render_image_with_grounded_legend(df_row):
    grounded_legend_path = (
        f"data/outputs/legends_outputs/grounded_legends/{df_row['page_id']}.pkl"
    )
    if not os.path.exists(grounded_legend_path):
        return pd.NA
    with open(grounded_legend_path, "rb") as f:
        grounded_legend = pickle.load(f)
    draw_grounded_legend(
        grounded_legend=grounded_legend,
        img_path=df_row["img_path"],
        output_path=f"data/legend_images/grounded_legends/{df_row['page_id']}.png",
    )


def render_image_with_grounded_arc_feats(df_row):
    grounded_arc_feats_path = (
        f"data/outputs/legends_outputs/simplified_arc_feats/{df_row['page_id']}.json"
    )
    if not os.path.exists(grounded_arc_feats_path):
        return pd.NA
    with open(grounded_arc_feats_path, "rb") as f:
        grounded_arc_feats = json.load(f)
    draw_grounded_arc_feats(
        grounded_arc_feats=grounded_arc_feats,
        img_path=df_row["img_path"],
        output_path=f"data/legend_images/arc_feats_grounded/{df_row['page_id']}.png",
    )


def get_center(box):
    x_coords = [vertex.x for vertex in box.vertices]
    y_coords = [vertex.y for vertex in box.vertices]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    return int(center_x), int(center_y)


def get_legend_values_and_positions(legend_fn, ocr_fn):
    ocr_texts = OCRTexts(ocr_fn)
    legend_json_path = legend_fn.replace(".txt", ".json")
    if not os.path.exists(legend_json_path):
        print(f"Legend json not found for {legend_json_path}")
        return pd.NA
    legend = json.load(open(legend_json_path))
    found_texts = ocr_texts.find_texts(format_keys(legend.keys()))
    for_prompt = {}
    for found_text in found_texts:
        for_prompt[legend[found_text.text]] = get_center(found_text.bounding_box)
    return for_prompt


def main():
    tqdm.pandas()
    args = get_args()
    df = pd.read_csv('data/clean_csv/dataset.csv')
    
    row = df[df.fn.str.contains("GrundrissSchlossSchellenberg")].iloc[0]
    render_image_with_grounded_legend(row)

    df.progress_apply(
        lambda row: render_image_with_grounded_legend(row),
        axis=1,
    )

    # legend_col_name = args.legends_col
    # legend_col_name = "ocr_legend_v3_fn"
    # imgs_with_legend = df[df[legend_col_name].notna()]
    # df["legend_img_path"] =
    # df.progress_apply(
    #     lambda row: (
    #         render_image_with_legend(
    #             df_row=row,
    #             legend_col_name=legend_col_name,
    #             output_path=os.path.join(
    #                 COL_NAME_TO_LEGEND_DIR[legend_col_name], f"{row['page_id']}.png"
    #             ),
    #         )
    #         if not os.path.exists(
    #             os.path.join(
    #                 COL_NAME_TO_LEGEND_DIR[legend_col_name], f"{row['page_id']}.png"
    #             )
    #         )
    #         else pd.NA
    #     ),
    #     # if isinstance(row[legend_col_name], str)
    #     # else df["legend_img_path"],
    #     axis=1,
    # )
    # pass


if __name__ == "__main__":
    main()
