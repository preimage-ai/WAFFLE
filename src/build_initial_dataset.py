import argparse
import os
import pandas as pd

from tqdm.auto import tqdm
from images.images_downloader import download_images
from ocr.models.gcp.vision_api_inference import GcpVisionInference
from ocr.models.gcp.ocr_texts import OCRTexts
from vlms.clip.clip_inference import ClipInference
from vlms.clip.categories import (
    NARROW_POSITIVE_SCORES,
    POSITIVE_CATEGORIES,
    NEGATIVE_CATEGORIES,
    get_highest_clip_category,
    get_score_for_categories,
    has_top_n_categories,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/scraped_data/with_fields.csv",
        help="path to scraped wiki-common images",
    )
    parser.add_argument(
        "--all_data_df_path",
        type=str,
        default="data/all_data.csv",
        help="path to df that contains the initial dataset",
    )
    parser.add_argument(
        "--img_output_dir",
        type=str,
        default="data/images",
        help="path to scraped wiki-common images",
    )
    parser.add_argument(
        "--ocr_output_dir",
        type=str,
        default="data/outputs/ocr_outputs",
        help="path to store extracted OCR texts",
    )
    parser.add_argument(
        "--clip_scores_output_dir",
        type=str,
        default="data/outputs/clip_scores_output",
        help="path to store CLIP category scores",
    )
    return parser.parse_args()


def create_initial_dataset(scraped_data_fn, img_url_to_path_fn, output_df_path):
    if os.path.exists(output_df_path):
        return pd.read_csv(output_df_path)

    scraped_df = pd.read_csv(scraped_data_fn)
    df = pd.DataFrame()
    df["page_id"] = scraped_df["pageid"].astype(int)
    df["category"] = scraped_df["subcat"]
    df["page_content"] = scraped_df["mw-imagepage-content"]
    df["fn"] = scraped_df["title"]
    df["img_url"] = scraped_df["img_fn"]

    img_url_to_path = pd.read_csv(img_url_to_path_fn)
    df = df[df.img_url.isin(img_url_to_path.url)]
    df["img_path"] = df.img_url.progress_apply(
        lambda url: img_url_to_path[img_url_to_path.url == url]["path"].values[0]
    )

    print(f"Saving initial dataset to {output_df_path}")
    df.to_csv(output_df_path, index=False)
    return df


def add_ocr_columns(df, output_dir, output_df_path):
    gcp_inference = GcpVisionInference()
    df["ocr_fn"] = df.progress_apply(
        lambda row: gcp_inference.store_document_text_detection(
            img_path=row["img_path"],
            output_path=os.path.join(output_dir, f"{row['page_id']}.json"),
        )
        if isinstance(row["img_path"], str)
        else pd.NA,
        axis=1,
    )
    df["ocr_texts"] = df["ocr_fn"].progress_apply(
        lambda fn: OCRTexts(fn).get_texts() if isinstance(fn, str) else pd.NA
    )
    print(f"Saving dataset with ocr to {output_df_path}")
    df.to_csv(output_df_path, index=False)
    return df


def add_clip_scores_column(df, output_dir, output_df_path):
    clip_inference = ClipInference()
    df["clip_scores_fn"] = df.progress_apply(
        lambda row: clip_inference.store_clip_scores(
            image_path=row["img_path"],
            categories_list=POSITIVE_CATEGORIES + NEGATIVE_CATEGORIES,
            output_path=os.path.join(output_dir, f"{row['page_id']}.json"),
        ),
        axis=1,
    )
    df["wide_clip_score"] = df["clip_scores_fn"].progress_apply(
        lambda fn: get_score_for_categories(fn, POSITIVE_CATEGORIES)
    )
    df["narrow_clip_score"] = df["clip_scores_fn"].progress_apply(
        lambda fn: get_score_for_categories(fn, NARROW_POSITIVE_SCORES)
    )
    df["top_5_cat_are_floorplan"] = df["clip_scores_fn"].progress_apply(
        lambda fn: has_top_n_categories(fn, NARROW_POSITIVE_SCORES)
    )
    df["highest_clip_category"] = df["clip_scores_fn"].progress_apply(
        lambda fn: get_highest_clip_category(fn)
    )
    print(f"Saving dataset with clip scores to {output_df_path}")
    df.to_csv(output_df_path, index=False)
    return df


def main():
    tqdm.pandas()
    args = get_args()
    df_path = args.all_data_df_path
    img_url_to_path_fn = download_images(
        input_file=args.input, output_dir=args.img_output_dir
    )
    df = create_initial_dataset(
        scraped_data_fn=args.input,
        img_url_to_path_fn=img_url_to_path_fn,
        output_df_path=df_path,
    )
    df = add_ocr_columns(df=df, output_dir=args.ocr_output_dir, output_df_path=df_path)
    df = add_clip_scores_column(
        df=df, output_dir=args.clip_scores_output_dir, output_df_path=df_path
    )


if __name__ == "__main__":
    main()
