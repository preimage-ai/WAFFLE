import json
import os

import pandas as pd
from tqdm.auto import tqdm
import pickle
from legends.grounded_legend import GroundedLegend, GroundedArcFeats
from ocr.models.gcp.ocr_texts import OCRTexts
from vlms.detr.detr_result import DetrResult


def get_key_formats(key):
    key_formats = set([key])
    if key.isnumeric():
        # sometimes number keys can have a leading 0
        try:
            key_formats.add(str(int(key)))
        except ValueError:
            print(f"Could not convert {key} to int")
    if key.isalpha():
        # sometimes text keys can be mislead for upper or lower case
        try:
            key_formats.add(key.upper())
            key_formats.add(key.lower())
        except ValueError:
            print(f"Could not convert {key} to upper or lower case")
    return key_formats


def format_keys(keys):
    formatted_keys = set([text.strip() for text in keys])
    for key in keys:
        formatted_keys.update(get_key_formats(key))
    return formatted_keys


def get_legend_values_in_ocr(legend_fn, ocr_fn, found_ratio_threshold=0.3):
    return get_legend_values_and_ocr_boxes(
        legend_fn, ocr_fn, found_ratio_threshold
    ).keys()


def get_legend_values_and_ocr_boxes(legend_fn, ocr_fn, found_ratio_threshold=0.3):
    legend_values_to_ocr_boxes = {}
    if not isinstance(legend_fn, str):
        return legend_values_to_ocr_boxes
    if not os.path.exists(legend_fn):
        print(f"Legend json not found for {legend_fn}")
        return legend_values_to_ocr_boxes
    legend = json.load(open(legend_fn))
    ocr_texts = OCRTexts(ocr_fn)
    found_keys = ocr_texts.find_texts(format_keys(legend.keys()))
    if len(found_keys) / len(legend) < found_ratio_threshold:
        return legend_values_to_ocr_boxes
    for key in found_keys:
        key_options = get_key_formats(key.text)
        for key_option in key_options:
            if key_option in legend:
                legend_value = legend[key_option]
                if legend_value in legend_values_to_ocr_boxes.keys():
                    legend_values_to_ocr_boxes[legend_value].append(key.bounding_box)
                else:
                    legend_values_to_ocr_boxes[legend_value] = [key.bounding_box]
                break
    return legend_values_to_ocr_boxes


def get_grounded_legend_with_od(
    legend_fn, ocr_fn, od_results_dir, found_ratio_threshold=0.2
):
    """
    Returns a grounded version of the legend with the OCR boxes
    that match the legend values and the object detection results.
    Returns empty if the legend is not found or the precentage of
    found keys is less than the threshold.
    """

    grounded_legend = GroundedLegend()
    if not isinstance(legend_fn, str):
        return grounded_legend
    if not os.path.exists(legend_fn):
        print(f"Legend json not found for {legend_fn}")
        return grounded_legend
    legend = json.load(open(legend_fn))

    detr_result = DetrResult(od_results_dir)
    block_candidates = detr_result.get_ocr_blocks(
        ocr_fn=ocr_fn, negative_labels=["legend"]
    )
    if not block_candidates:
        return grounded_legend
    ocr_candidates = OCRTexts(blocks=block_candidates)
    found_keys = ocr_candidates.find_texts(
        format_keys(legend.keys()), filter_text_chuncks=False
    )
    if len(found_keys) / len(legend) < found_ratio_threshold:
        return grounded_legend
    for key in found_keys:
        key_options = get_key_formats(key.text)
        for key_option in key_options:
            if key_option in legend:
                legend_value = legend[key_option]
                grounded_legend.update_legend_value(
                    key.text, legend_value, key.bounding_box
                )
                break
    page_id = os.path.basename(ocr_fn).split(".")[0]
    grounded_legend.to_json(
        f"data/outputs/legends_outputs/grounded_legends/{page_id}.json"
    )
    pickle.dump(
        grounded_legend,
        open(f"data/outputs/legends_outputs/grounded_legends/{page_id}.pkl", "wb"),
    )
    return grounded_legend


def format_arc_feats(arc_feat_fn):
    lines = open(arc_feat_fn).read().splitlines()
    # find the first sequence of lines that start all with a "• "
    # anything after that is an extension and should be disregarded
    for i, line in enumerate(lines):
        if not line.strip().startswith("•"):
            lines = lines[:i]
            break
    # remove lines that start with "I " as they are usually part of the llm talk
    lines = [
        line.replace("•", "").strip()
        for line in lines
        if not line.replace("•", "").strip().lower().startswith("i ")
    ]
    # remove parentheses in lines and their content if they exist
    lines = [line.split("(")[0].strip() for line in lines]
    return lines


def get_grounded_arch_features_with_od(arc_feat_fn, ocr_fn, od_results_dir):
    """
    Stores a grounded version of the architechtural features with
    the OCR boxes that match them and the object detection results.
    """
    arc_feat_json_fn = arc_feat_fn.replace(".txt", ".json")
    # if os.path.exists(arc_feat_json_fn):
    #     return
    if not isinstance(arc_feat_fn, str):
        return
    if not os.path.exists(arc_feat_fn):
        # print(f"Architechtural features not found for {arc_feat_fn}")
        return
    arc_feats = format_arc_feats(arc_feat_fn)
    if not arc_feats:
        print(f"All architechtural features filtered for {arc_feat_fn}")
        return

    detr_result = DetrResult(od_results_dir)
    # using the same lables and filtering as when the features were extracted
    block_candidates = [
        block_candidate
        for block_candidate in detr_result.get_ocr_blocks(
            ocr_fn=ocr_fn, label="floorplan", negative_labels=["legend"]
        )
        if len(block_candidate.paragraphs) == 1
        and len(block_candidate.text.split()) < 3
    ]
    if not block_candidates:
        return

    ocr_candidates = OCRTexts(blocks=block_candidates)
    found_arch_feats = ocr_candidates.find_texts(
        arc_feats,
        filter_text_chuncks=False,
        lower_case=True,
        substrings=True,
    )

    grounded_arc_feats = GroundedArcFeats()
    for arch_feat in found_arch_feats:
        grounded_arc_feats.update_arch_feat_value(
            arch_feat.text, arch_feat.bounding_box
        )
    if grounded_arc_feats.feats:
        grounded_arc_feats.to_json(arc_feat_json_fn)
    return


if __name__ == "__main__":
    tqdm.pandas()
    df = pd.read_csv("data/clean_csv/dataset.csv")
    
    row = df[df.page_id == 12238833].iloc[0]
    get_grounded_legend_with_od(
        legend_fn=row["legend_fn_v2"],
        ocr_fn=row["ocr_fn"],
        od_results_dir=row["od_results_dir"],
    )

    df.progress_apply(
        lambda row: get_grounded_legend_with_od(
            legend_fn=row["legend_fn_v2"],
            ocr_fn=row["ocr_fn"],
            od_results_dir=row["od_results_dir"],
        ),
        axis=1,
    )
    # df.progress_apply(
    #     lambda row: get_grounded_arch_features_with_od(
    #         arc_feat_fn=f"data/outputs/legends_outputs/arc_feats_from_ocr_with_od/{row['page_id']}.txt",
    #         ocr_fn=row["ocr_fn"],
    #         od_results_dir=row["od_results_dir"],
    #     ),
    #     axis=1,
    # )
