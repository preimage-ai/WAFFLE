"""
Contains methods for extracting text from out image data using llms.
"""

import json
import os
import numpy as np
import pandas as pd

from legends.legend_utils import get_legend_values_in_ocr
from ocr.models.gcp.ocr_texts import OCRTexts
from vlms.detr.detr_result import DetrResult

ENGLISH_TRANSLATION_PROMPT = open(
    "src/llms/prompts/english_translation.txt", "r"
).read()


def get_page_content_table_value(page_content, table_header):
    if not isinstance(page_content, str):
        return pd.NA
    # Each table value is separated by 3 newlines (sometimes more..)
    table_values = page_content.split("\n \n \n ")
    matches = [row for row in table_values if row.strip().startswith(table_header)]
    if matches:
        return matches[0].replace(table_header, "").strip()
    return pd.NA


def get_column_value(df_row, col_name):
    return (
        df_row[col_name]
        if col_name in df_row and isinstance(df_row[col_name], str)
        else pd.NA
    )


def replace_with_column_value(prompt, df_row, col_name):
    col_val = get_column_value(df_row, col_name)
    if isinstance(col_val, str):
        return prompt.replace(f"{{{col_name}}}", col_val)
    return prompt


def replace_with_column_values(prompt, df_row, col_names):
    for col_name in col_names:
        prompt = replace_with_column_value(prompt, df_row, col_name)
    return prompt


def get_modified_prompt(prompt, df_row):
    filename = get_column_value(df_row, "fn")
    if filename is not pd.NA:
        filename = filename[5:]
        prompt = prompt.replace("{fn}", filename)

    im_page_content = get_column_value(df_row, "page_content")
    description = pd.NA
    if im_page_content is not pd.NA:
        description = get_page_content_table_value(im_page_content, "Description")
        # Sometimes the filename appears again in the description
        if filename is not pd.NA and isinstance(description, str):
            description = description.replace(filename + "\n", "")
            description = description.replace(filename, "")
    has_description = (
        isinstance(description, str) and description != "" and len(description) < 1500
    )
    description = description if has_description else ""
    replacement = (
        "{description}" if has_description else "* Entity description:\n{description}\n"
    )
    prompt = prompt.replace(replacement, description)

    wiki_shows = get_column_value(df_row, "wiki_shows")
    has_wiki = isinstance(wiki_shows, str) and wiki_shows != ""
    wiki_shows = wiki_shows if has_wiki else ""
    replacement = "{wiki_shows}" if has_wiki else "* Wiki page summary:\n{wiki_shows}\n"
    prompt = prompt.replace(replacement, wiki_shows)

    ocr_texts = get_column_value(df_row, "ocr_texts")
    # don't use ocr texts if they're too long
    use_ocr = isinstance(ocr_texts, str) and ocr_texts != "" and len(ocr_texts) < 2000
    ocr_texts = ocr_texts if use_ocr else ""
    replacement = (
        "{ocr_texts}"
        if use_ocr
        else "* Texts that appear in the image (extracted with OCR):\n{ocr_texts}\n"
    )
    prompt = prompt.replace(replacement, ocr_texts)

    prompt = replace_with_column_values(
        prompt,
        df_row,
        [
            "category",
            "building_name",
            "building_type",
            "wiki_building_name",
            "wiki_building_type",
        ],
    )

    return prompt


def has_description(df_row):
    description = get_page_content_table_value(df_row["page_content"], "Description")
    return isinstance(description, str) and description != ""


def get_single_token_answer(llm, prompt, df_row):
    prompt = get_modified_prompt(prompt, df_row)
    ans_raw = llm.run_prompt(prompt, max_new_tokens=1)
    return ans_raw[0].upper() if len(ans_raw) > 0 else pd.NA


def get_bracketed_answer(llm, prompt, df_row, post_processer_func):
    if df_row:
        prompt = get_modified_prompt(prompt, df_row)
    ans = llm.run_prompt(prompt)
    return post_processer_func(ans)


def get_english_translation(llm, text):
    """
    Translate the text to english using the llm model.
    """
    return llm.run_prompt(ENGLISH_TRANSLATION_PROMPT.replace("{text}", text))


def get_english_bracketed_answer(
    llm, prompt, df_row, post_processer_func, english_detector
):
    prompt = get_modified_prompt(prompt, df_row)
    ans = post_processer_func(llm.run_prompt(prompt))
    if not english_detector.has_eng_word(ans):
        ans = post_processer_func(get_english_translation(llm, ans))
    return ans


def replace_with_ocr_legend_candidate(prompt, ocr_legend_candidate_fn):
    ocr_legend_candidate = None
    if isinstance(ocr_legend_candidate_fn, str):
        with open(ocr_legend_candidate_fn, "r") as f:
            ocr_legend_candidate = filter_blocks_with_little_text(f.read())
    use_ocr_candidate = (
        isinstance(ocr_legend_candidate, str)
        and ocr_legend_candidate != ""
        and len(ocr_legend_candidate) < 2000
    )
    if not use_ocr_candidate:
        return ""
    return prompt.replace("{ocr_legend_candidate}", ocr_legend_candidate)


def replace_with_ocr_arch_feat_candidate_from_od(prompt, df_row):
    detr_result = DetrResult(df_row["od_results_dir"])
    block_candidates = detr_result.get_ocr_blocks(
        df_row["ocr_fn"], label="floorplan", negative_labels=["legend"]
    )
    # we filter out blocks with more than one paragraph and with more than 2 words
    # this is a hueristic to identify words that aren't part of a sentance or a title,
    # but rather an architechtural label.
    arch_feat_candidates = list(
        set(
            [
                block_candidate.text.strip().lower()
                for block_candidate in block_candidates
                if len(block_candidate.paragraphs) == 1
                and len(block_candidate.text.split()) < 3
            ]
        )
    )
    if len(arch_feat_candidates) == 0:
        return np.NaN
    ocr_arch_feat_candidate = "\n".join(arch_feat_candidates)
    if len(ocr_arch_feat_candidate) > 2000:
        return np.NaN
    return prompt.replace("{ocr_arch_feat_candidate}", ocr_arch_feat_candidate)


def get_has_arch_feat_with_od_answer(llm, prompt, df_row):
    has_legend = get_column_value(df_row, "grounded_legend_fn")
    if isinstance(has_legend, str) and has_legend == "Y":
        # filter out pages with a legend to avoid confusion
        return np.NaN
    prompt = replace_with_ocr_arch_feat_candidate_from_od(prompt, df_row)
    if not isinstance(prompt, str):
        return np.NaN
    prompt = get_modified_prompt(prompt, df_row)
    return get_single_token_answer(llm, prompt, df_row)


def get_arch_feat_content_with_od_answer(llm, prompt, df_row):
    has_arch_feat = get_column_value(df_row, "has_arch_feat_with_od")
    if not isinstance(has_arch_feat, str) or has_arch_feat != "Y":
        return np.NaN
    arch_feat_fn = f"data/outputs/legends_outputs/arc_feats_from_ocr_with_od/{df_row['page_id']}.txt"
    if os.path.exists(arch_feat_fn):
        return arch_feat_fn
    prompt = replace_with_ocr_arch_feat_candidate_from_od(prompt, df_row)
    if not isinstance(prompt, str):
        return np.NaN
    prompt = get_modified_prompt(prompt, df_row)
    arch_feats = llm.run_prompt(prompt, max_new_tokens=2000)
    with open(arch_feat_fn, "w") as f:
        f.write(arch_feats)
    return arch_feat_fn


def get_simplified_arc_feat(
    llm, prompt, df_row, arc_feat, post_processer_func, english_detector
):
    simplified_arc_feat = get_english_bracketed_answer(
        llm,
        prompt.replace("{arch_feat}", arc_feat),
        df_row=df_row,
        post_processer_func=post_processer_func,
        english_detector=english_detector,
    )
    if "don't know" in simplified_arc_feat:
        # if the simplified arc feat contains "don't know", it's probably not a valid answer
        return None
    return simplified_arc_feat


def get_simplified_legend_v2(
    llm, prompt, df_row, post_processer_func, english_detector
):
    simplified_legend_fn = (
        f"data/outputs/legends_outputs/simplified_legends_v2/{df_row['page_id']}.json"
    )
    if os.path.exists(simplified_legend_fn):
        return simplified_legend_fn
    legend_fn = df_row["legend_fn_v2"]
    if not isinstance(legend_fn, str) or not os.path.exists(legend_fn):
        return np.NaN
    legend = json.load(open(legend_fn, "r"))
    simplified_legend = {}
    for key, arc_feat in legend.items():
        simplified_arc_feat = get_simplified_arc_feat(
            llm, prompt, df_row, arc_feat, post_processer_func, english_detector
        )
        simplified_legend[key] = {
                "original_value": arc_feat,
                "simplified_value": simplified_arc_feat,
            }
    if not simplified_legend:
        return np.NaN
    json.dump(simplified_legend, open(simplified_legend_fn, "w"), indent=4)
    return simplified_legend_fn


def get_simplified_arc_feats(
    llm, prompt, df_row, post_processer_func, english_detector
):
    arc_feats_fn = df_row["ocr_arch_feat_with_od_fn"]
    if not isinstance(arc_feats_fn, str) or not os.path.exists(arc_feats_fn):
        return np.NaN
    arc_feats = open(arc_feats_fn, "r").read().split("\n")
    simplified_arc_feats = {}
    for arc_feat in arc_feats:
        simplified_arc_feat = get_simplified_arc_feat(
            llm, prompt, df_row, arc_feat, post_processer_func, english_detector
        )
        simplified_arc_feats[arc_feat] = simplified_arc_feat
    if not simplified_arc_feats:
        return np.NaN
    simplified_arc_feats_fn = (
        f"data/outputs/legends_outputs/simplified_arc_feats/{df_row['page_id']}.json"
    )
    json.dump(simplified_arc_feats, open(simplified_arc_feats_fn, "w"), indent=4)
    return simplified_arc_feats_fn


def replace_with_ocr_legend_candidate_from_od(prompt, df_row):
    detr_result = DetrResult(df_row["od_results_dir"])
    block_candidates = detr_result.get_ocr_blocks(df_row["ocr_fn"], "legend")
    if len(block_candidates) == 0:
        return np.NaN
    ocr_legend_candidate = "\n\n".join(
        [block_candidate.text for block_candidate in block_candidates]
    )
    if len(ocr_legend_candidate) > 2000:
        return np.NaN
    return prompt.replace("{ocr_legend_candidate}", ocr_legend_candidate)


def get_has_legend_with_od_answer(llm, prompt, df_row):
    prompt = replace_with_ocr_legend_candidate_from_od(prompt, df_row)
    if not isinstance(prompt, str):
        return np.NaN
    prompt = get_modified_prompt(prompt, df_row)
    return get_single_token_answer(llm, prompt, df_row)


def get_legend_content_with_od_answer(llm, prompt, df_row):
    has_legend = get_column_value(df_row, "has_legend_with_od")
    if not isinstance(has_legend, str) or has_legend != "Y":
        return np.NaN
    prompt = replace_with_ocr_legend_candidate_from_od(prompt, df_row)
    if not isinstance(prompt, str):
        return np.NaN
    prompt = get_modified_prompt(prompt, df_row)
    legend = llm.run_prompt(prompt, max_new_tokens=2000)
    legend_fn = f"data/outputs/legends_outputs/from_ocr_with_od/{df_row['page_id']}.txt"
    with open(legend_fn, "w") as f:
        f.write(legend)
    return legend_fn


def get_has_legend_answer(llm, prompt, df_row):
    has_ocr_legend_v3 = get_column_value(df_row, "has_ocr_legend_v3")
    if not isinstance(has_ocr_legend_v3, str):
        return pd.NA
    if has_ocr_legend_v3 == "Y":
        return has_ocr_legend_v3
    ocr_legend_candidate_fn = get_column_value(df_row, "ocr_legend_candidate_fn")
    if not isinstance(ocr_legend_candidate_fn, str):
        return pd.NA
    prompt = replace_with_ocr_legend_candidate(prompt, ocr_legend_candidate_fn)
    if not prompt:
        return pd.NA
    prompt = get_modified_prompt(prompt, df_row)
    # elif "page_content" in df_row and not has_description(df_row):
    #     return pd.NA
    return get_single_token_answer(llm, prompt, df_row)


def contains_short_words(line, len_of_label_key_threshold=3):
    line_words = line.split()
    label_candidates = [
        text
        for text in line_words
        if len(text) < len_of_label_key_threshold or text.isnumeric()
    ]
    if len(label_candidates) > 0.7 * len(line_words):
        # the line contains a lot of label-like candidates - it's probably not part of a legend
        return True


def filter_blocks_with_little_text(ocr_legend_candidate):
    filtered = [
        line
        for line in ocr_legend_candidate.split("\n")
        if not contains_short_words(line)
    ]
    if len(filtered) < 3:
        return ""
    return "\n".join(filtered)


def get_legend_content_answer(llm, prompt, df_row):
    has_ocr_legend_v3 = get_column_value(df_row, "has_ocr_legend_v3")
    # if not isinstance(has_ocr_legend_v3, str) or not has_ocr_legend_v3 == "Y":
    #     return pd.NA
    if not isinstance(has_ocr_legend_v3, str):
        return pd.NA
    ocr_legend_candidate_fn = get_column_value(df_row, "ocr_legend_candidate_fn")
    if not isinstance(ocr_legend_candidate_fn, str):
        return pd.NA
    prompt = replace_with_ocr_legend_candidate(prompt, ocr_legend_candidate_fn)
    if not prompt:
        return pd.NA
    prompt = get_modified_prompt(prompt, df_row)
    legend = llm.run_prompt(prompt, max_new_tokens=2000)
    legend_fn = f"data/outputs/legends_outputs/from_ocr/{df_row['page_id']}.txt"
    with open(legend_fn, "w") as f:
        f.write(legend)
    return legend_fn


def get_simplified_legend_content_answer(llm, prompt, df_row):
    legend_fn = df_row["legend_fn"]
    if not isinstance(legend_fn, str) or "from_ocr" in legend_fn:
        return pd.NA

    legend = json.load(open(legend_fn, "r"))
    legend_text = ""
    for key, value in legend.items():
        legend_text += f"{key}: {value}\n"
    prompt = prompt.replace("{legend}", legend_text)
    prompt = prompt.replace("{building_type}", df_row["building_type"])
    simplified_legend = llm.run_prompt(prompt, max_new_tokens=2000)
    simplified_legend_fn = (
        f"data/outputs/legends_outputs/simplified_legends/{df_row['page_id']}.txt"
    )
    with open(simplified_legend_fn, "w") as f:
        f.write(simplified_legend)
    return simplified_legend_fn


def get_caption_answer(llm, prompt, df_row):
    legend_fn = get_column_value(df_row, "legend_fn")
    if not isinstance(legend_fn, str):
        return pd.NA
    architectural_features = get_legend_values_in_ocr(legend_fn, df_row["ocr_fn"])
    if not architectural_features:
        return pd.NA
    prompt = prompt.format(
        building_type=df_row["building_type"],
        architectural_features="\n".join(architectural_features),
    )
    captions_raw = llm.run_prompt(prompt, max_new_tokens=250)
    captions = [
        caption.replace("•", "").strip()
        for caption in captions_raw.split("\n")
        if "•" in caption
    ]
    if not captions:
        return pd.NA
    captions_fn = f"data/outputs/captions_outputs/{df_row['page_id']}.txt"
    with open(captions_fn, "w") as f:
        f.write("\n".join(captions))
        f.close()
    return captions_fn
