import argparse
import json
import os
import numpy as np

import pandas as pd
from tqdm.auto import tqdm
import regex
from legends.grounded_legend import get_box_from_ocr_box
from legends.legend_formatter import check_letters
from llms.sentence_embeddings import SentenceEmbeddings
from ocr.models.gcp.ocr_texts import OCRTexts
from vlms.detr.detr_result import DetrResult
from googletrans import Translator

ALLOWED_SHORT_WORDS = json.load(
    open("src/legends/simplified_arc_exceptions/allowed_short_words.json")
)
WORD_MAPPINGS = json.load(open("src/legends/simplified_arc_exceptions/mappings.json"))
WORDS_TO_IGNORE = json.load(
    open("src/legends/simplified_arc_exceptions/words_to_ignore.json")
)
WORDS_TO_REDUCE_TO = json.load(
    open("src/legends/simplified_arc_exceptions/words_to_reduce_to.json")
)
WORDS_TO_REMOVE_FROM = json.load(
    open("src/legends/simplified_arc_exceptions/words_to_remove_from.json")
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--df_path",
        type=str,
        default="data/clean_csv/dataset.csv",
        help="path to df that contains the initial dataset",
    )
    return parser.parse_args()


def check_line(line):
    # filter lines that start with "I " as they are usually part of the llm talk
    # remove lines that have a single letter - we check the line after removing non letter characters
    return not line.lower().startswith("i ") and check_letters(
        regex.sub(r"[^\p{L}]", "", line)
    )


def format_arc_feats(arc_feat_fn):
    lines = open(arc_feat_fn).read().splitlines()
    # find the first sequence of lines that start all with a "• "
    # anything after that is an extension and should be disregarded
    for i, line in enumerate(lines):
        if not line.strip().startswith("•"):
            lines = lines[:i]
            break

    lines = [line.replace("•", "").strip() for line in lines if check_line(line)]
    return lines


class ArcFeatsFormatter:
    def __init__(self, bag_of_words=[]):
        self.translator = Translator()
        self.sentence_embeddings = SentenceEmbeddings()
        if bag_of_words:
            self.sentence_embeddings.set_bag_of_words(bag_of_words)

    def simplify_arc_feat(self, arc_feat):
        """
        Simplify the given architectural feature:
        * remove any non-letter characters
        * translate to english
        """
        if arc_feat.startswith("("):
            arc_feat = arc_feat.split("(")[1]
        # remove non-letter characters and extract the first part of the string
        clean = regex.sub(r"[^\p{L} ]", "", arc_feat.split("(")[0]).lower()

        clean_translated = self.translator.translate(clean)
        if clean_translated.src == "en":
            return clean

        # the text isn't in english, so check if there is a parentheses value, if yes then it is usually a translation
        parentheses_val = regex.findall(r"\(([^)]*)", arc_feat)
        if parentheses_val and not parentheses_val[0].startswith("#"):
            candidate = parentheses_val[0]
            candidate = regex.sub(r"[^\p{L} ]", "", candidate.split("(")[0]).lower()
            # if the candidate is in english and isn't too long, then it is the simplified value
            candidate_translated = self.translator.translate(candidate)
            if candidate_translated.src == "en" and len(candidate.split()) < 3:
                return candidate

        return clean_translated.text

    def format_and_simplify_arc_feats(self, df_row):
        """
        Stores a grounded version of the architechtural features with
        the OCR boxes that match them and the object detection results.
        """

        simplified_arc_feat_fn = f'data/outputs/legends_outputs/simplified_arc_feats_v2/{df_row["page_id"]}.json'
        if os.path.exists(simplified_arc_feat_fn):
            return
        arc_feat_fn = df_row["ocr_arch_feat_with_od_fn"]
        if not isinstance(arc_feat_fn, str) or not os.path.exists(arc_feat_fn):
            # print(f"Architechtural features not found for {arc_feat_fn}")
            return
        arc_feats = format_arc_feats(arc_feat_fn)
        if not arc_feats:
            print(f"All architechtural features filtered for {arc_feat_fn}")
            return

        detr_result = DetrResult(df_row["od_results_dir"])
        # using the same lables and filtering as when the features were extracted
        block_candidates = [
            block_candidate
            for block_candidate in detr_result.get_ocr_blocks(
                ocr_fn=df_row["ocr_fn"], label="floorplan", negative_labels=["legend"]
            )
            if len(block_candidate.paragraphs) == 1
            and len(block_candidate.text.split()) < 3
        ]
        if not block_candidates:
            return

        ocr_candidates = OCRTexts(blocks=block_candidates)
        grounded_arc_feats = {}

        sorted_arc_feats = sorted(arc_feats, key=len, reverse=True)

        for arc_feat in sorted_arc_feats:
            found_feats = ocr_candidates.find_texts(
                [arc_feat],
                filter_text_chuncks=False,
                lower_case=True,
                substrings=True,
            )
            if not found_feats:
                continue
            simplify_arc_feat = self.simplify_arc_feat(arc_feat)
            grounded_arc_feats[arc_feat] = {
                "simplified_arc_feat": simplify_arc_feat,
                "ocr_texts": [],
            }
            for found_feat in found_feats:
                grounded_arc_feats[arc_feat]["ocr_texts"].append(
                    {
                        "text": found_feat.text,
                        "ocr_box": get_box_from_ocr_box(
                            found_feat.bounding_box,
                        ),
                    }
                )
                ocr_candidates.remove_ocr_text(found_feat)
        if grounded_arc_feats:
            json.dump(grounded_arc_feats, open(simplified_arc_feat_fn, "w"), indent=4)
            return grounded_arc_feats

        return

    def simplify_arc_feats_legends(self, legend_fn):
        simplified_legend_fn = legend_fn.replace(
            "grounded_legends", "simplified_grounded_legends"
        )
        if os.path.exists(simplified_legend_fn):
            return
        legend = json.load(open(legend_fn, "rb"))
        simplified_legend = {}
        for legend_key, legend_values in legend.items():
            arc_feat = legend_values["value"]
            # translated = self.translator.translate(arc_feat).text
            simplified = self.simplify_arc_feat(arc_feat)
            simplified_post_processed = post_process_simlified_arc_feat(simplified)
            # if len(translated.split()) >= 3:
            #     # if the translation is long, search for a close short word
            #     simplified = self.sentence_embeddings.get_closest_word(translated)
            # legend_values["translated_value"] = translated
            if simplified_post_processed:
                legend_values["simplified_value"] = simplified_post_processed
                simplified_legend[legend_key] = legend_values
        if simplified_legend:
            json.dump(simplified_legend, open(simplified_legend_fn, "w"), indent=4)
        else:
            print(f"Filtered all arc feats for {legend_fn}")


def post_process_simlified_arc_feat(arc_feat):
    # remove all one-letter words as they are usually not useful,
    # and remove words that are in the remove list
    arc_feat_ = " ".join(
        [
            word
            for word in arc_feat.split()
            if len(word) > 1 and word not in WORDS_TO_REMOVE_FROM
        ]
    )
    # remove all duplicate words
    arc_feat_ = " ".join(dict.fromkeys(arc_feat_.split()))
    arc_feat_ = arc_feat_.lower().strip()
    if arc_feat_ in WORD_MAPPINGS.keys():
        arc_feat_ = WORD_MAPPINGS[arc_feat_]
    if len(arc_feat_) < 4 and arc_feat_ not in ALLOWED_SHORT_WORDS:
        return None
    if arc_feat_ in WORDS_TO_IGNORE:
        return None
    for word in WORDS_TO_REDUCE_TO:
        if word in arc_feat_:
            arc_feat_ = word
    return arc_feat_


def post_process_simlified_arc_feats(simplified_arc_feats_fn):
    simplified_arc_feats = json.load(open(simplified_arc_feats_fn, "rb"))
    filtered_arc_feats = {}
    for arc_feat, values in simplified_arc_feats.items():
        simplified_arc_feat = post_process_simlified_arc_feat(
            values["simplified_arc_feat"]
        )
        if simplified_arc_feat:
            filtered_arc_feats[arc_feat] = values
            filtered_arc_feats[arc_feat]["simplified_arc_feat"] = simplified_arc_feat
    if filtered_arc_feats:
        json.dump(
            filtered_arc_feats,
            open(
                simplified_arc_feats_fn.replace(
                    "simplified_arc_feats_v3", "simplified_arc_feats_v4"
                ),
                "w",
            ),
            indent=4,
        )
    else:
        print(f"Filtered all arc feats for {simplified_arc_feats_fn}")


def save_unified_grounded(page_id):
    """
    Saves a single json file for page_id including all the grounded
    information we have on it - from a legend or a list of arc feats.
    The json will be a mapping between an architectural feature in
    it's simplified version and the ocr boxes that match it. In case
    an image contains both a legend and a list of arc feats, the
    json will include both.
    """
    unified_fn = f"data/outputs/legends_outputs/unified_grounded/{page_id}.json"
    # if os.path.exists(unified_fn):
    #     return unified_fn

    legend_fn = (
        f"data/outputs/legends_outputs/simplified_grounded_legends/{page_id}.json"
    )
    arc_feats_fn = (
        f"data/outputs/legends_outputs/simplified_arc_feats_v4/{page_id}.json"
    )
    unified = {}

    if os.path.exists(legend_fn):
        legend = json.load(open(legend_fn, "rb"))
        for legend_key, legend_value in legend.items():
            simplified_value = legend_value["simplified_value"]
            ocr_boxes = legend_value["ocr_boxes"]
            if simplified_value not in unified:
                unified[simplified_value] = {
                    "ocr_boxes": ocr_boxes,
                    "legend_keys": [legend_key],
                }
            else:
                unified[simplified_value]["ocr_boxes"] += ocr_boxes
                unified[simplified_value]["legend_keys"].append(legend_key)

    if os.path.exists(arc_feats_fn):
        arc_feats = json.load(open(arc_feats_fn, "rb"))
        for arc_feat in arc_feats.values():
            simplified_arc_feat = arc_feat["simplified_arc_feat"]
            ocr_boxes = [ocr_text["ocr_box"] for ocr_text in arc_feat["ocr_texts"]]
            if simplified_arc_feat not in unified:
                unified[simplified_arc_feat] = {
                    "ocr_boxes": ocr_boxes,
                }
            else:
                unified[simplified_arc_feat]["ocr_boxes"] += ocr_boxes

    if unified:
        json.dump(unified, open(unified_fn, "w"), indent=4)
        return unified_fn
    return np.NaN


if __name__ == "__main__":
    from tqdm.auto import tqdm

    tqdm.pandas()

    df = pd.read_csv(get_args().df_path)
    df["grounded_unified_fn"] = df["page_id"].progress_apply(save_unified_grounded)

    arc_feat_dir = "data/outputs/legends_outputs/simplified_arc_feats_v4"
    # files = os.listdir(arc_feat_dir)
    # random.shuffle(files)

    # for arc_feats_fn in tqdm(files):
    #     simplified_arc_feats_fn = os.path.join(arc_feat_dir, arc_feats_fn)
    #     post_process_simlified_arc_feats(simplified_arc_feats_fn)

    formatter = ArcFeatsFormatter()
    simplified_arc_feats = []
    for arc_feat_fn in tqdm(os.listdir(arc_feat_dir)):
        arc_feats = json.load(open(os.path.join(arc_feat_dir, arc_feat_fn), "rb"))
        simplified_arc_feats += [
            arc_feat["simplified_arc_feat"] for arc_feat in arc_feats.values()
        ]
    # formatter.sentence_embeddings.set_bag_of_words(list(set(simplified_arc_feats)))

    legends_dir = "data/outputs/legends_outputs/grounded_legends"
    files = os.listdir(legends_dir)
    files = [fn for fn in files if fn.endswith(".json")]
    # random.shuffle(files)

    for legend_fn in tqdm(files):
        formatter.simplify_arc_feats_legends(os.path.join(legends_dir, legend_fn))
    pass
