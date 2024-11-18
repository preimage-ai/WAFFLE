# taken from https://github.com/morrisalp/fitw-wiki-data/blob/main/src/topic.py#L10

import argparse
import shutil
import numpy as np
from tqdm.auto import tqdm
from llms import extract_text_methods
from llms.llama_inference import LlamaInference
import os
import pandas as pd

from llms.text_post_processer import EnglishLanguageDetector, TextPostProcesser
from llms.canonical_reps_manager import CanonicalRepsManager


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--df_path",
        type=str,
        default="data/clean_csv/dataset.csv",
        help="path to df that contains the initial dataset",
    )
    parser.add_argument(
        "--llama_model_name",
        type=str,
        default="meta-llama/Llama-2-13b-chat-hf",
        help="path to df that contains the initial dataset",
    )
    parser.add_argument(
        "--col_name",
        type=str,
        default="topic_answer",
        help="the name of the column to add to the dataset",
    )
    parser.add_argument(
        "--split_runs",
        type=int,
        default=1,
        help="The number of times to split the dataset and run the function so that it can be run in parallel.",
    )
    parser.add_argument(
        "--run_number",
        type=int,
        default=0,
        help="The run number for this script. Should be between 0 and split_runs - 1.",
    )
    return parser.parse_args()


class TextualDataExtractor:
    PROMPTS_DIR = "src/llms/prompts"
    ENGLISH_TRANSLATION_PROMPT_FN = "english_translation.txt"

    def __init__(self, args):
        self.llama_inference = LlamaInference(args.llama_model_name)
        # self.english_detector = EnglishLanguageDetector()
        self.text_post_processer = TextPostProcesser(
            canonical_reps_manager=None,
            english_detector=None,
            # english_detector=self.english_detector,
        )
        self.col_name_to_fn = {
            "topic_answer": extract_text_methods.get_single_token_answer,
            "building_name": lambda llm, prompt, df_row: extract_text_methods.get_bracketed_answer(
                llm, prompt, df_row, self.text_post_processer.format_bracketed_answer
            ),
            "building_type": lambda llm, prompt, df_row: extract_text_methods.get_english_bracketed_answer(
                llm,
                prompt,
                df_row,
                self.text_post_processer.post_process,
                self.english_detector,
            ),
            "raw_location_info": lambda llm, prompt, df_row: extract_text_methods.get_bracketed_answer(
                llm,
                prompt,
                df_row,
                self.text_post_processer.format_bracketed_answer,
            ),
            "location_info": lambda llm, prompt, df_row: extract_text_methods.get_bracketed_answer(
                llm,
                prompt,
                df_row,
                self.text_post_processer.format_city_state_country_answer,
            ),
            (
                "city",
                "state",
                "country",
            ): lambda llm, prompt, df_row: self.text_post_processer.format_city_state_country_answer(
                df_row["raw_location_info"]
            ),
            "has_ocr_legend": extract_text_methods.get_has_legend_answer,
            "has_ocr_legend_v2": extract_text_methods.get_has_legend_answer,
            "has_ocr_legend_v3": extract_text_methods.get_has_legend_answer,
            "has_legend_with_od": extract_text_methods.get_has_legend_with_od_answer,
            "has_arch_feat_with_od": extract_text_methods.get_has_arch_feat_with_od_answer,
            "ocr_legend_fn": extract_text_methods.get_legend_content_answer,
            "ocr_legend_v3_fn": extract_text_methods.get_legend_content_answer,
            "ocr_legend_with_od_fn": extract_text_methods.get_legend_content_with_od_answer,
            "ocr_arch_feat_with_od_fn": extract_text_methods.get_arch_feat_content_with_od_answer,
            "simplified_legend_fn": extract_text_methods.get_simplified_legend_content_answer,
            "simplified_legend_fn_v2": lambda llm, prompt, df_row: extract_text_methods.get_simplified_legend_v2(
                llm,
                prompt,
                df_row,
                self.text_post_processer.post_process,
                self.english_detector,
            ),
            "simplified_arc_feats_fn": lambda llm, prompt, df_row: extract_text_methods.get_simplified_arc_feats(
                llm,
                prompt,
                df_row,
                self.text_post_processer.post_process,
                self.english_detector,
            ),
        }
        with open(
            os.path.join(self.PROMPTS_DIR, self.ENGLISH_TRANSLATION_PROMPT_FN), "r"
        ) as f:
            self.english_translation_prompt = f.read()

    def _get_start_and_end_idx(self, df_len, run_number, split_runs):
        indexes = range(df_len)
        groups = np.array_split(indexes, split_runs)
        group_ranges = [(group[0], group[-1]) for group in groups]
        return group_ranges[run_number]

    def add_column_to_df(self, df_path, col_name, run_number=0, split_runs=1):
        """
        Extract col_name on a subset of a dataframe and save the results.

        The col_name should have a corresponding function in self.col_name_to_fn.
        If split_runs is greater than 1, this method will split the dataframe into
        split_runs and run the function on a subset of the dataframe.
        run_number should be between 0 and split_runs - 1.
        """
        df = pd.read_csv(df_path)

        start_idx, end_idx = self._get_start_and_end_idx(
            len(df), run_number, split_runs
        )
        split_df = df.iloc[start_idx : end_idx + 1]
        col_func = self.col_name_to_fn[col_name]
        col_name = list(col_name) if isinstance(col_name, tuple) else col_name
        print(
            f"Adding column: {col_name} to {df_path}, run {run_number}, {start_idx} to {end_idx}"
        )
        prompt_name = "location_info" if "location_info" in col_name else str(col_name)
        prompt_path = os.path.join(self.PROMPTS_DIR, prompt_name + ".txt")
        prompt = ""
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as f:
                prompt = f.read()

        split_df[col_name] = split_df.progress_apply(
            lambda row: col_func(self.llama_inference, prompt, row),
            axis=1,
        ).to_list()
        df_new_path = (
            df_path.replace(".csv", f"_{run_number}.csv") if split_runs > 1 else df_path
        )
        print(
            f"Saving {col_name} to {df_new_path}, run {run_number}, {start_idx} to {end_idx}"
        )
        split_df.to_csv(df_new_path, index=False)


def concat_and_save_dfs(original_df_path, split_runs):
    if split_runs == 1:
        return
    df_paths = [
        original_df_path.replace(".csv", f"_{run_number}.csv")
        for run_number in range(split_runs)
    ]
    df = pd.concat([pd.read_csv(df_path) for df_path in df_paths])
    df = df.reset_index(drop=True)
    df.to_csv(original_df_path, index=False)
    for df_path in df_paths:
        shutil.move(df_path, df_path.replace("data/", "data/archive_splits/"))


def main():
    args = get_args()
    textual_data_extractor = TextualDataExtractor(args)

    textual_data_extractor.add_column_to_df(
        df_path=args.df_path,
        col_name=args.col_name,
        run_number=args.run_number,
        split_runs=args.split_runs,
    )


if __name__ == "__main__":
    tqdm.pandas()
    main()
