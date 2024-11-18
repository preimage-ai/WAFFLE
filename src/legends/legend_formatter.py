import argparse
import pandas as pd
import regex as re
from tqdm.auto import tqdm
import json
import os
from legends.legend_image_renderer import format_texts
from legends.legend_utils import format_keys

from ocr.models.gcp.ocr_texts import OCRTexts

# still missing:
# color (e.g. red: Nave, blue: Chapel, green: Aisle, etc.)
# open text like this: Tore (Gates) - numbered 1 to 6
# images with multiple legends/repeatative keys

INDEXES = [
    r"\d{1,2}°?\p{L}?\'?",  # numbers with optional single char, 1 2 3 or 1a 2b 3c, can have ' at the end
    r"\p{L}{1,2}°?\'?\d?",  # chars with optional single number, a b c or a1 b2 c3, can have ' after the char
    # r"[A-Z]{1,4}",  # all caps indexes
    r"(I|i|II|ii|III|iii|IV|iv|V|v|VI|vi|VII|vii|VIII|vii|IX|ix|X|x)",  # roman numerals, I II III or i ii iii
]
SPACE = r"\s*"  # space
SPACE_PLUS = r"\s+"  # space

MUTIPLE_INDEXES_DELIMITERS = [
    r"\s*,\s*",  # comma ,
    r"\s*\-\s*",  # hyphen -
    r"\s*—\s*",  # em dash —
    r"\s*/\s*",  # slash /
    r"\s*\.+\s*",  # dot .
    SPACE_PLUS,  # at least one space
]

BULLET_LIST_SIGN = r"^[^\p{L}\d]"  # not a letter or a number

HYPHEN_CHAR = r"\s*\-\s*"  # hyphen -
DELIMITERS = [
    r"\.",  # dot .
    r"\)",  # closing parenthesis )
    r":",  # colon :
    r"\-",  # dash -
    r"=",  # equals sign =
    r"—",  # em dash —
    SPACE_PLUS,  # at least one space
]

TWO_LETTER_WORDS = list(pd.read_csv("src/legends/two_letter_words.csv").word)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legends_dir",
        type=str,
        default="data/outputs/legends_outputs/from_caption",
        help="directory of the raw legends extracted using the LLM",
    )
    return parser.parse_args()


def __get_patterns_for_line(line):
    indexes = []
    for index in INDEXES:
        if re.search(index, line):
            indexes.append(index)
    delimiters = []
    for delimiter in DELIMITERS:
        if re.search(delimiter, line):
            delimiters.append(delimiter)
    return [
        # indexes are at the begining of the row (e.g. 1. item, 2. other item etc.)
        r"^({index_0}({mul_idx_delimiter}{index_1}({delimiter_0})?)*{space}{delimiter_1})".format(
            index_0=index_0,
            index_1=index_1,
            mul_idx_delimiter=mul_idx_delimiter,
            space=SPACE,
            delimiter_0=delimiter_0,
            delimiter_1=delimiter_1,
        )
        for index_0 in indexes
        for index_1 in indexes
        for mul_idx_delimiter in MUTIPLE_INDEXES_DELIMITERS
        for delimiter_0 in delimiters
        for delimiter_1 in delimiters
    ] + [
        # indexes are at the end of the row in parenthesis (e.g. item (1), other item (2) etc.)
        r"\(({index_0}({mul_idx_delimiter}{index_1})*)\)$".format(
            index_0=index_0, index_1=index_1, mul_idx_delimiter=mul_idx_delimiter
        )
        for index_0 in indexes
        for index_1 in indexes
        for mul_idx_delimiter in MUTIPLE_INDEXES_DELIMITERS
    ] + [
        # indexes are at the end of the row with a delimiter preceeding them (e.g. item-1, other item-2 etc.)
        r"({delimiter}{space}{index_0}({mul_idx_delimiter}{index_1})*)$".format(
            delimiter=delimiter, space=SPACE, index_0=index_0, index_1=index_1, mul_idx_delimiter=mul_idx_delimiter
        )
        # we remove SPACE_PLUS in this case to avoid lines that end with a single/double char for some reason.
        for delimiter in list(set(delimiters) - set([SPACE_PLUS]))
        for index_0 in indexes
        for index_1 in indexes
        for mul_idx_delimiter in MUTIPLE_INDEXES_DELIMITERS
    ]


def __clean_line(line):
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    line = re.sub(BULLET_LIST_SIGN, "", line)
    return line.strip()


def check_letters(line):
    """
    Check if a string contains at least two letters.

    Parameters:
    s (str): The input string.

    Returns:
    bool: True if the string contains at least two letters, False otherwise.
    Excludes languages that often use single chars to represent a word: Chinese, Japanese and Korean.
    """
    single_char_languages_pattern = r"\p{Han}|\p{Hangul}|\p{Hiragana}|\p{Katakana}"
    return len(re.findall(r'\p{L}', line)) >= 2 or re.search(single_char_languages_pattern, line)

def __maybe_split_key(key):
    keys = []
    if "," in key or "/" in key:
        delimeter = "," if "," in key else "/"
        keys = [key_part.strip() for key_part in key.split(delimeter)]
    elif "-" in key or "—" in key:
        # this is a range, return all numbers/chars in the range inclusive
        delimeter = "-" if "-" in key else "—"
        key_parts = [key_part.strip() for key_part in key.split(delimeter)]
        for i in range(len(key_parts) - 1):
            from_key = key_parts[i]
            to_key = key_parts[i + 1]
            if from_key.isnumeric() and to_key.isnumeric():
                keys += [str(i) for i in range(int(from_key), int(to_key) + 1)]
            # not numbers, probably chars
            elif len(from_key) > 1 or len(to_key) > 1:
                # not single chars, can't refer to as a range
                keys += [from_key, to_key]
            elif from_key.isalpha() and to_key.isalpha():
                # single alpha chars, probably a range
                keys += list(map(chr, range(ord(from_key), ord(to_key) + 1)))
    if keys:
        return keys
    return [key]


def __get_matched_patterns(line):
    line_dict = {}
    for pattern in __get_patterns_for_line(line):
        match = re.search(pattern, line)
        if not match:
            continue
        # verify that the match is not only a part of the key - see that the begining of the value starts after it
        verification_pattern = ""
        if match.end() < len(line):
            # the value starts after the match
            verification_pattern = (
                pattern
                + r"((\s?\(?(\p{L}|\d|\s|«){3,}(\p{L}|\d|«))|(\s?(\p{L}|\d|«){2,}))"
            )
        else:
            # the match is at the end, the value starts before the match
            verification_pattern = (
                r"((\(?(\p{L}|\d|»|\s){3,}(\p{L}|\d|»)\s?)|((\p{L}|\d|»){2,}\s?))"
                + pattern
            )
        verification_match = re.search(verification_pattern, line)
        if match and verification_match:
            value = line.replace(match.group(0), "").strip()
            key = match.group(1)
            key = key.strip()
            if key.lower() in TWO_LETTER_WORDS:
                # this is probably a line with just texts and not in key:value format
                continue
            for delimiter in DELIMITERS:
                delimiter_regex = SPACE + delimiter + SPACE
                key = re.sub(delimiter_regex + "$", "", key)
                key = re.sub("^" + delimiter_regex, "", key)
                value = re.sub(delimiter_regex + "$", "", value)
                value = re.sub("^" + delimiter_regex, "", value)
            if value:
                keys = __maybe_split_key(key)
                if len(keys) > 1:
                    pass
                for key in keys:
                    line_dict[key] = value
            break
    return line_dict


def __match_patterns(line, legend_dict):
    line_dict = __get_matched_patterns(line)
    for key, value in line_dict.items():
        if key in legend_dict:
            legend_dict[key] += ", " + value
        else:
            legend_dict[key] = value
    return legend_dict


def format_legend(legend_fn):
    """
    Takes in a filename containing a legend and formats it into a dict representing the legend

    Args:
    legend_fn (str): The filename of the legend

    Returns:
    dict: The legend in a dict format
    """
    with open(legend_fn) as f:
        legend = f.read()
    legend_lines = legend.split("\n")
    legend_dict = {}
    for line in legend_lines:
        line = __clean_line(line)
        if (
            line == ""
            or line.lower().startswith("i hope")
            or line.lower().startswith("i kept")
            or line.lower().startswith("i have")
            or line.lower().startswith("i think")
            or line.lower().startswith("i deduced")
            or line.lower().startswith("key")
            or line.lower().startswith("value")
            or len(line) < 3
        ):
            # skip empty lines, short lines, and lines that start with "I hope" as they can be interperted as keys
            continue
        legend_dict = __match_patterns(line, legend_dict)
    if len(legend_dict) < 3:
        # too small to be a legend
        return {}
    return legend_dict


def save_legend_candidates(ocr_texts, output_fn, sort=False):
    """
    Scans the OCR texts for legend regexs and saves the candidates to a file

    Args:
    ocr_texts (list): The OCR text of the image
    output_fn (str): The filename to save the legend candidates to
    """
    # if os.path.exists(output_fn):
    #     return output_fn
    legend_candidates = []
    for text in [ocr_text[1:-1] for ocr_text in ocr_texts[1:-1].split(", ")]:
        if __get_matched_patterns(text):
            # we add the original line as we don't want to loose its format.
            # later when we get the actual legend we'll format it.
            legend_candidates.append(text)
    if legend_candidates and len(legend_candidates) > 1:
        with open(output_fn, "w") as f:
            if sort:
                legend_candidates = sorted(legend_candidates, key=str.lower)
            f.write("\n".join(legend_candidates))
            f.close()
        return output_fn
    else:
        return pd.NA

def save_legend_candidates_with_blocks(ocr_fn, output_fn, sort=False):
    """
    Scans the OCR texts for legend regexs and saves the candidates to a file

    Args:
    ocr_fn (string): The OCR texts filename
    output_fn (str): The filename to save the legend candidates to
    """
    if os.path.exists(output_fn):
        return output_fn
    ocr_texts = OCRTexts(ocr_fn)
    # we use the block level to capture texts that appear close as a heuristic for a legend.
    block_texts = ocr_texts.get_ocr_texts(level='block')
    legend_candidates = []
    for block_text in block_texts:
        for paragraph in block_text.paragraphs:
            text = paragraph.text
            if not check_letters(text):
                # skip lines with less than two letters
                continue
            if __get_matched_patterns(text):
                # we add the original line as we don't want to loose its format.
                # later when we get the actual legend we'll format it.
                legend_candidates.append(text)
    if legend_candidates and len(legend_candidates) > 2:
        with open(output_fn, "w") as f:
            if sort:
                legend_candidates = sorted(legend_candidates, key=str.lower)
            f.write("\n".join(legend_candidates))
            f.close()
        return output_fn
    else:
        return pd.NA

def save_legend_candidates_from_df(df_path):
    """
    Scans the OCR texts for legend regexs and saves the candidates to a file

    Args:
    df_path (str): The path to the dataframe containing the OCR text
    """
    tqdm.pandas()
    df = pd.read_csv(df_path)
    df["ocr_legend_candidate_fn"] = df.progress_apply(
        lambda row: save_legend_candidates_with_blocks(
            row["ocr_fn"],
            f"data/outputs/legends_outputs/from_ocr_candidates/{row['page_id']}.txt",
        ),
        axis=1,
    )
    df.to_csv(df_path, index=False)


def save_ocr_formatted_legend(ocr_legend_fn):
    if not isinstance(ocr_legend_fn, str):
        return pd.NA
    dict_path = ocr_legend_fn.replace(".txt", ".json").replace(
        "from_ocr_candidates/", "from_ocr_v2/"
    )
    if os.path.exists(dict_path):
        return dict_path
    legend = format_legend(ocr_legend_fn)
    if legend:
        json.dump(legend, open(dict_path, "w"), indent=4)
        return dict_path
    return pd.NA


def save_ocr_formatted_legends(df_path):
    """
    Scans the OCR texts for legend regexs and saves the candidates to a file

    Args:
    df_path (str): The path to the dataframe containing the OCR text
    """
    tqdm.pandas()
    df = pd.read_csv(df_path)
    df["ocr_legend_v2_fn"] = df["ocr_legend_v2_fn"].progress_apply(
        save_ocr_formatted_legend
    )
    pass
    # df.to_csv(df_path, index=False)


def get_unified_legend_path(df_row):
    """
    Returns a single legend path for each image.
    We'll choose the legend that has the most OCR key matches.
    If they all have the same, use the following choice order:
    1. If there exists a caption legend, use it
    2. Otherwise if there exists an OCR legend, use it
    3. Lastly choose the Wiki legend if it exists
    The logic is set in this order under the assumption that legends asociated
    to the image are probably more accurate and the Wiki legend may refer to a
    different image.

    Args:
    df_path (str): The path to the dataframe containing the OCR text
    """
    caption_legend_fn = df_row["caption_legend_fn"]
    ocr_legend_v2_fn = df_row["ocr_legend_v2_fn"]
    wiki_legend_fn = df_row["wiki_legend_fn"]

    existing_legends = [
        legend
        for legend in [caption_legend_fn, ocr_legend_v2_fn, wiki_legend_fn]
        if isinstance(legend, str)
    ]
    if not existing_legends:
        return pd.NA
    if len(existing_legends) == 1:
        return existing_legends[0]
    # if there are multiple legends, choose the one with the most found keys
    maximal_legend_fn = pd.NA
    max_count = 0
    for legend_fn in existing_legends:
        with open(legend_fn) as f:
            legend = json.load(f)
        ocr_texts = OCRTexts(df_row["ocr_fn"])
        found_texts = ocr_texts.find_texts(format_keys(legend.keys()))
        if len(found_texts) > max_count:
            max_count = len(found_texts)
            maximal_legend_fn = legend_fn
    return maximal_legend_fn


if __name__ == "__main__":
    args = get_args()
    legends_dir = "data/outputs/legends_outputs/from_ocr_with_od"
    for file in tqdm(os.listdir(legends_dir)):
        if file.endswith(".txt"):
            legend_fn = os.path.join(legends_dir, file)
            dict_path = legend_fn.replace(".txt", ".json")
            if os.path.exists(dict_path):
                continue
            legend_dict = format_legend(legend_fn)
            if legend_dict:
                json.dump(legend_dict, open(dict_path, "w"), indent=4)
    pass
