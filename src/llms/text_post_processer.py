import math
import re
from geonamescache.mappers import country
import requests
import nltk
from nltk.corpus import brown
from collections import Counter
import pandas as pd
from tqdm.auto import tqdm

PREFIXES = [
    "type of ",
    "floor plans of ",
    "floor plan of ",
    "proposed or planned ",
    "map of ",
    "lands of ",
    "site plan of ",
    "pg plans of ",
    "plans of ",
    "schematic plans of ",
    "maps of ",
    "proposal for a ",
    "a ",
    "an ",
]
SUFFIXES = [
    " of new jersey",
    " in the united states",
    " of washington",
    " in vienna",
    " in north brabant",
    " of chicago",
    " in italy",
    " in bergen, norway",
    " in germany",
    " in rome",
    " of iowa",
    " in trondheim, norway",
    " in city",
    " by type",
    " in the netherlands",
]


IN_LANGUAGE_FORMATS = ["in {0}", "In {0}"]
LANGUAGES = [
    "English",
    "French",
    "Uppercase",
    "Dutch",
    "Spanish",
    "Norwegian",
    "Russian",
    "Turkish",
    "German",
    "Korean",
]


IN_LANGUAGE_VARIATIONS = [
    template.format(lang)
    for template in [t + ":" for t in IN_LANGUAGE_FORMATS]
    for lang in LANGUAGES + [lang.lower() for lang in LANGUAGES]
] + [
    template.format(lang)
    for template in IN_LANGUAGE_FORMATS
    for lang in LANGUAGES + [lang.lower() for lang in LANGUAGES]
]


class EnglishLanguageDetector:
    def __init__(self):
        nltk.download("brown")
        frequencies = Counter(i.lower() for i in tqdm(brown.words()))
        self.df = pd.DataFrame(
            [(v, c) for v, c in frequencies.items()], columns=["word", "freq"]
        )

    def has_eng_word(self, text, freq_cutoff=3):
        df_ = self.df[self.df.freq >= freq_cutoff]
        vocab = set(df_.word)
        return len([w for w in text.split() if w.lower() in vocab]) >= math.ceil(
            len(text.split()) / 2
        )


class TextPostProcesser:
    def __init__(self, canonical_reps_manager, english_detector):
        url = "https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/british_spellings.json"
        self.english_detector = english_detector
        self.british_to_american = requests.get(url).json()
        self.canonical_reps_manager = canonical_reps_manager
        self.country_mapper = country(from_key="name", to_key="iso")

    # taken from https://stackoverflow.com/questions/42329766/python-nlp-british-english-vs-american-english
    def _americanize(self, string):
        for british_spelling, american_spelling in self.british_to_american.items():
            string = string.replace(british_spelling, american_spelling)

        return string

    def _postprocess(self, x):
        if x.startswith("insert "):
            # e.g. insert category here, insert building type here, ...
            x = "building"  # most generic label
        if x == "historic american buildings survey drawing":  # common case
            x = "historic american building"
        if " of a " in x:
            x = " ".join(x.split(" of a ")[1:])
        if " of an " in x:
            x = " ".join(x.split(" of an ")[1:])
        if " less than " in x:
            x = x.split(" less than ")[0]
        for p in PREFIXES:
            if x.startswith(p):
                x = x[len(p) :]
        for s in SUFFIXES:
            if x.endswith(s):
                x = x[: -len(s)]
        words = x.split()
        if len(words) > 2 and words[-2] == "in":
            x = " ".join(words[:-2])  # remove "in X" e.g. "in scotland"
        if x.startswith("historic ") and len(x.split()) > 3:
            x = " ".join(x.split()[1:])  # remove "historic ..." from long labels
        return self._americanize(x)

    def upper_to_title(self, x):
        return x.title() if x.isupper() else x

    def format_bracketed_answer(self, ans_raw):
        # add < if its missing in order to identify html tags
        if not ans_raw.startswith("<"):
            ans_raw = "<" + ans_raw
        # remove html tags
        ans_raw = re.sub(r"\<[a-zA-Z0-9]\>|\<\/[a-zA-Z0-9]\>", "", ans_raw)
        # some heuristics for cleaning the answers
        for var in IN_LANGUAGE_VARIATIONS:
            ans_raw = ans_raw.replace(var, "")

        in_parenthesis = re.search(r"\((.*?)\)", ans_raw)
        in_angled_brackets = re.search(r"\<(.*?)\>", ans_raw)

        angled_brackets_ans, parenthesis_ans = "", ""

        if not in_angled_brackets and not in_parenthesis:
            return self.upper_to_title(ans_raw.strip())
        # if only one answer exists, return it
        if in_angled_brackets:
            angled_brackets_ans = self.upper_to_title(
                in_angled_brackets.group(1).strip()
            )
            if not in_parenthesis:
                # return self.canonical_reps_manager.get_canonical_rep(
                #     angled_brackets_ans
                # )
                return angled_brackets_ans
        if in_parenthesis:
            parenthesis_ans = self.upper_to_title(in_parenthesis.group(1).strip())
            if not in_angled_brackets:
                # return self.upper_to_title(
                #     self.canonical_reps_manager.get_canonical_rep(parenthesis_ans)
                # )
                return parenthesis_ans

        # there are two answers, one in angled brackets and one in parenthesis.
        if in_parenthesis.group(0) in angled_brackets_ans:
            angled_brackets_ans = self.upper_to_title(
                angled_brackets_ans.replace(in_parenthesis.group(0), "")
            )
        # prefer the angled brackets as the main answer, choose the parenthesis only if the angled brackets are not in english
        main_ans = angled_brackets_ans
        # secondary_ans = parenthesis_ans
        if (
            parenthesis_ans != ""
            and self.english_detector.has_eng_word(parenthesis_ans)
        ) and (
            not self.english_detector.has_eng_word(angled_brackets_ans)
            or angled_brackets_ans == ""
        ):
            main_ans = parenthesis_ans
            # secondary_ans = angled_brackets_ans
        # if there are two answers, save them to the canonical reps file
        # return self.canonical_reps_manager.get_canonical_rep_and_update(
        #     main_ans, secondary_ans
        # )
        return main_ans

    def format_city_state_country_answer(self, location_info):
        location_parts = (
            location_info.split(";")
            if ";" in location_info
            else location_info.split(",")
        )
        city, state, country = "", "", ""
        if len(location_parts) == 3:
            city, state, country = location_parts
        elif len(location_parts) == 2:
            city, country = location_parts
        elif len(location_parts) == 1:
            country = location_parts[0]
        else:
            print(
                f"Location answer '{location_info}' should contain between 1 and 3 parts, but got {len(location_parts)}"
            )
            return location_info

        removable_strs = ["city:", "state:", "country:", "unknown"]
        removable_strs = [s.title() for s in removable_strs] + removable_strs
        for removable_str in removable_strs:
            city = city.replace(removable_str, "").strip()
            state = state.replace(removable_str, "").strip()
            country = country.replace(removable_str, "").strip()

        if country.lower() == state.lower() or city.lower() == state.lower():
            state = ""
        if city.lower() == "city":
            city = ""
        if state.lower() == "state":
            state = ""
        if country.lower() == "country":
            country = ""

        if not self.country_mapper(country) and self.country_mapper(city):
            # if the country is not recognized, but the city is, assume the order is reversed
            # - the city is the country and the country is the city
            tmp = country
            country = city
            city = tmp

        return city, state, country

    def post_process(self, ans):
        """
        Returns the stripped column if exists, otherwise strips the column in the following way:
        1. turns the string to lower case
        2. replaces '_' with ' '
        3. removes all non-alphanumeric characters, except for ', ", (, ), / and :
        """
        ans = self.format_bracketed_answer(ans)
        ans = ans.replace("_", " ").lower().strip()
        # Remove all non-alphanumeric characters, except for ', ", (, ), / and :
        ans = re.sub(r"[^a-zA-Z0-9',\"/():\-\s]+", "", ans)
        if ":" in ans:
            # if the column contains a colon it could mean a title appears before the actual value
            ans = ans.split(":")[1].strip()
        return self._postprocess(ans)
    
    
def post_process_type(building_type):
    if 'type of ' in building_type:
        building_type = building_type.replace('type of ', '').strip()
    if 'building' in building_type:
        pass
    if 'plan' in building_type:
        pass
    return building_type

def post_process_type_main():
    import pandas as pd
    df = pd.read_csv('data/large_dataset_modified_building_info.csv')
    unique_building_types = pd.DataFrame()
    unique_building_types['building_type'] = df['building_type'].unique()
    
    unique_building_types['modified_building_type'] = unique_building_types['building_type'].apply(post_process_type)
    pass

