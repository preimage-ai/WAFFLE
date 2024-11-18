# taken from https://chat.openai.com/share/f3e0b41e-887b-4a28-9eed-6e5a5511dfb2
import json


PREFIXES = [
    "an illustration of ",
    "a drawing of ",
    "a sketch of ",
    "a picture of ",
    "a photo of ",
    "a document of ",
    "an image of ",
    "a visual representation of ",
    "a graphic of ",
    "a rendering of ",
    "a diagram of ",
    "",
]

NEGATIVE_PREFIXES = [
    "a 3d simulation of ",
    "a 3d model of ",
    "a 3d rendering of ",
]

POSITIVE_CATEGORY_SUFFIXES = [
    "a floor plan",
    "an architectural layout",
    "a blueprint of a building",
    "a layout design",
]

NEGATIVE_CATEGORY_SUFFIXES = [
    "a map",
    "a building",
    "people",
    "an aerial view",
    "a cityscape",
    "a landscape",
    "a topographic representation",
    "a satellite image",
    "geographical features",
    "a mechanical design",
    "an engineering sketch",
    "an abstract pattern",
    "wallpaper",
    "a Window plan",
    "a tomb plan",
    "a staircase plan",
]

POSITIVE_CATEGORIES = [
    prefix + suffix for prefix in PREFIXES for suffix in POSITIVE_CATEGORY_SUFFIXES
]

NARROW_POSITIVE_SCORES = [prefix + "a floor plan" for prefix in PREFIXES]


def get_negative_categories(
    prefixes=PREFIXES + NEGATIVE_PREFIXES, suffixes=NEGATIVE_CATEGORY_SUFFIXES
):
    categories = [prefix + suffix for prefix in prefixes for suffix in suffixes]
    categories += [
        category.replace(" of ", "")
        for category in prefixes
        if category.endswith(" of ")
    ]
    return categories


NEGATIVE_CATEGORIES = get_negative_categories()


def get_score_for_categories(categories_dict_fn, positive_categories):
    try:
        with open(categories_dict_fn, "r") as f:
            categories_dict = json.load(f)
            return sum(categories_dict[key] for key in positive_categories)
    except Exception as inst:
        print(f"Error reading {categories_dict_fn}:")
        print(inst)
        return 0
    
def has_top_n_categories(categories_dict_fn, positive_categories, n=5):
    try:
        with open(categories_dict_fn, "r") as f:
            categories_dict = json.load(f)
            sorted_categories = sorted(
                categories_dict.items(), key=lambda item: item[1], reverse=True
            )
            top_categories = [key for key, _ in sorted_categories[:n]]
            positive_categories_in_top = [
                category for category in positive_categories if category in top_categories
            ]
            return True if len(positive_categories_in_top) >= n else False
    except Exception as inst:
        print(f"Error reading {categories_dict_fn}:")
        print(inst)
        return False


def get_highest_clip_category(categories_dict_fn):
    try:
        with open(categories_dict_fn, "r") as f:
            categories_dict = json.load(f)
            return max(categories_dict, key=categories_dict.get)
    except Exception as inst:
        print(f"Error reading {categories_dict_fn}:")
        print(inst)
        return False