import json
import os
import pickle
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from vlms.clipseg.inpainter import Inpainter
from ocr.models.gcp.ocr_texts import OCRTexts
from llms.sentence_embeddings import SentenceEmbeddings
from tqdm.auto import tqdm
from vlms.clipseg.v2.sizing_utils import (
    crop_to_box,
    get_box_corners,
    inflate_box_with_ratio,
    create_mask,
    resize_array,
    resize_image,
    safe_resize,
)
from vlms.clipseg.v2.plot_utils import plot_gt, plot_datum


TEST_COUNTRIES = json.load(open("data/clean_csv/test_countries.json"))
MAX_INPAINTING_SIZE = 1024
INPAINTER = Inpainter()


class CLIPSegGT:
    def __init__(self, img, pos_labels_to_boxes, neg_labels_to_boxes):
        self.pos_labels_to_boxes = pos_labels_to_boxes
        self.neg_labels_to_boxes = neg_labels_to_boxes
        self.scores = create_mask(
            img, [box for value in self.pos_labels_to_boxes.values() for box in value]
        )
        self.mask = create_mask(
            img,
            [box for value in self.pos_labels_to_boxes.values() for box in value]
            + [box for value in self.neg_labels_to_boxes.values() for box in value],
        )

    def crop_masks(self, bbox):
        self.scores = crop_to_box(self.scores, bbox)
        self.mask = crop_to_box(self.mask, bbox)
        return self

    def add_prefix_to_labels(self, prefix):
        self.pos_labels_to_boxes = {
            f"[{prefix}] {label}": boxes
            for label, boxes in self.pos_labels_to_boxes.items()
        }
        self.neg_labels_to_boxes = {
            f"[{prefix}] {label}": boxes
            for label, boxes in self.neg_labels_to_boxes.items()
        }
        return self


def get_inpainting_boxes(df_row):
    ocr_texts = OCRTexts(df_row["ocr_fn"])
    ocr_texts.filter_blocks_with_text_chuncks()

    return [
        get_box_corners(word.bounding_box)
        for block in ocr_texts.blocks
        for paragraph in block.paragraphs
        for word in paragraph.words
    ]


def get_inpainted_size(img):
    """
    returns the number closest to the maximum size of the image that devides in 8
    (this is a requirement for using the inpainter)
    """
    return max(img.size) + ((8 - max(img.size)) % 8)


def get_inpainted_image(df_row, img, floorplan_bbox, inpainter):
    # inpainted_fn = f"data/for_segmentation/inpainted_images/{df_row['page_id']}.png"
    inpainted_fn = f"data/for_segmentation/inpainted_test_images/{df_row['page_id']}.png"
    if os.path.exists(inpainted_fn):
        return Image.open(inpainted_fn).convert("RGB")
    floorplan_cropped = crop_to_box(img, floorplan_bbox)

    try:
        inpainting_mask = crop_to_box(
            create_mask(img, get_inpainting_boxes(df_row)), floorplan_bbox
        )

        size = min(get_inpainted_size(floorplan_cropped), MAX_INPAINTING_SIZE)

        floorplan_img_resized = resize_image(floorplan_cropped, size)
        inpainting_mask_resized = resize_array(inpainting_mask, size)

        inpainted = inpainter.call(
            floorplan_img_resized, inpainting_mask_resized, size, prompt="remove"
        )

        inpainted_resized = safe_resize(
            inpainted, floorplan_cropped.width, floorplan_cropped.height
        )

        inpainted_resized.save(inpainted_fn)
        return inpainted_resized
    except Exception as e:
        print(f"Error inpainting {df_row['page_id']}: {e}. Returning original image.")
        return floorplan_cropped


class CLIPSegDatum:
    def __init__(self, img, gts, df_row, inpaint_and_crop=True):
        self.img = img
        self.gts = gts
        if inpaint_and_crop:
            floorplan_bbox = json.loads(df_row.floorplan_max_od_box)
            self.img = get_inpainted_image(df_row, img, floorplan_bbox, INPAINTER)
            gts = [gt.crop_masks(floorplan_bbox) for gt in gts]
            # remove gts who's positive scores have been completely cropped out
            self.gts = [gt for gt in gts if gt.scores.sum() > 0]
        # add building type prefix to gts
        self.gts = [
            gt.add_prefix_to_labels(df_row.building_type.upper()) for gt in self.gts
        ]
        self.df_row = df_row


def get_gt_for_label(
    img,
    label,
    labels_to_boxes,
    sentence_embeddings,
    similarity_upper_threshold=0.7,
    similarity_lower_threshold=0.4,
):
    pos_labels_to_boxes = {label: [box for box in labels_to_boxes[label]]}
    neg_labels_to_boxes = {}

    other_labels = list(labels_to_boxes.keys())
    other_labels.remove(label)
    for other_label in other_labels:
        similarity = sentence_embeddings.get_similarity(label, other_label)
        other_boxes = [other_box for other_box in labels_to_boxes[other_label]]
        if similarity < similarity_lower_threshold:
            neg_labels_to_boxes[other_label] = other_boxes
        elif similarity > similarity_upper_threshold:
            pos_labels_to_boxes[other_label] = other_boxes

    return CLIPSegGT(img, pos_labels_to_boxes, neg_labels_to_boxes)


def valitate_datum_candidate(df_row):
    if not isinstance(df_row.grounded_unified_fn, str) or not os.path.exists(
        df_row.grounded_unified_fn
    ):
        print("Warning: grounded_unified_fn does not exist for", df_row.page_id)
        return False
    if not isinstance(df_row.floorplan_max_od_box, str):
        print("Warning: no floorplan box for", df_row.page_id)
        return False
    return True


def create_datum(df_row, common_labels, sentece_embeddings):
    if not valitate_datum_candidate(df_row):
        return None
    grounded = json.load(open(df_row.grounded_unified_fn))
    labels_to_boxes_and_source = {
        label: {
            "ocr_boxes": value["ocr_boxes"],
            "source": "legend" if "legend_keys" in value else "arc_feat",
        }
        for label, value in grounded.items()
    }
    # labels_to_boxes = {label: value["ocr_boxes"] for label, value in grounded.items()}
    if common_labels:
        labels_to_boxes_and_source = {
            label: value
            for label, value in labels_to_boxes_and_source.items()
            if label in common_labels
        }
        # labels_to_boxes = {
        #     label: boxes
        #     for label, boxes in labels_to_boxes.items()
        #     if label in common_labels
        # }
        if not labels_to_boxes_and_source:
            print("Warning: no common labels for", df_row.page_id)
            return None
        # if not labels_to_boxes:
        #     print("Warning: no common labels for", df_row.page_id)
        #     return None
    ratio = 1 / sum(
        [len(value["ocr_boxes"]) for value in labels_to_boxes_and_source.values()]
    )
    # ratio = 1 / sum([len(boxes) for boxes in labels_to_boxes.values()])
    # first run used the default min_scale_factor=3, try again with a dynamic lower value
    # we'll try again with a dynamic approach to the min_scale_factor:
    # if the grounded info is from architecural features on the image, this means the boxes
    # are larger (they contain words instead of just labels) so we can use a lower min_scale_factor

    # labels_to_boxes = {
    #     label: [
    #         inflate_box_with_ratio(img, box, ratio, min_scale_factor=min_scale_factor)
    #         for box in boxes
    #     ]
    #     for label, boxes in labels_to_boxes.items()
    # }
    img = Image.open(df_row.img_path).convert("RGB")
    labels_to_boxes = {
        label: [
            inflate_box_with_ratio(
                img,
                box,
                ratio,
                min_scale_factor=3 if value["source"] == "arc_feat" else 2,
            )
            for box in value["ocr_boxes"]
        ]
        for label, value in labels_to_boxes_and_source.items()
    }

    labels_added = set()
    gts = []
    for label in labels_to_boxes.keys():
        if label in labels_added:
            continue
        gt = get_gt_for_label(img, label, labels_to_boxes, sentece_embeddings)
        labels_added.update(gt.pos_labels_to_boxes.keys())
        gts.append(gt)
    if gts:
        return CLIPSegDatum(img, gts, df_row)
    else:
        return None


def create_dataset(df, common_labels, output_path):
    sentence_embeddings = SentenceEmbeddings("paraphrase-multilingual-mpnet-base-v2")
    dataset = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        datum = create_datum(row, common_labels, sentence_embeddings)
        if datum is not None:
            dataset.append(datum)

    with open(output_path, "wb") as f:
        print("Saving dataset to:", output_path)
        pickle.dump(dataset, f)

    # for validation..
    test_dataset = [
        data for data in dataset if data.df_row["country"] in TEST_COUNTRIES
    ]
    train_dataset = [
        data for data in dataset if data.df_row["country"] not in TEST_COUNTRIES
    ]
    test_path = output_path.replace(".pkl", "_test.pkl")
    train_path = output_path.replace(".pkl", "_train.pkl")

    with open(test_path, "wb") as f:
        print("Saving test dataset to:", test_path)
        pickle.dump(test_dataset, f)
    with open(train_path, "wb") as f:
        print("Saving train dataset to:", train_path)
        pickle.dump(train_dataset, f)

    return dataset


def create_test_data_from_coco(coco_path, output_path):
    def boxs_from_segmentation(segmentation):
        xmin = int(min(segmentation[0::2]))
        ymin = int(min(segmentation[1::2]))
        xmax = int(max(segmentation[0::2]))
        ymax = int(max(segmentation[1::2]))
        return [xmin, ymin, xmax, ymax]

    import torchvision

    df = pd.read_csv("data/clean_csv/dataset.csv")

    coco_dir = os.path.dirname(coco_path)

    coco = torchvision.datasets.CocoDetection(root=coco_dir, annFile=coco_path)
    imgs_id_to_path = {
        id: os.path.join(coco_dir, image["file_name"])
        for id, image in coco.coco.imgs.items()
    }
    imgs_id_to_annotation = coco.coco.imgToAnns
    id2label = {k: v["name"] for k, v in coco.coco.cats.items()}
    dataset = []

    for id, path in imgs_id_to_path.items():
        img = Image.open(path).convert("RGB")
        page_id = int(os.path.basename(path).split(".")[0].split("_")[0])
        try:
            df_row = df[df.page_id == page_id].iloc[0]
        except IndexError:
            print("Warning: page_id not found in dataset.csv", page_id)
            continue
        annotations = imgs_id_to_annotation[id]
        labels_to_boxes = {
            id2label[annotation["category_id"]]: [] for annotation in annotations
        }
        for annotation in annotations:
            labels_to_boxes[id2label[annotation["category_id"]]] += [
                boxs_from_segmentation(segmentation)
                for segmentation in annotation["segmentation"]
            ]

        gts = []
        for label in labels_to_boxes.keys():
            gts.append(
                CLIPSegGT(
                    img,
                    pos_labels_to_boxes={label: labels_to_boxes[label]},
                    neg_labels_to_boxes={
                        l: labels_to_boxes[l]
                        for l in labels_to_boxes.keys()
                        if l != label
                    },
                )
            )
        dataset.append(CLIPSegDatum(img, gts, df_row, inpaint_and_crop=False))
    with open(output_path, "wb") as f:
        print("Saving dataset to:", output_path)
        pickle.dump(dataset, f)
    return dataset


if __name__ == "__main__":

    from collections import Counter

    create_test_data_from_coco(
        "data/for_segmentation/test_images_2/segmentation_tests_2.json",
        "data/for_segmentation/test_images_2/segmentation_tests_2.pkl",
    )

    tqdm.pandas()
    df = pd.read_csv("data/clean_csv/dataset.csv")

    df_grounded = df[df["grounded_unified_fn"].notna()]
    labels = []

    df_grounded["grounded_unified_fn"].progress_apply(
        lambda x: labels.extend(json.load(open(x)).keys())
    )

    labels_counter = Counter(labels)
    common_labels = set(
        [label for label, count in labels_counter.items() if count > 10]
    )

    create_dataset(
        df_grounded,
        common_labels,
        "data/for_segmentation/common_labels_smaller_bboxes_dataset.pkl",
    )

    import pandas as pd

    df = pd.read_csv("data/clean_csv/dataset.csv")
    page_ids = [
        104351459,
        15340001,
        46491694,
        52072425,
        7278676,
        7303402,
        74706842,
        88722275,
        12127814,
        18711314,
        38171318,
        393872,
        39518924,
        40311725,
        40501672,
        65286192,
        7194368,
        73573502,
        919940,
        10874861,
        12849405,
        128767491,
        2275116,
        2761098,
        402171,
        4109586,
        48557745,
        51300484,
        5173613,
        61931619,
        61982202,
        66427277,
        70630004,
        9467891,
        1009441,
        117065579,
        11807463,
        2097790,
        2791661,
        3526292,
        36493679,
        4650346,
        49477716,
        70264967,
        7135638,
        72006530,
        7279839,
        73674150,
        74166015,
        76898705,
        8449784,
        90946167,
        18387708,
        25516427,
        5668475,
        11832179,
        121222251,
        128721596,
        24041276,
        29769379,
        33849711,
        33954004,
        34299359,
        34299423,
        34308249,
        34310705,
        34333100,
        34530634,
        34666561,
        34864232,
        37187601,
        37930572,
        38224264,
        43017871,
        43489009,
        43502659,
        84231833,
        91464948,
    ]
    rows = df[df.page_id.isin(page_ids)]
    inpainter = Inpainter()
    for _, row in rows.iterrows():
        img_path = row.img_path.replace('data/', '/scratch/kerenganon/WAFFLE/')
        inpainted = get_inpainted_image(
            row,
            Image.open(img_path).convert("RGB"),
            json.loads(row.floorplan_max_od_box),
            inpainter,
        )