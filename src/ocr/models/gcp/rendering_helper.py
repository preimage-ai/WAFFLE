import json
import os
import urllib.request

from tqdm.auto import tqdm
from PIL import Image, ImageDraw, ImageFont
from enum import Enum
from google.cloud.vision_v1.types import BoundingPoly
from legends.grounded_legend import get_box_from_ocr_box

from ocr.models.gcp.ocr_texts import OCRTexts, Symbol

# Several functions taken from
# https://github.com/GoogleCloudPlatform/python-docs-samples/blob/7a74b96f48e99f6bc28a9f5fcb279c76066d2131/vision/snippets/document_text/doctext_test.py
# Documentation at https://cloud.google.com/vision/docs/fulltext-annotations

# a path used for storing images temprary when reading from url
TMP_IMAGE_PATH = "tmp_image.png"


class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


def __draw_boxes(image, bounds, color):
    """Draws a border around the image using the hints in the vector list.

    Args:
        image: the input image object.
        bounds: list of coordinates for the boxes.
        color: the color of the box.

    Returns:
        An image with colored bounds added.
    """
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        draw.polygon(
            [
                bound.vertices[0].x,
                bound.vertices[0].y,
                bound.vertices[1].x,
                bound.vertices[1].y,
                bound.vertices[2].x,
                bound.vertices[2].y,
                bound.vertices[3].x,
                bound.vertices[3].y,
            ],
            None,
            color,
        )
    return image


def draw_box_with_text(
    image,
    bound,
    text,
    box_color="red",
    text_color="black",
    text_background_color="cyan",
    outline_color="blue",
):
    draw = ImageDraw.Draw(image)

    if isinstance(bound, BoundingPoly):
        bound = get_box_from_ocr_box(bound)

    draw.polygon(
        [
            bound[0],  # xmin
            bound[1],  # ymin
            bound[2],  # xmax
            bound[1],  # ymin
            bound[2],  # xmax
            bound[3],  # ymax
            bound[0],  # xmin
            bound[3],  # ymax
        ],
        None,
        box_color,
    )

    # Draw text above the polygon
    font = ImageFont.truetype("src/ocr/models/gcp/Times New Roman.ttf", 16)
    text_width, text_height = font.getsize(text)
    x = (bound[0] + bound[2]) / 2 - text_width / 2
    y = (bound[1] + bound[3]) / 2 - text_height / 2 - 20
    draw.rectangle(
        (x - 2, y, x + text_width + 2, y + text_height + 2),
        fill=text_background_color,
        outline=outline_color,
    )
    draw.text((x, y), text, font=font, fill=text_color)
    return image


def __get_document_bounds(document, feature):
    """Finds the document bounds given an image and feature type.

    Args:
        document: the document containing the image's text annotations
        feature: feature type to detect.

    Returns:
        List of coordinates for the corresponding feature type.
    """

    bounds = []

    # Collect specified feature bounds by enumerating all document features
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if feature == FeatureType.SYMBOL:
                            bounds.append(symbol.bounding_box)

                    if feature == FeatureType.WORD:
                        bounds.append(word.bounding_box)

                if feature == FeatureType.PARA:
                    bounds.append(paragraph.bounding_box)

            if feature == FeatureType.BLOCK:
                bounds.append(block.bounding_box)

    # The list `bounds` contains the coordinates of the bounding boxes.
    return bounds


def render_visualization(document, image_uri, output_path):
    """Outlines document features (blocks, paragraphs and words) given an image document parsing.

    Args:
        image_uri: path to the input image.
        document: parsed document of the image in image_uri.
        output_path: path to the output image.
    """
    urllib.request.urlretrieve(image_uri, TMP_IMAGE_PATH)
    image = Image.open(TMP_IMAGE_PATH)
    os.remove(TMP_IMAGE_PATH)

    bounds = __get_document_bounds(document, FeatureType.BLOCK)
    __draw_boxes(image, bounds, "blue")
    bounds = __get_document_bounds(document, FeatureType.PARA)
    __draw_boxes(image, bounds, "red")
    bounds = __get_document_bounds(document, FeatureType.WORD)
    __draw_boxes(image, bounds, "yellow")

    if output_path != 0:
        image.save(output_path)
    else:
        image.show()


def draw_boxes_with_texts(
    texts,
    document_fn,
    img_path,
    output_path,
    box_color="red",
    text_color="black",
    text_background_color="cyan",
    outline_color="blue",
    found_ratio_threshold=0.3,
):
    ocr_texts = OCRTexts(document_fn)
    found_texts = ocr_texts.find_texts(texts)
    if len(found_texts) / len(texts) < found_ratio_threshold:
        # print(f"Found {len(found_texts)} out of {len(texts)} texts in {document_fn} which is less than {found_ratio_threshold}")
        return
    if not found_texts:
        print(f"Could not find any of the texts {texts} in {document_fn}")
        return
    image = Image.open(img_path)
    image = image.convert("RGB")
    for text in found_texts:
        text_background_color = (
            "PaleGreen" if isinstance(text, Symbol) else text_background_color
        )

        image = draw_box_with_text(
            image,
            bound=text.bounding_box,
            text=text.text,
            box_color=box_color,
            text_color=text_color,
            text_background_color=text_background_color,
            outline_color=outline_color,
        )
    image.save(output_path)
