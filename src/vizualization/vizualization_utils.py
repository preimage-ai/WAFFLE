import json
import os
import base64

from ocr.models.gcp.ocr_texts import OCRTexts
from google.cloud.vision_v1.types.text_annotation import TextAnnotation


MAX_WIDTH_IN_PX = 800
IMG_URL_TAG = '<img src="{0}" style="max-width: {1}px; height: auto;">'
IMG_PATH_TAG = '<img src="data:{0};base64,{1}" style="max-width: {2}px; height: auto;">'
OCR_RENDERING_PATH = "ocr/ocr_renderings/"
BLACKED_OUT_TEXT = '<p style="background-color:#000000;"> N/A </p>'


def print_img(img_path, html_out, max_width_in_px=MAX_WIDTH_IN_PX):
    if os.path.exists(img_path):
        with open(img_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
        print(IMG_PATH_TAG.format(img_path, encoded_string, max_width_in_px), file=html_out)
    else:
        print(IMG_URL_TAG.format(img_path, max_width_in_px), file=html_out)
    

def get_document(df_row):
    try:
        with open(df_row["ocr_fn"], "r") as f:
            return TextAnnotation.from_json(json.load(f))
    except Exception as e:
        print(f"Failed to load json ocr output from {df_row['ocr_fn']} with error: {e}")
        return


def print_2_column_table_row(name, data, html_out):
    print(f"<tr><td>{name}</td> <td>{data}</td> </tr>", file=html_out)


def print_3_column_table_row(name, data_1, data_2, html_out):
    print(
        f"<tr><td>{name}</td> <td>{data_1}</td> <td>{data_2}</td> </tr>", file=html_out
    )

def print_4_column_table_row(name, data_1, data_2, data_3, html_out):
    print(
        f"<tr><td>{name}</td> <td>{data_1}</td> <td>{data_2}</td> <td>{data_3}</td> </tr>", file=html_out
    )

def add_background_color(text, background_color):
    return f'<p style="background-color:{background_color};"> {text} </p>'


def get_hyperlink_str(url):
    return f'<a href="{url}">{url}</a>'

def print_ocr_texts(df_row, html_out, confidence_threshold=0.5):
    print(f"<h5>OCR texts with confidence > {confidence_threshold}</h5>", file=html_out)
    print(f"<table><tr> <th>Text</th> <th>confidence</th> </tr>", file=html_out)

    ocr_texts = OCRTexts(df_row['ocr_fn']).get_ocr_texts()
    for ocr_text in ocr_texts:
        if ocr_text.confidence > confidence_threshold:
            print_2_column_table_row(ocr_text.text, ocr_text.confidence, html_out)

    print(f"</table>", file=html_out)
    print(f"<br><b></b><br>", file=html_out)


def get_wiki_joint_value(wiki_row, column_name):
    wiki_row = wiki_row.dropna(subset=[column_name])
    if not wiki_row.empty and isinstance(wiki_row[column_name].values[0], str):
        return ", ".join(wiki_row[column_name].values)
    return BLACKED_OUT_TEXT