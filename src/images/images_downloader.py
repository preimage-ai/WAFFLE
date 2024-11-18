import json
import os
import pandas as pd

from tqdm.auto import tqdm
from img2dataset import download


def save_img_urls(output_path, urls):
    """
    Save a list of image urls to a text file
    """
    with open(output_path, "w") as f:
        for url in urls:
            f.write(url + "\n")


def get_paths_of_type(dir_path, suffix):
    """
    Get paths of files of a certain type in a directory
    """
    paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(suffix):
                paths.append(os.path.join(root, file))
    return paths


def save_img_url_to_path(img_dir):
    """
    Save a mapping between image urls and image paths from the downloaded output
    Returns the path to the saved file
    """
    json_paths = get_paths_of_type(img_dir, "json")
    json_paths = [path for path in json_paths if "stats" not in path]
    error_counter = 0
    url_to_path = {}
    for json_path in tqdm(json_paths):
        try:
            with open(json_path, "r") as json_file:
                json_data = json.load(json_file)
                url_to_path[json_data["url"]] = json_path.replace(".json", ".jpg")
        except Exception as inst:
            print(f"Error reading {json_path}:")
            print(inst)
            error_counter += 1
    print(f"Completed dict creation, error reading {error_counter} files")
    df = pd.DataFrame(list(url_to_path.items()), columns=["url", "path"])
    url_to_path_fn = os.path.join(img_dir, "img_url_to_path.csv")
    df.to_csv(url_to_path_fn, index=False)
    print(
        f"Saved image url to path mapping for {len(url_to_path)} images under {url_to_path_fn}"
    )
    return url_to_path_fn


import pathlib


def modify_img_url(url):
    """
    Modify the url to remove the 'thumb/' part and the '?' part
    """
    url = url.replace("thumb/", "")
    if "?" in url:
        url_parts = url.split("?")
        url = "".join(url_parts[:-1])
        url = url.split("?")[0]
    if ".svg" in url or ".tif" in url:
        delimiter = ".svg" if ".svg" in url else ".tif"
        url_parts = url.split(delimiter)
        url = "".join(url_parts[:-1])
    suffix = pathlib.Path(url).suffix
    if url.count(suffix) > 1:
        url_parts = url.split("/")
        url = "/".join(url_parts[:-1])
    return url


import requests
from bs4 import BeautifulSoup


def get_last_a_tag(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        span_content = soup.find("span", class_="mw-filepage-other-resolutions")
        a_tags = span_content.find_all("a")
        # Select the last 'a' tag
        last_a_tag = a_tags[-1]
        # Get the URL in the last 'a' tag
        last_url = last_a_tag.get("href")
        return last_url
    except Exception as inst:
        print(f"Error getting last a tag from {url}:")
        print(inst)
        return None


def download_images(input_file, output_dir):
    """
    Dowload the images from the urls in the input file to the output directory
    Returns a mapping between the image urls and the stored image paths
    """
    df = pd.read_csv(input_file)
    print(f"Loaded Dataframe from {input_file}")
    temp_urls_path = "data/temp_urls.txt"
    save_img_urls(temp_urls_path, df["img_url"].tolist())
    print(f"Extracted {len(df)} image urls to {temp_urls_path}")

    download(
        thread_count=64,
        resize_only_if_bigger=True,
        image_size=2500,
        url_list=temp_urls_path,
        output_folder=output_dir,
    )
    print(f"Saved images to {output_dir}")
    os.remove(temp_urls_path)
    return save_img_url_to_path(output_dir)