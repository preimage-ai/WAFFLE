import json
import cv2
import numpy as np
from PIL import Image


def get_new_size(size, aspect_ratio):
    return (
        (size, round(size / aspect_ratio))
        if aspect_ratio > 1
        else (round(size * aspect_ratio), size)
    )


def resize_image(image, size):
    new_size = get_new_size(size, image.width / image.height)
    resized_image = image.resize(new_size, resample=Image.NEAREST)
    final_image = Image.new("RGB", (size, size))
    final_image.paste(
        resized_image, ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
    )
    return final_image


def resize_array(array, size):
    new_size = get_new_size(size, array.shape[1] / array.shape[0])
    resized_array = cv2.resize(array, new_size, interpolation=cv2.INTER_NEAREST)
    final_array = np.zeros((size, size))
    final_array[
        (size - new_size[1]) // 2 : (size + new_size[1]) // 2,
        (size - new_size[0]) // 2 : (size + new_size[0]) // 2,
    ] = resized_array
    return final_array


def center_crop_res(img, res):
    # Calculate the aspect ratio of img
    aspect_ratio = img.size[0] / img.size[1]

    # Calculate the center of res
    center_x = res.shape[1] // 2
    center_y = res.shape[0] // 2

    # Calculate the dimensions of the cropped area
    if res.shape[1] > res.shape[0] * aspect_ratio:
        new_width = int(res.shape[0] * aspect_ratio)
        new_height = res.shape[0]
    else:
        new_width = res.shape[1]
        new_height = int(res.shape[1] / aspect_ratio)

    # Calculate the coordinates of the cropped area
    left = center_x - new_width // 2
    right = center_x + new_width // 2
    top = center_y - new_height // 2
    bottom = center_y + new_height // 2

    # Crop res
    return res[top:bottom, left:right]


def center_crop_to_aspect_ratio(img_to_crop, original_img):
    # Calculate the aspect ratio of img
    aspect_ratio = original_img.size[0] / original_img.size[1]

    width, height = img_to_crop.size
    new_width = min(width, height * aspect_ratio)
    new_height = new_width / aspect_ratio

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return img_to_crop.crop((left, top, right, bottom))


def safe_resize(img,  target_width, target_height, pad_color=0):
    # Calculate the aspect ratio of the image
    aspect = img.width / img.height

    # Calculate the target aspect ratio
    target_aspect = target_width / target_height

    # Calculate the new width and height based on the aspect ratio
    if aspect > target_aspect:
        new_width = target_height
        new_height = round(target_height / aspect)
    else:
        new_height = target_width
        new_width = round(target_width * aspect)

    # Resize the image
    img = img.resize((new_width, new_height))

    # Create a new image with the pad color and the target size
    new_img = Image.new("RGB", (target_width, target_height), pad_color)

    # Calculate the position to paste the resized image
    paste_width = (target_width - new_width) // 2
    paste_height = (target_height - new_height) // 2

    # Paste the resized image on the new image
    new_img.paste(img, (paste_width, paste_height))

    return new_img
    


def crop_and_resize_res(img, res):
    cropped_res = center_crop_res(img, res)
    # Resize cropped_res to the size of img
    return cv2.resize(
        cropped_res.numpy(),
        (img.size[0], img.size[1]),
        interpolation=cv2.INTER_LINEAR,
    )


def crop_to_box(img_or_mask, box, size=None):
    bbox = [int(x) for x in box]
    bb_x0, bb_y0, bb_x1, bb_y1 = bbox
    if isinstance(img_or_mask, np.ndarray):
        cropped = img_or_mask[bb_y0:bb_y1, bb_x0:bb_x1]
        return resize_array(cropped, size) if size else cropped
    cropped = img_or_mask.crop(bbox)
    return resize_image(img_or_mask.crop(bbox), size) if size else cropped


def inflate_box(img, x_0, y_0, x_1, y_1, scale_factor=3):
    new_box = np.zeros(4, dtype=int)
    width = x_1 - x_0
    height = y_1 - y_0
    new_width = width * scale_factor
    new_height = height * scale_factor
    center_x = (x_0 + x_1) // 2
    center_y = (y_0 + y_1) // 2
    new_box[0] = max(0, center_x - new_width // 2)
    new_box[1] = max(0, center_y - new_height // 2)
    new_box[2] = min(img.size[0], center_x + new_width // 2)
    new_box[3] = min(img.size[1], center_y + new_height // 2)
    return new_box


def get_box_corners(box):
    vertices = box.vertices
    x_0 = min([vertice.x for vertice in vertices])
    x_1 = max([vertice.x for vertice in vertices])
    y_0 = min([vertice.y for vertice in vertices])
    y_1 = max([vertice.y for vertice in vertices])
    return x_0, y_0, x_1, y_1


def create_mask(img, boxes):
    mask = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
    for box in boxes:
        mask[box[1] : box[3], box[0] : box[2]] = 1
    return mask


def inflate_box_with_ratio(img, box, ratio=0.13, min_scale_factor=3):
    """
    Inflates a box to a size that is ratio*min(img.size).

    Parameters
    img : PIL.Image
        The image that the box is in.
    box : VertexBbox
        The box to inflate.
    ratio : float
        The target ratio the box size should be in relation to the given image

    Returns the inflated box.
    """
    # limit the ratio to 0.13 to prevent the box from becoming too large
    ratio = min(ratio, 0.13)

    try:
        x_0, y_0, x_1, y_1 = get_box_corners(box)
    except:
        x_0, y_0, x_1, y_1 = box
    width = x_1 - x_0
    height = y_1 - y_0
    min_dimension = max(min(width, height), 1)
    # Calculate the target size
    target_size = ratio * min(img.size)

    # Calculate the scale factor, make it at least 3 to prevent the box from becoming too small
    scale_factor = max(target_size / min_dimension, min_scale_factor)
    return inflate_box(img, x_0, y_0, x_1, y_1, scale_factor)
