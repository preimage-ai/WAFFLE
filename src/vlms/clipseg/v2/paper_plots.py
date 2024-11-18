import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pickle
from skimage import exposure


from tqdm.auto import tqdm
import numpy as np
from vlms.clipseg.v2.data import CLIPSegDatum, CLIPSegGT
from vlms.clipseg.v2.clipseg_inference import ClipSegInference
from vlms.clipseg.v2.dataset import DEFAULT_SIZE
from vlms.clipseg.v2.metrics_calculator import set_base_and_ft_results, set_cck5_results
from vlms.clipseg.v2.plot_utils import show_boolean_mask
from vlms.clipseg.v2.sizing_utils import (
    center_crop_to_aspect_ratio,
    center_crop_res,
    resize_image,
)
from PIL import ImageDraw


import numpy as np
from matplotlib import pyplot as plt

C = np.zeros((256, 4), dtype=np.float32)
# lightseagreen color
# C[:, 0] = 32 / 255 # red component
# C[:, 1] = 178 / 255 # green component
# C[:, 2] = 170 / 255 # blue component
C[:, 1] = 1.0  # green component
alpha = 0.9
C[:, -1] = np.linspace(0, alpha, 256)
CMAP = ListedColormap(C)


def modify_scores(scores, gamma):
    scores_ = exposure.adjust_gamma(scores, gamma=gamma)
    normalized_scores = (scores_ - np.min(scores_)) / (
        np.max(scores_) - np.min(scores_)
    )
    return normalized_scores


def draw_boxes(
    img,
    boxes,
    box_color="red",
):
    img_ = img.copy()
    draw = ImageDraw.Draw(img_)
    for box in boxes:
        draw.rectangle(
            (box[0], box[1], box[2], box[3]),
            fill=None,
            outline=box_color,
            width=5
        )
    return img_


def plot_gt_bbox(img, gt, output_path):
    img_ = img.copy()
    img_ = draw_boxes(
        img_,
        [box for value in gt.pos_labels_to_boxes.values() for box in value],
        box_color="orangered",
    )
    # Create a new figure
    _, ax = plt.subplots(1)
    ax.axis("off")
    plt.axis("off")

    # Display the image
    ax.imshow(img_)

    # Show the plot
    plt.savefig(output_path)
    plt.close()


def draw_gt_contours(img, gt_scores, box_color):
    # Convert the mask to an 8-bit single-channel image
    gt_scores_ = (gt_scores * 255).astype(np.uint8)
    # Find the contours of the square
    contours, _ = cv2.findContours(
        gt_scores_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for contour in contours:
        # Get the rectangle that contains the square
        rect = cv2.minAreaRect(contour)
        # Get the corners of the rectangle
        corners = cv2.boxPoints(rect)
        # Convert the corners to integers
        corners = corners.astype(np.uint8)
        # Draw the rectangle
        boxes.append(
            [
                min(corners[:, 0]),
                min(corners[:, 1]),
                max(corners[:, 0]),
                max(corners[:, 1]),
            ]
        )
    return draw_boxes(img, boxes, box_color)


def plot_image_preds(original_img, img, gt_scores, scores, output_path):
    # draw_boxes(
    #     img=img,
    #     boxes=[box for value in gt.pos_labels_to_boxes.values() for box in value],
    #     box_color="orangered",
    # )
    # img_ = draw_gt_contours(img, gt_scores, "orangered")
    img_ = center_crop_to_aspect_ratio(img, original_img)
    scores_ = exposure.adjust_gamma(center_crop_res(original_img, scores), gamma=1)
    normalized_scores = (scores_ - np.min(scores_)) / (
        np.max(scores_) - np.min(scores_)
    )
    # Create a new figure
    _, ax = plt.subplots(1)
    ax.axis("off")
    plt.axis("off")

    # Display the image
    ax.imshow(img_)

    # Display the scores
    ax.imshow(normalized_scores, cmap=CMAP)

    # Show the plot
    plt.savefig(output_path)
    plt.close()


# def plot_image_preds(img, original_img, scores, output_path):
#     img_ = center_crop_to_aspect_ratio(img, original_img)
#     scores_ = exposure.adjust_gamma(center_crop_res(img_, scores), gamma=0.5)
#     normalized_scores = (scores_ - np.min(scores_)) / (
#         np.max(scores_) - np.min(scores_)
#     )
#     # Create a new figure
#     _, ax = plt.subplots(1)

#     # Display the image
#     ax.imshow(img_)

#     # Display the scores
#     ax.imshow(normalized_scores, cmap=CMAP)

#     # Show the plot
#     plt.savefig(output_path)
#     plt.close()


if __name__ == "__main__":
    with open(
        "/scratch/kerenganon/WAFFLE/for_segmentation/common_labels_smaller_bboxes_dataset_test.pkl",
        "rb",
    ) as f:
        test_data_a = pickle.load(f)
    test_data_a = [datum for datum in test_data_a if datum.df_row.page_id == 55924048]

    with open(
        "data/for_segmentation/test_images/segmentation_tests.pkl",
        "rb",
    ) as f:
        test_data_b = pickle.load(f)
    
    with open(
        "data/for_segmentation/test_images_2/segmentation_tests_2.pkl",
        "rb",
    ) as f:
        test_data_c = pickle.load(f)

    test_data = test_data_a + test_data_b
    # test_data = test_data_c
    
    # REMOVE TEMPORARY
    new_test_data = set_cck5_results(test_data_a + test_data_b + test_data_c)
    ft_model_path = (
        "checkpoints/segmentation/common_labels_smaller_bboxes_with_entr_loss/epoch_20"
    )
    plots_dir = os.path.join(ft_model_path, "test_images_for_paper_cc5k")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    for datum in tqdm(new_test_data):
        for gt in datum.gts:
            for label in gt.pos_labels_to_boxes.keys():
                plot_image_preds(
                    original_img=datum.img,
                    img=datum.img_,
                    gt_scores=gt.scores_,
                    scores=gt.cc5k_res_,
                    output_path=f"{plots_dir}/{datum.df_row.page_id}_{label}_ft.png",
                )
    
    # END REMOVE TEMPORARY
    
    page_ids = [402171, 70630004]
    test_data = [datum for datum in test_data if datum.df_row.page_id in page_ids]

    ft_model_path = (
        "checkpoints/segmentation/common_labels_smaller_bboxes_with_entr_loss/epoch_20"
    )
    test_data = set_base_and_ft_results(
        base_model_path="CIDAS/clipseg-rd64-refined",
        ft_model_path=ft_model_path,
        test_data=test_data,
    )

    for datum in tqdm(test_data):
        plots_dir = os.path.join(ft_model_path, "test_images_for_paper")
        # plots_dir = os.path.join(ft_model_path, "test_images_for_paper_v2")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        for gt in datum.gts:
            for label in gt.pos_labels_to_boxes.keys():
                plot_gt_bbox(
                    img=datum.img,
                    gt=gt,
                    output_path=f"{plots_dir}/{datum.df_row.page_id}_{label}_gt.png",
                )
                # plot_image_preds(
                #     original_img=datum.img,
                #     img=datum.img_,
                #     gt_scores=gt.scores_,
                #     scores=gt.scores_,
                #     output_path=f"{plots_dir}/{datum.df_row.page_id}_{label}_gt.png",
                # )
                # plot_image_preds(
                #     original_img=datum.img,
                #     img=datum.img_,
                #     gt_scores=gt.scores_,
                #     scores=gt.base_res_,
                #     output_path=f"{plots_dir}/{datum.df_row.page_id}_{label}_base.png",
                # )
                # plot_image_preds(
                #     original_img=datum.img,
                #     img=datum.img_,
                #     gt_scores=gt.scores_,
                #     scores=gt.ft_res_,
                #     output_path=f"{plots_dir}/{datum.df_row.page_id}_{label}_ft.png",
                # )
    pass
