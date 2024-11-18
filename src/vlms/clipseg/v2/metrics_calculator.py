from argparse import ArgumentParser
import json
import os
import pickle
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import (
    average_precision_score,
    jaccard_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)
from vlms.clipseg.v2.dataset import DEFAULT_SIZE
from vlms.clipseg.v2.sizing_utils import resize_array, resize_image
from vlms.clipseg.v2.data import CLIPSegDatum, CLIPSegGT
from vlms.clipseg.v2.clipseg_inference import ClipSegInference
from collections import namedtuple, Counter
from matplotlib import pyplot as plt

Metrics = namedtuple(
    "Metrics",
    "ap ba jaccard_025 jaccard_05 jaccard_075 dice_025 dice_05 dice_075 label",
)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--test_dataset_fn",
        type=str,
        default="/scratch/kerenganon/WAFFLE/for_segmentation/common_labels_smaller_bboxes_dataset_test.pkl",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="CIDAS/clipseg-rd64-refined",
    )
    parser.add_argument(
        "--ft_model_path",
        type=str,
        default="checkpoints/segmentation/common_labels_smaller_bboxes_with_entr_loss/epoch_20",
    )

    return parser.parse_args()


def get_flattened_values(test_data, label=None):
    gts = [gt for datum in test_data for gt in datum.gts]
    # if label:
    #     gts = [gt for gt in gts if gt.labels[0] == label]
    base = []
    ft = []
    gt = []
    for gt_iter in gts:
        num_labels = len(gt_iter.pos_labels_to_boxes.keys())
        for i in range(num_labels):
            base_res = gt_iter.base_res_ if num_labels == 1 else gt_iter.base_res_[i]
            ft_res = gt_iter.ft_res_ if num_labels == 1 else gt_iter.ft_res_[i]
            scores = gt_iter.scores_
            base.append(base_res.ravel())
            ft.append(ft_res.ravel())
            gt.append(scores.ravel())
    concated_base = np.concatenate(base)
    concated_ft = np.concatenate(ft)
    concated_gt = np.concatenate(gt)
    return concated_base, concated_ft, concated_gt


def plot_roc(test_data, output_path, is_ft, x_axis_limit=1.0, label=None):
    concated_base, concated_ft, concated_gt = get_flattened_values(test_data, label)
    concated_scores = concated_ft if is_ft else concated_base
    fpr, tpr, _ = roc_curve(concated_gt, concated_scores)
    auc = roc_auc_score(concated_gt, concated_scores)
    print(f"AUC for label {label}: {auc}")
    # plot the roc curve for the model
    _, ax = plt.subplots(1, 1)
    ax.set_xlim([0.0, x_axis_limit])
    plt.plot(fpr, tpr, linestyle="--", label="FT" if is_ft else "Base")
    # axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # show the legend
    plt.legend()
    if label:
        plt.title(f"ROC curve for {label}")
    # show the plot
    plt.savefig(output_path)
    plt.close()
    return auc


class MetricsCalculator:
    def __init__(self, is_ft):
        self.is_ft = is_ft

    def process_single_image(self, datum):
        label_to_metrics = {}
        # print(f"Processing {datum.df_row.img_url}")
        for gt in datum.gts:
            try:
                labels = gt.pos_labels_to_boxes.keys()
                for i, label in enumerate(labels):
                    gt_scores = resize_array(gt.scores, DEFAULT_SIZE)

                    gt_scores_flat = gt_scores.ravel()
                    pred_array = gt.ft_res_ if self.is_ft else gt.base_res_
                    if len(labels) > 1:
                        pred_array = pred_array[i]
                    pred_array_flat = pred_array.ravel()

                    # calculate metrics
                    AP = average_precision_score(gt_scores_flat, pred_array_flat)

                    tpr = (
                        gt_scores_flat * pred_array_flat
                    ).sum() / gt_scores_flat.sum()
                    tnr = ((1 - gt_scores_flat) * (1 - pred_array_flat)).sum() / (
                        1 - gt_scores_flat
                    ).sum()

                    balanced_acc = (tpr + tnr) / 2

                    pred_array_flat_thresh025 = pred_array_flat > 0.25
                    jscore025 = jaccard_score(gt_scores_flat, pred_array_flat_thresh025)
                    dice025 = f1_score(gt_scores_flat, pred_array_flat_thresh025)

                    pred_array_flat_thresh05 = pred_array_flat > 0.5
                    jscore05 = jaccard_score(gt_scores_flat, pred_array_flat_thresh05)
                    dice05 = f1_score(gt_scores_flat, pred_array_flat_thresh05)

                    pred_array_flat_thresh075 = pred_array_flat > 0.75
                    jscore075 = jaccard_score(gt_scores_flat, pred_array_flat_thresh075)
                    dice075 = f1_score(gt_scores_flat, pred_array_flat_thresh075)

                    label_to_metrics[label] = Metrics(
                        AP,
                        balanced_acc,
                        jscore025,
                        jscore05,
                        jscore075,
                        dice025,
                        dice05,
                        dice075,
                        label,
                    )
            except Exception as e:
                print(f"Error processing {datum.df_row.img_url}: {e}")

        return label_to_metrics


def calc_cc5k_metrics(test_data, output_path):
    concated_gt = np.concatenate(
        [gt.scores_.ravel() for datum in test_data for gt in datum.gts]
    )
    concated_cc5k = np.concatenate(
        [gt.cc5k_res_.ravel() for datum in test_data for gt in datum.gts]
    )
    auc = roc_auc_score(concated_gt, concated_cc5k)

    def process_single_image(datum):
        label_to_metrics = {}
        for gt in datum.gts:
            try:
                labels = gt.pos_labels_to_boxes.keys()
                for i, label in enumerate(labels):
                    gt_scores_flat = gt.scores_.ravel()
                    pred_array = gt.cc5k_res_
                    pred_array_flat = pred_array.ravel()

                    # calculate metrics
                    AP = average_precision_score(gt_scores_flat, pred_array_flat)

                    tpr = (
                        gt_scores_flat * pred_array_flat
                    ).sum() / gt_scores_flat.sum()
                    tnr = ((1 - gt_scores_flat) * (1 - pred_array_flat)).sum() / (
                        1 - gt_scores_flat
                    ).sum()

                    balanced_acc = (tpr + tnr) / 2

                    pred_array_flat_thresh025 = pred_array_flat > 0.25
                    jscore025 = jaccard_score(gt_scores_flat, pred_array_flat_thresh025)
                    dice025 = f1_score(gt_scores_flat, pred_array_flat_thresh025)

                    pred_array_flat_thresh05 = pred_array_flat > 0.5
                    jscore05 = jaccard_score(gt_scores_flat, pred_array_flat_thresh05)
                    dice05 = f1_score(gt_scores_flat, pred_array_flat_thresh05)

                    pred_array_flat_thresh075 = pred_array_flat > 0.75
                    jscore075 = jaccard_score(gt_scores_flat, pred_array_flat_thresh075)
                    dice075 = f1_score(gt_scores_flat, pred_array_flat_thresh075)

                    label_to_metrics[label] = Metrics(
                        AP,
                        balanced_acc,
                        jscore025,
                        jscore05,
                        jscore075,
                        dice025,
                        dice05,
                        dice075,
                        label,
                    )
            except Exception as e:
                print(f"Error processing {datum.df_row.img_url}: {e}")

        return label_to_metrics

    all_metrics = [process_single_image(datum) for datum in test_data]

    all_labels_to_metrics = {}
    for label_to_metric in all_metrics:
        for label, metric in label_to_metric.items():
            if label not in all_labels_to_metrics:
                all_labels_to_metrics[label] = []
            all_labels_to_metrics[label].append(metric)

    json.dump(
        all_labels_to_metrics,
        open(output_path.replace(".txt", "_all_labels.json"), "w"),
        indent=4,
    )

    all_metrics_flattened = [
        metric for _, metrics in all_labels_to_metrics.items() for metric in metrics
    ]

    f = open(output_path, "w")

    print("all_category_metric :")
    ap = np.mean([m.ap for m in all_metrics_flattened if not np.isnan(m.ap)])
    f.write("\tAP (average precision):\t" + str(ap) + "\n")
    print("\tAP (average precision):\t" + str(ap))
    ba = np.mean([m.ba for m in all_metrics_flattened if not np.isnan(m.ba)])
    f.write("\tBalanced accuracy:\t" + str(ba) + "\n")
    print("\tBalanced accuracy:\t" + str(ba))

    jaccard025 = np.mean(
        [m.jaccard_025 for m in all_metrics_flattened if not np.isnan(m.jaccard_025)]
    )
    f.write("\tJaccard score (IoU):\t" + str(jaccard025) + f" (Threshold: 0.25)\n")
    jaccard05 = np.mean(
        [m.jaccard_05 for m in all_metrics_flattened if not np.isnan(m.jaccard_05)]
    )
    f.write("\tJaccard score (IoU):\t" + str(jaccard05) + f" (Threshold: 0.5)\n")
    jaccard075 = np.mean(
        [m.jaccard_075 for m in all_metrics_flattened if not np.isnan(m.jaccard_075)]
    )
    f.write("\tJaccard score (IoU):\t" + str(jaccard075) + f" (Threshold: 0.75)\n")

    dice025 = np.mean(
        [m.dice_025 for m in all_metrics_flattened if not np.isnan(m.dice_025)]
    )
    f.write("\tDice score (F1):\t" + str(dice025) + f" (Threshold: 0.25)\n")
    dice05 = np.mean(
        [m.dice_05 for m in all_metrics_flattened if not np.isnan(m.dice_05)]
    )
    f.write("\tDice score (F1):\t" + str(dice05) + f" (Threshold: 0.5)\n")
    dice075 = np.mean(
        [m.dice_075 for m in all_metrics_flattened if not np.isnan(m.dice_075)]
    )
    f.write("\tDice score (F1):\t" + str(dice075) + f" (Threshold: 0.75)\n")

    f.write("\tAUC:\t" + str(auc) + "\n")
    print("\tAUC:\t" + str(auc))
    f.write("\n")
    f.close()


def calc_metrics(test_data, is_ft, output_path):
    metrics_calculator = MetricsCalculator(is_ft)

    auc = plot_roc(
        test_data, output_path=output_path.replace(".txt", "_roc.png"), is_ft=is_ft
    )

    all_metrics = [
        metrics_calculator.process_single_image(datum) for datum in test_data
    ]
    all_labels_to_metrics = {}
    for label_to_metric in all_metrics:
        for label, metric in label_to_metric.items():
            if label not in all_labels_to_metrics:
                all_labels_to_metrics[label] = []
            all_labels_to_metrics[label].append(metric)

    json.dump(
        all_labels_to_metrics,
        open(output_path.replace(".txt", "_all_labels.json"), "w"),
        indent=4,
    )

    all_metrics_flattened = [
        metric for _, metrics in all_labels_to_metrics.items() for metric in metrics
    ]

    f = open(output_path, "w")

    print("all_category_metric :")
    ap = np.mean([m.ap for m in all_metrics_flattened if not np.isnan(m.ap)])
    f.write("\tAP (average precision):\t" + str(ap) + "\n")
    print("\tAP (average precision):\t" + str(ap))
    ba = np.mean([m.ba for m in all_metrics_flattened if not np.isnan(m.ba)])
    f.write("\tBalanced accuracy:\t" + str(ba) + "\n")
    print("\tBalanced accuracy:\t" + str(ba))

    jaccard025 = np.mean(
        [m.jaccard_025 for m in all_metrics_flattened if not np.isnan(m.jaccard_025)]
    )
    f.write("\tJaccard score (IoU):\t" + str(jaccard025) + f" (Threshold: 0.25)\n")
    jaccard05 = np.mean(
        [m.jaccard_05 for m in all_metrics_flattened if not np.isnan(m.jaccard_05)]
    )
    f.write("\tJaccard score (IoU):\t" + str(jaccard05) + f" (Threshold: 0.5)\n")
    jaccard075 = np.mean(
        [m.jaccard_075 for m in all_metrics_flattened if not np.isnan(m.jaccard_075)]
    )
    f.write("\tJaccard score (IoU):\t" + str(jaccard075) + f" (Threshold: 0.75)\n")

    dice025 = np.mean(
        [m.dice_025 for m in all_metrics_flattened if not np.isnan(m.dice_025)]
    )
    f.write("\tDice score (F1):\t" + str(dice025) + f" (Threshold: 0.25)\n")
    dice05 = np.mean(
        [m.dice_05 for m in all_metrics_flattened if not np.isnan(m.dice_05)]
    )
    f.write("\tDice score (F1):\t" + str(dice05) + f" (Threshold: 0.5)\n")
    dice075 = np.mean(
        [m.dice_075 for m in all_metrics_flattened if not np.isnan(m.dice_075)]
    )
    f.write("\tDice score (F1):\t" + str(dice075) + f" (Threshold: 0.75)\n")

    f.write("\tAUC:\t" + str(auc) + "\n")
    print("\tAUC:\t" + str(auc))
    f.write("\n")
    f.close()


def set_base_and_ft_results(base_model_path, ft_model_path, test_data):
    base_inference = ClipSegInference(base_model_path)
    ft_inference = ClipSegInference(ft_model_path)
    for datum in tqdm(test_data):
        datum.img_ = resize_image(datum.img, DEFAULT_SIZE)
        for gt in datum.gts:
            base_res = (
                base_inference.call(datum.img_, list(gt.pos_labels_to_boxes.keys()))
                .cpu()
                .numpy()
            )
            ft_res = (
                ft_inference.call(datum.img_, list(gt.pos_labels_to_boxes.keys()))
                .cpu()
                .numpy()
            )
            gt.base_res_ = base_res
            gt.ft_res_ = ft_res
            gt.scores_ = resize_array(gt.scores, DEFAULT_SIZE)
    return test_data


def set_cck5_results(test_data):
    import torch

    label2id = {
        "porch": 0,  # in CC5K this is "outdoors"
        "kitchen": 1,
        "living room": 2,
        "bedroom": 3,
        "bath": 4,
        "entry": 5,
    }
    cc5k_fns = os.listdir("data/for_segmentation/c5k_preds/residential")
    cc5k_page_ids = [int(fn.split(".")[0].split("_")[1]) for fn in cc5k_fns]

    new_data = []
    for datum in tqdm(test_data):
        datum.img_ = resize_image(datum.img, DEFAULT_SIZE)
        page_id = datum.df_row.page_id
        if page_id not in cc5k_page_ids:
            continue
        tensor_fn = [fn for fn in cc5k_fns if str(page_id) in fn][0]
        try:
            cc5k_tensor = torch.load(
                f"data/for_segmentation/c5k_preds/residential/{tensor_fn}",
                map_location=torch.device("cpu"),
                weights_only=True,
            )
        except Exception as e:
            print(f"Error loading {page_id}: {e}")
            continue
        new_gts = []
        for gt in datum.gts:
            matching_labels = [
                label
                for label in label2id.keys()
                for key in gt.pos_labels_to_boxes.keys()
                if label in key
            ]
            if (
                len(matching_labels) == 0 or len(matching_labels) > 1
            ):  # in this case we should have at most one label per gt
                continue
            label = matching_labels[0]

            cc5k_res = resize_array(cc5k_tensor[label2id[label]].numpy(), DEFAULT_SIZE)
            if np.max(cc5k_res) == 0.0:
                # update one pixel to a small value to avoid roc_auc_score error
                cc5k_res[0, 0] = 1e-6
            gt.cc5k_res_ = cc5k_res
            gt.scores_ = resize_array(gt.scores, DEFAULT_SIZE)
            new_gts.append(gt)
        datum.gts = new_gts
        new_data.append(datum)
    return new_data


if __name__ == "__main__":
    args = get_args()

    with open(args.test_dataset_fn, "rb") as f:
        test_data = pickle.load(f)
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

    test_data = test_data + test_data_b + test_data_c

    # images from the test set that don't have good GTs
    page_ids_to_remove = [
        103050112,
        106435727,
        11048852,
        115163635,
        18319037,
        18335153,
        20214015,
        36195035,
        36966312,
        4101040,
        42614834,
        7221344,
        7279838,
        7357337,
        76123872,
        84589678,
    ]

    test_data = [
        datum for datum in test_data if datum.df_row.page_id not in page_ids_to_remove
    ]

    # REMOVE TEMPORARY
    # new_test_data = set_cck5_results(test_data)
    # metrics_dir = os.path.join(args.ft_model_path, "metrics")
    # calc_cc5k_metrics(new_test_data, os.path.join(metrics_dir, "cc5k.txt"))

    # END REMOVE TEMPORARY

    # test_data = set_base_and_ft_results(
    #     args.base_model_path, args.ft_model_path, test_data
    # )

    metrics_dir = os.path.join(args.ft_model_path, "metrics")
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    categories = ["resident", "church", "castle"]
    for category in categories:
        if category == "church":
            cat_test_data = [
                datum
                for datum in test_data
                if "church" in datum.df_row.building_type
                or "cathedral" in datum.df_row.building_type
            ]
        else:
            cat_test_data = [
                datum for datum in test_data if category in datum.df_row.building_type
            ]

        calc_metrics(
            cat_test_data,
            is_ft=False,
            output_path=os.path.join(metrics_dir, f"{category}_base.txt"),
        )
        calc_metrics(
            cat_test_data,
            is_ft=True,
            output_path=os.path.join(metrics_dir, f"{category}_ft.txt"),
        )

    calc_metrics(
        test_data, is_ft=False, output_path=os.path.join(metrics_dir, "base.txt")
    )
    calc_metrics(test_data, is_ft=True, output_path=os.path.join(metrics_dir, "ft.txt"))
    pass
