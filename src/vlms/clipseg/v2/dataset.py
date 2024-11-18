import pickle
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from vlms.clipseg.v2.sizing_utils import resize_array, resize_image

DEFAULT_SIZE=352

class CLIPSegTransform:
    def __init__(self): ...

    def random_crop(self, img, scores, mask):
        W, H = img.size
        w = int(W * np.random.uniform(0.7, 1))
        h = int(H * np.random.uniform(0.7, 1))
        x0 = np.random.randint(W - w)
        y0 = np.random.randint(H - h)
        img = TF.crop(img, y0, x0, h, w)
        scores = np.array(TF.crop(Image.fromarray(scores), y0, x0, h, w))
        mask = np.array(TF.crop(Image.fromarray(mask), y0, x0, h, w))
        return img, scores, mask

    def random_contrast(self, img):
        contrast_factor = np.random.uniform(0.0, 2.0)
        img = TF.adjust_contrast(img, contrast_factor=contrast_factor)
        return img

    def random_resize(self, img, scores, mask):
        W, H = img.size
        H2 = int(H * np.random.uniform(0.7, 1))
        W2 = int(W * np.random.uniform(0.7, 1))
        img = TF.resize(img, (H2, W2))
        scores = np.array(
            TF.resize(
                Image.fromarray(scores),
                (H2, W2),
                interpolation=InterpolationMode.NEAREST,
            )
        )
        mask = np.array(
            TF.resize(
                Image.fromarray(mask),
                (H2, W2),
                interpolation=InterpolationMode.NEAREST,
            )
        )
        # Note: important to use NEAREST interpolation on masks so that indices are not changed
        return img, scores, mask

    def __call__(self, img, scores, mask):
        old_img, old_gt_mask, old_mask = img, scores, mask

        if np.random.random() > 0.5:
            img, scores, mask = self.random_crop(img, scores, mask)
            while np.sum(scores) == 0:
                # make sure the label's gt is not completely cropped out
                if np.random.random() > 0.9:
                    # print 10% of the times
                    print("Warning: gt cropped out, trying again")
                img, scores, mask = self.random_crop(old_img, old_gt_mask, old_mask)

        if np.random.random() > 0.5:
            img = self.random_contrast(img)

        if np.random.random() < 0.5:
            img, scores, mask = self.random_resize(img, scores, mask)
        return img, scores, mask


class CLIPSegItem:
    def __init__(self, img, scores, mask, label):
        self.img = img
        self.scores = scores
        self.mask = mask
        self.label = label


class CLIPSegDataset(Dataset):
    def __init__(self, pckl_path, augment=False):
        with open(pckl_path, "rb") as f:
            dataset = pickle.load(f)
        self.lookup_table = {}
        for datum in dataset:
            for gt in datum.gts:
                for label in gt.pos_labels_to_boxes.keys():
                    idx = len(self.lookup_table)
                    self.lookup_table[idx] = (datum.img, gt, label)
        self.augment = augment
        self.transform = CLIPSegTransform()

    def __len__(self):
        return len(self.lookup_table)

    def __getitem__(self, idx):
        img, gt, label = self.lookup_table[idx]
        scores = gt.scores
        mask = gt.mask
        if self.augment:
            img, scores, mask = self.transform(img, scores, mask)
        return CLIPSegItem(
            img=resize_image(img, DEFAULT_SIZE),
            scores=resize_array(scores, DEFAULT_SIZE),
            mask=resize_array(mask, DEFAULT_SIZE),
            label=label,
        )
