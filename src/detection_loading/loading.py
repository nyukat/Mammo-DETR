# Copyright (C) 2020 Yanqi Xu, Yiqiu Shen, Laura Heacock, Carlos Fernandez-Granda, Krzysztof J. Geras

# This file is part of Mammo-DETR.
#
# Mammo-DETR is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Mammo-DETR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Mammo-DETR.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tqdm
import cv2
import scipy
from PIL import Image
import pickle
import src.detection_loading.loading_mammogram as loading_mammogram                
import src.detection_loading.duke as duke                

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

class ConvertCocoPolysToMask(object):
    """
    This function is copied from https://github.com/facebookresearch/detr/blob/091a817eca74b8b97e35e4531c1c39f89fbe38eb/datasets/coco.py
    """
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, img_size, target):
        w, h = img_size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        # NOTE: original code transforms box to [x0, y0, x1, y1]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        # NOTE: I made the change to change the box to [cx, cy, width, height] format
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        #target["original_box"] = [obj["bbox"] for obj in anno]
        target["orig_size"] = torch.tensor([int(h), int(w)])
        target["size"] = torch.tensor([int(h), int(w)])
        return target

def collect_annotations(bseg, mseg):
    if np.sum(bseg) == 0:
        banno = []
    else:
        banno = duke.DDSMAugmenter.get_connected_components(bseg, 0, "benign")

    if np.sum(mseg) == 0:
        manno = []
    else:
        manno = duke.DDSMAugmenter.get_connected_components(mseg, 1, "malignant")
    return banno + manno

def load_segmentation_mammogram(meta_data, view, seg_dir, crop_size=(2944,1920)):
    """
    Load segmentation and return the numpy matrices
    :param meta_data:
    :param seg_dir:
    :param crop_size:
    :return:
    """
    # When there is no lesions, return zero masks.
    # Assumption: lesions is a field in the metadata and lesions is a list.

    short_file_path = meta_data[view][0]

    benign_seg_path = os.path.join(seg_dir, f"{short_file_path}_benign.png")
    malignant_seg_path = os.path.join(seg_dir, f"{short_file_path}_malignant.png")
    benign_seg_np = np.zeros(crop_size)
    malignant_seg_np = np.zeros(crop_size)

    if os.path.exists(benign_seg_path):
        
        benign_seg_np += loading_mammogram.load_mammogram_img(benign_seg_path, crop_size, view,
                                                                meta_data["best_center"][view][0],
                                                                meta_data["horizontal_flip"])
        benign_seg_np = scipy.ndimage.binary_erosion(benign_seg_np, iterations=5)
        benign_seg_np = scipy.ndimage.binary_dilation(benign_seg_np, iterations=5)

    if os.path.exists(malignant_seg_path):
        malignant_seg_np += loading_mammogram.load_mammogram_img(malignant_seg_path, crop_size, view,
                                                                meta_data["best_center"][view][0],
                                                                meta_data["horizontal_flip"])
        malignant_seg_np = scipy.ndimage.binary_erosion(malignant_seg_np, iterations=5)
        malignant_seg_np = scipy.ndimage.binary_dilation(malignant_seg_np, iterations=5)   
    return benign_seg_np, malignant_seg_np

def load_mammogram_img(meta_data, view, img_dir,  crop_size=(2944,1920)):
    """
    Function that loads a mammogram image using the meta data
    :param meta_data:
    :param img_dir:
    :param crop_size:
    :return:
    """
    img_path = os.path.join(img_dir, f"{meta_data[view][0]}.png")
    #loading_view = meta_data["View"][0] + "-" + meta_data["View"][1:
    img = loading_mammogram.load_mammogram_img(img_path, crop_size, view,
                                               meta_data["best_center"][view][0], meta_data["horizontal_flip"])
    img_pil = Image.fromarray(img / img.max())
    return img_pil

def load_single_image(meta_data, index, view, img_dir, seg_dir,transformations, anno_prepare_func = ConvertCocoPolysToMask()):
     
    # step #1: load pil images
    img_pil = load_mammogram_img(meta_data, view, img_dir)

    # step #3: load segmentations
    bseg_np, mseg_np = load_segmentation_mammogram(meta_data, view, seg_dir)
    bseg_pil = Image.fromarray(bseg_np.astype("uint8"))
    mseg_pil = Image.fromarray(mseg_np.astype("uint8"))

    # step #4: transformation
    sample = {"img": img_pil, "bseg": bseg_pil, "mseg": mseg_pil}
    res = transformations(sample)
    img, bseg, mseg = res["img"], res["bseg"], res["mseg"]
    bseg = (bseg>0).float()
    mseg = (mseg > 0).float()

    # step #5: segmentation to bounding boxes
    _, h, w = img.size()
    annotations = load_detection_from_mask(index, h, w, bseg.data.numpy()[0, :, :], mseg.data.numpy()[0, :, :], anno_prepare_func) 
    return img, (bseg_np, mseg_np), annotations

def load_detection_from_mask(index, orig_h, orig_w, bseg, mseg, anno_prepare_func = ConvertCocoPolysToMask()):
    # step #5: segmentation to bounding boxes
    raw_annotations = collect_annotations(bseg, mseg)
    # transform to coco format {"bbox":[x,y,width,height]}
    annotations = []
    for i in range(len(raw_annotations)):
        orig_anno = raw_annotations[i]
        anno = {"segmentation": [],  # TODO: add segmentation later
                "area": orig_anno["Width"] * orig_anno["Height"],  # TODO: adjust area later
                "bbox": [orig_anno["X"], orig_anno["Y"], orig_anno["Width"], orig_anno["Height"]],
                "category_id": orig_anno["Class"],  # 0 benign, 1 malignant
                "image_id": index,
                "iscrowd": 0,
                }
        annotations.append(anno)
    
    annotations = anno_prepare_func((orig_h, orig_w), {"image_id": index, "annotations": annotations})
    return annotations