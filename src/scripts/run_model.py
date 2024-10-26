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
# ------------------------------------------------------------------------

"""
Script that executes the model pipeline.
"""

import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tqdm
import cv2
import matplotlib.cm as cm
from src.utilities import pickling, tools
from src.constants import VIEWS, PERCENT_T_DICT
from src.detection_loading import loading, transformations
from src.modeling.def_detr.deformable_detr import build_deformable_detr
from PIL import Image
import pickle

def cxcywh_to_x0y0wh(bbox):
    cx, cy, w, h = bbox
    x0 = cx - w/2
    y0 = cy - h/2
    return x0, y0, w, h

def scale_bbox(bbox, img_w, img_h):
    x0, y0, ww, hh = bbox
    return x0*img_w, y0*img_h, ww*img_w, hh*img_h

def collect_topk_boxes_and_logits(boxes, pred_logits, topk=50, include_background=True):
    """
    Function that retrieve topk boxes according to pred_logits from boxes
    :param boxes: N, X, 4
    :param pred_logits: N, X, num_class
    :param topk:
    :return: N, num_class, topk, 4
    """
    batch_size, _, n_class = pred_logits.size()
    if include_background:
        n_class -= 1
    logits, top_box_idx = pred_logits.topk(dim=1, k=topk)
    all_instance_top_boxes = []
    for i in range(batch_size):
        instance_top_boxes = []
        for c in range(n_class):
            ic_top_boxes = [boxes[i, top_box_idx[i, j, c], :].unsqueeze(0) for j in range(topk)]
            ic_top_boxes = torch.cat(ic_top_boxes, dim=0)
            instance_top_boxes.append(ic_top_boxes.unsqueeze(0))
        instance_top_boxes = torch.cat(instance_top_boxes, dim=0)
        all_instance_top_boxes.append(instance_top_boxes.unsqueeze(0))
    all_instance_top_boxes = torch.cat(all_instance_top_boxes, dim=0)
    return all_instance_top_boxes, logits, top_box_idx[:,:,1]


def fetch_cancer_label_by_view(view, cancer_label):
    """
    Function that fetches cancer label using a view
    """
    if view in ["L-CC", "L-MLO"]:
        return cancer_label["left_benign"], cancer_label["left_malignant"]
    elif view in ["R-CC", "R-MLO"]:
        return cancer_label["right_benign"], cancer_label["right_malignant"]



def run_model(model, exam_list, parameters):
    """
    Run the model over images in sample_data.
    Save the predictions as csv and visualizations as png.
    """
    if (parameters["device_type"] == "gpu") and torch.has_cudnn:
        device = torch.device("cuda:{}".format(parameters["gpu_number"]))
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # initialize data holders
    pred_dict = {"image_index": [], "benign_pred": [], "malignant_pred": [],
     "benign_label": [], "malignant_label": [], "annotations":[], "box_pred":[],"box_annotation":[]}
    index = 0
    with torch.no_grad():
        # iterate through each exam
        for datum in tqdm.tqdm(exam_list):
            for view in VIEWS.LIST:
                
                # load image
                short_file_path = datum[view][0]
                index += 1

                H, W = (2944,1920)
                test_transformation =  transformations.compose_transform(augmentation=None, resize=(H, W), image_format='greyscale') 
                loaded_image, (benign_seg, malignant_seg), annotations = loading.load_single_image(datum, index, view, parameters["image_path"], parameters["segmentation_path"], test_transformation)

                # convert python 2D array into 4D torch tensor in N,C,H,W format
                tensor_batch = loaded_image.unsqueeze(0).to(device)
                # forward propagation
                output_dict = model(tensor_batch)
  
                prob_pred = output_dict["pred_logits"]
                # make sure cancer predictions are probability rather than logits
                if prob_pred.sum() < 0:
                    prob_pred = torch.exp(prob_pred)
                prob_pred_cpu = prob_pred.data.cpu()
                box_pred_cpu = output_dict["pred_boxes"].data.cpu()
                top_boxes, top_logits, _ = collect_topk_boxes_and_logits(box_pred_cpu, prob_pred_cpu, topk=5)
                
                pred_numpy, _ = top_logits.max(dim=1)
                benign_pred, malignant_pred = pred_numpy[0, 0], pred_numpy[0, 1]

                # propagate holders
                benign_label, malignant_label = fetch_cancer_label_by_view(view, datum["cancer_label"])
                pred_dict["image_index"].append(short_file_path)
                pred_dict["benign_pred"].append(benign_pred)
                pred_dict["malignant_pred"].append(malignant_pred)
                pred_dict["benign_label"].append(benign_label)
                pred_dict["malignant_label"].append(malignant_label)
                pred_dict["annotations"].append(annotations)
                pred_dict["box_pred"].append(top_logits)
                pred_dict["box_annotation"].append(top_boxes)
           
    return pred_dict


def run_single_model(model_path, data_path, parameters):
    """
    Load a single model and run on sample data
    """
    # construct model
    model, criterion, postprocessors = build_deformable_detr()

    # load parameters
    if parameters["device_type"] == "gpu":
        model.load_state_dict(torch.load(model_path)['model'], strict=True)
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu")['model'], strict=True)
    # load metadata
    exam_list = pickling.unpickle_from_file(data_path)
    # run the model on the dataset
    output_dict = run_model(model, exam_list, parameters)
    return output_dict


def start_experiment(model_path, data_path, output_path, model_index, parameters):
    """
    Run the model on sample data and save the predictions as a csv file
    """
    # make sure model_index is valid
    valid_model_index = ["1", "2", "3"]
    assert model_index in valid_model_index, "Invalid model_index {0}. Valid options: {1}".format(model_index, valid_model_index)
    # create directories
    os.makedirs(output_path, exist_ok=True)

    # set percent_t for the model
    single_model_path = os.path.join(model_path, "model_checkpoints_{0}.pt".format(model_index))
    output_dict = run_single_model(single_model_path, data_path, parameters)

    # save the predictions
    with open(os.path.join(output_path, "predictions.pkl"), 'wb') as f:
        pickle.dump(output_dict, f)



def main():
    # retrieve command line arguments
    parser = argparse.ArgumentParser(description='Run Mammo-DETR on the sample data')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--segmentation-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    parser.add_argument("--gpu-number", type=int, default=0)
    parser.add_argument("--model-index", type=str, default="1")
    args = parser.parse_args()

    parameters = {
    }
    start_experiment(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        model_index=args.model_index,
        parameters=parameters,
    )

if __name__ == "__main__":
    main()
