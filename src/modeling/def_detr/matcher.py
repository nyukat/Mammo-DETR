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
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch, gin
from scipy.optimize import linear_sum_assignment
from torch import nn
from src.modeling.def_detr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
import random

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 alpha: float = 0.25,
                 gamma: float = 2,
                 distance_function = "center"):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma
        self.distance_function = distance_function
        self.matcher_type = 'hungarian'
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def return_cost_class(self):
        return self.cost_matrix_class

    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            self.cost_matrix_class = cost_class.view(bs, num_queries, -1)

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            
            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
            

def center_point_distance(pred_bbox, label_bbox, w_img, h_img):
    """
    Distance = Euclidean distance of the center points; threshold = 1/2 * bbox_gt's diagonal
    :param pred_bbox:
    :param label_bbox:
    :param w_img:
    :param h_img:
    :return:
    """
    # calculate the distance of each bbox_pred to each bbox_gt
    bbox_to_label_dist = []
    gt_bbox_thresholds = []
    for label_idx in range(label_bbox.size()[0]):
        # calculate distance
        x_dist_sq = torch.square((pred_bbox[:, 0] - label_bbox[label_idx, 0]) * w_img)
        y_dist_sq = torch.square((pred_bbox[:, 1] - label_bbox[label_idx, 1]) * h_img)
        dist = torch.sqrt(x_dist_sq + y_dist_sq).unsqueeze(1)
        bbox_to_label_dist.append(dist)
        # calculate threshold
        min_threshold = 100 * h_img / 2866
        w_box = label_bbox[label_idx, 2] * w_img
        h_box = label_bbox[label_idx, 3] * h_img
        half_diag_threshold = torch.sqrt(h_box * h_box + w_box * w_box) / 2
        threshold = max(min_threshold, half_diag_threshold)
        gt_bbox_thresholds.append(threshold)
    bbox_to_label_dist = torch.cat(bbox_to_label_dist, dim=1)
    gt_bbox_thresholds = torch.Tensor(gt_bbox_thresholds).to(bbox_to_label_dist.device)
    return bbox_to_label_dist, gt_bbox_thresholds


@gin.configurable
def iou_distance(pred_bbox, label_bbox, w_img, h_img, min_iou=0.1):
    """
    Distance = 1.0 - iou, threshold = 1.0 - min_iou
    :param pred_bbox:
    :param label_bbox:
    :param w_img:
    :param h_img:
    :param min_iou:
    :return:
    """
    pred_bbox = box_cxcywh_to_xyxy(pred_bbox)
    label_bbox = box_cxcywh_to_xyxy(label_bbox)
    ious, _ = box_iou(pred_bbox, label_bbox)
    min_iou_distance = torch.ones(label_bbox.size()[0]) * (1 - min_iou)
    return 1 - ious, min_iou_distance.to(pred_bbox.device)


def distance_match(pred_bbox, label_bbox, w_img, h_img, distance_type="center", dist_cost=None, min_dist = 0.2):
    """
    Function that match each each bbox_pred to bbox_gt based on distance.
    :param pred_bbox: num_of_pred_bbox, 4; bbox: (cx, cy, w, h), cx corresponds to w, cy corresponds to h
    :param label_bbox: num_of_gt_bbox, 4
    :param w_img:
    :param h_img:
    :param distance_type:
    :return:
    """
    # calculate the distance of each bbox_pred to each bbox_gt
    if distance_type == "center":
        bbox_to_label_dist, gt_bbox_thresholds = center_point_distance(pred_bbox, label_bbox, w_img, h_img)
    elif distance_type == "iou":
        bbox_to_label_dist, gt_bbox_thresholds = iou_distance(pred_bbox, label_bbox, w_img, h_img)
    elif distance_type == "reference":
        bbox_to_label_dist = dist_cost
        gt_bbox_thresholds = torch.ones(dist_cost.shape[1]) * min_dist
    else:
        raise ValueError(f"bad distance_type {distance_type}")
    # match each bbox_pred to the bbox_gt with shortest distance
    min_dist, min_idx = torch.min(bbox_to_label_dist, dim=1)
    # make sure the min distance is < threshold
    min_threshold = gt_bbox_thresholds[min_idx]
    bbox_matched_flag = min_dist <= min_threshold
    # assign -1 to any bbox_pred that is not matched to any gt
    not_matched_idx = -torch.ones(min_threshold.size()).to(min_threshold.device)
    bbox_matched_idx = torch.where(bbox_matched_flag.bool(), min_idx.long(), not_matched_idx.long())
    return bbox_matched_idx

def find_key_from_value(dictionary, target_value):
    keys = []
    for key, value in dictionary.items():
        if value == target_value:
            keys.append(key)
    return keys

@gin.configurable
class IsTPMatcher(HungarianMatcher):
    """
    Class that modify the Hungarian Matcher results:
    - if a bbox_pred is matched to a bbox_gt by Hungarian Matcher, this match will remains unchanged.
    - if a bbox_pred is not matched to any bbox_gt by Hungarian Matcher, but it's close to a bbox_gt, this bbox_pred
      will be matched to this bbox_gt. Definition of close: distance <= threshold. Distance: center point, iou.
    """
    def __init__(self, cost_class, cost_bbox, cost_giou, alpha, gamma, distance_function="center", ref_distance = 0.1):
        super().__init__(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou, alpha=alpha, gamma=gamma)
        self.is_tp = None
        self.distance_function = distance_function
        self.matcher_type = 'istp'
        self.ref_distance = ref_distance

    def forward(self, outputs, targets, reference_points):
        # Hungarian matcher result
        hungarian_results = super().forward(outputs, targets)
        with torch.no_grad():
            bs, num_queries, _ = reference_points.shape
            out_bbox = reference_points.flatten(0, 1)  # [batch_size * num_queries, 4]
            tgt_bbox = torch.cat([v["boxes"][:,:2] for v in targets])
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=2)
            C = cost_bbox.view(bs, num_queries, -1).cpu()
            sizes = [len(v["boxes"]) for v in targets]

            self.hungarian_results = hungarian_results
            self.cost_matrix_class = super().return_cost_class()
            new_match = []
            duplicate_match = []
            for i, c in enumerate(C.split(sizes, -1)):
                pred_bbox = outputs["pred_boxes"][i]  # N_bbox, 4
                pred_logits = outputs['pred_logits'][i].sigmoid()
                gt_bbox = targets[i]["boxes"] # N_gt, 4
                h_img, w_img = targets[i]["orig_size"]
                
                # only replace Hungarian matcher results on images with gt_box
                if len(gt_bbox) > 0:
                    # add hungarian matched result to match_idx_dict
                    # Hungarian match: {pred_bbox_idx -> gt_bbox_idx}
                    match_idx_dict = {}
                    dup_match_idx_dict = {}
                    hun_pred_idx_tensor, hun_gt_idx_tensor = hungarian_results[i]
                    hun_pred_idx = hun_pred_idx_tensor.data.cpu().numpy()
                    hun_gt_idx = hun_gt_idx_tensor.data.cpu().numpy()
                    for j in range(len(hun_pred_idx)):
                        match_idx_dict[hun_pred_idx[j]] = hun_gt_idx[j]
                    # add distance-based match to match_idx_dict
                    distance_match_result = distance_match(None, None, None, None, 'reference', c[i], min_dist = self.ref_distance)
                    #distance_match_result = distance_match(pred_bbox, gt_bbox, w_img, h_img, self.distance_function)

                    # choose all matched tp box
                    for j in range(len(distance_match_result)):
                        if distance_match_result[j] != -1 and j not in match_idx_dict:
                            #if ref_distance_match_result[j] != -1:
                            # h_js = find_key_from_value(match_idx_dict, distance_match_result[j]) 
                            # if pred_logits[j,1]/pred_logits[h_js].max(0)[0][-1] > 0.7:
                            #     match_idx_dict[j] = distance_match_result[j]
                            # else:
                                # dup_match_idx_dict[j] = distance_match_result[j] 
                            match_idx_dict[j] = distance_match_result[j]
                        else:
                            dup_match_idx_dict[j] = distance_match_result[j] 
                    # convert back to tuple
                    new_match.append((
                        torch.as_tensor(list(match_idx_dict.keys()), dtype=torch.int64),
                        torch.as_tensor(list(match_idx_dict.values()), dtype=torch.int64)
                    ))
                    duplicate_match.append((
                        torch.as_tensor(list(dup_match_idx_dict.keys()), dtype=torch.int64),
                        torch.as_tensor(list(dup_match_idx_dict.values()), dtype=torch.int64)
                    )) 
                else:
                    new_match.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)))
                    duplicate_match.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64))) 
            self.duplicate_match = duplicate_match
        return new_match


@gin.configurable
class ReferenceMatcher(nn.Module):
    """
    Class that modify the Hungarian Matcher results:
    - if a bbox_pred is matched to a bbox_gt by Hungarian Matcher, this match will remains unchanged.
    - if a bbox_pred is not matched to any bbox_gt by Hungarian Matcher, but it's close to a bbox_gt, this bbox_pred
      will be matched to this bbox_gt. Definition of close: distance <= threshold. Distance: center point, iou.
    """
    def __init__(self, distance = 0.1, distance_function = 'center'):
        super().__init__()
        self.distance = distance
        self.matcher_type = 'reference'
        self.distance_function = distance_function
        

    def forward(self, outputs, targets, reference_points):
        with torch.no_grad():
            bs, num_queries, _ = reference_points.shape
    
            out_bbox = reference_points.flatten(0, 1)  # [batch_size * num_queries, 4]
            tgt_bbox = torch.cat([v["boxes"][:,:2] for v in targets])
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=2)
            C = cost_bbox.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            new_match = []
            duplicate_match = []
            for i, c in enumerate(C.split(sizes, -1)):
                pred_bbox = outputs["pred_boxes"][i]  # N_bbox, 4
                gt_bbox = targets[i]["boxes"] # N_gt, 4
                h_img, w_img = targets[i]["orig_size"]
                if sizes[i] != 0:
                    indices = linear_sum_assignment(c[i])
                    match_idx_dict = {}
                    dup_match_idx_dict = {}
                    hun_pred_idx, hun_gt_idx = indices
                    for j in range(len(hun_pred_idx)):
                        match_idx_dict[hun_pred_idx[j]] = hun_gt_idx[j]
                    # add distance-based match to match_idx_dict
                    distance_match_result = distance_match(None, None, None, None, 'reference', c[i], min_dist = self.distance)
                    iou_match_result = distance_match(pred_bbox, gt_bbox, w_img, h_img, self.distance_function)
                    for j in range(len(distance_match_result)):
                        if distance_match_result[j] != -1 and j not in match_idx_dict:# and iou_match_result[j] != -1:
                            match_idx_dict[j] = distance_match_result[j]
                        if j not in match_idx_dict and iou_match_result[j] != -1: 
                            dup_match_idx_dict[j] = iou_match_result[j]
                    # convert back to tuple
                    new_match.append((
                        torch.as_tensor(list(match_idx_dict.keys()), dtype=torch.int64),
                        torch.as_tensor(list(match_idx_dict.values()), dtype=torch.int64)
                    ))
                    duplicate_match.append((
                        torch.as_tensor(list(dup_match_idx_dict.keys()), dtype=torch.int64),
                        torch.as_tensor(list(dup_match_idx_dict.values()), dtype=torch.int64)
                    )) 
                else:
                    new_match.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)))
                    duplicate_match.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64))) 
            self.duplicate_match = duplicate_match
        return new_match


def build_matcher(args):
    if args.matcher_class == "hungarian":
        return HungarianMatcher(cost_class=args.set_cost_class,
                                cost_bbox=args.set_cost_bbox,
                                cost_giou=args.set_cost_giou,
                                alpha=args.matcher_alpha,
                                gamma=args.matcher_gamma)
    elif args.matcher_class == "istp":
        return IsTPMatcher(cost_class=args.set_cost_class,
                                cost_bbox=args.set_cost_bbox,
                                cost_giou=args.set_cost_giou,
                                alpha=args.matcher_alpha,
                                gamma=args.matcher_gamma,
                                ref_distance=args.min_match_dist)
    elif args.matcher_class == 'reference':
        return ReferenceMatcher(distance = args.min_match_dist)
    else:
        raise ValueError(f"no such matcher type {args.matcher_class}")