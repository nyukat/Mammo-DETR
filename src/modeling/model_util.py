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
Common functions shared by all models
"""
#from src.callbacks import evaluation_metrics
from torch import nn
from collections import OrderedDict
import torch, gin


class AttentionFeatureAggregator(nn.Module):
    """
    Feature aggregator that uses gated attention to produce an attention-weighted average of output vector.
    """
    def __init__(self, feature_dim, num_classes=1, hidden_dim=128, mode="ordered"):
        super(AttentionFeatureAggregator, self).__init__()
        self.mil_attn_V = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.mil_attn_U = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.mil_attn_w = nn.Linear(hidden_dim, num_classes, bias=False)
        self.mode = mode

    def calculate_raw_attention_scores(self, img_features):
        attn_projection = torch.sigmoid(self.mil_attn_U(img_features)) * torch.tanh(self.mil_attn_V(img_features))
        raw_attn_score = self.mil_attn_w(attn_projection)
        return raw_attn_score

    def forward_ordered(self, img_features):
        """
        Forward function for cases in which each bag has same number of instances.
        :param img_features: batch_size, n_instance, dim
        :return:
        """
        # for each group, calculate attention score: batch_size,
        raw_attn_scores = self.calculate_raw_attention_scores(img_features)
        # normalize the attention scores
        attn_scores = torch.softmax(raw_attn_scores, dim=1)
        # calcualte attention-weighted average
        attn_weighted_avg_vec = (img_features * attn_scores).sum(dim=1)
        return attn_scores, attn_weighted_avg_vec

    def forward_unordered(self, img_features, group_df):
        """
        Forward function for cases in which each bag has unequal number of instances.
        :param img_features:
        :param group_df:
        :return:
        """
        # for each group, calculate attention score
        raw_attn_scores = self.calculate_raw_attention_scores(img_features)
        # normalize the attention score according to its group
        # in place change
        for group in group_df["img_idx"]:
            # mask out the impact of a percentage of images
            softmax_input = raw_attn_scores[group]
            raw_attn_scores[group] = torch.softmax(softmax_input, dim=0)
        # TODO: sum me up and take average
        return raw_attn_scores, None

    def forward(self, img_features, group_df=None):
        if self.mode == "ordered":
            return self.forward_ordered(img_features)
        elif self.mode == "unordered":
            assert group_df is not None
            return self.forward_unordered(img_features, group_df)


def top_k_percent_pooling(saliency_map, percent_k):
    """
    Function that perform the top k percent pooling
    """
    N, C, W, H = saliency_map.size()
    cam_flatten = saliency_map.view(N, C, -1)
    top_k = int(round(W * H * percent_k))
    selected_area, selected_idx = cam_flatten.topk(top_k, dim=2)
    saliency_map_pred = selected_area.mean(dim=2)
    return saliency_map_pred, selected_idx


# def loss_dice(map_like_pred, cls_labels, mask_labels, match_mode="mask"):
#     pos_dice, seg_missing = evaluation_metrics.calculate_dice_coefficient(map_like_pred, mask_labels, match_mode=match_mode)
#     neg_dice, _ = evaluation_metrics.calculate_dice_coefficient(1.0 - map_like_pred, 1.0 - mask_labels, match_mode=match_mode)
#     res = (cls_labels * (1 - seg_missing).float()) * pos_dice + (1 - cls_labels) * neg_dice
#     loss_dice = 1.0 - res.mean()
#     return loss_dice


def positive_missing_annotation_mask(cls_label, annotations):
    """
    Returns a [N_batch, N_class] torch.Tensor mask where A[i,c] = 0 means ith instance has positive class c
    But don't have any annotation of class c.
    :param cls_label: N_batch, N_class
    :param annotations: [{'labels': torch.Tensor([0])}, {}, ...]
    :return:
    """
    with torch.no_grad():
        has_annotations = torch.zeros(cls_label.size()).to(cls_label.get_device())
        for i in range(len(annotations)):
            bbox_cls_labels = annotations[i]["labels"]
            if len(bbox_cls_labels.size()) > 0 and bbox_cls_labels.size()[0] > 0:
                for c in bbox_cls_labels:
                    has_annotations[i][c] = 1
        cls_negative_mask = 1 - cls_label
        cls_positive_and_has_anno_mask = cls_label * has_annotations
        pos_missing_anno_mask = cls_negative_mask + cls_positive_and_has_anno_mask
        return pos_missing_anno_mask

@gin.configurable
def update_ema_teacher(model_student, model_teacher, ema_keep_rate=0.9996):
    """
    Function that updates the weights of the teacher network using EMA of student network
    :param model_student:
    :param model_teacher:
    :param ema_keep_rate:
    :return:
    """
    student_model_dict = {
        key: value for key, value in model_student.state_dict().items()
        }
    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] * (1 - ema_keep_rate) + value * ema_keep_rate
            )
    model_teacher.load_state_dict(new_teacher_dict)
