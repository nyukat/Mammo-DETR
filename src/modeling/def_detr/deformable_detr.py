# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math,time
from src.modeling.def_detr.transformer import TransformerDecoder, TransformerDecoderLayer
from src.modeling.def_detr.util import box_ops
from src.modeling.def_detr.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy
from src.modeling import model_util



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Classification_Head(nn.Module):
    def __init__(self, hidden_dim = 256, num_classes=2, num_layers= 6):
        super().__init__()
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_layers)])
        self.num_layers = num_layers
    def forward(self, hs):
        pred = []
        for i in range(self.num_layers):
            pred.append(self.class_embed[i](hs[i]))
        return pred

class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        self.transformer.decoder.with_box_refine = with_box_refine
        if with_box_refine or two_stage:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward_original(self, samples, metadata):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, \
            enc_outputs_coord_unact, memory = self.transformer(srcs, masks, pos, query_embeds)
        
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    def feature_extractor(self, samples):
        """
        :param samples:
        :return: projected_src: N, 256, hh, ww
        :return: pos: [tensor(N, 256, hh, ww)]
        :return: mask: N, hh, ww, binary tensor
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        self.pre_proj_features = features

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        return srcs, pos, masks

    def transformer_fusion(self, srcs, masks, pos):
        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
            
        hs, init_reference, inter_references, enc_outputs_class, \
            enc_outputs_coord_unact, memory = self.transformer(srcs, masks, pos, query_embeds)
        return hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, memory

    def compute_heads(self, hs, init_reference, inter_references, memory):
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        return outputs_class, outputs_coord

    def prepare_outputs(self, outputs_class, outputs_coord, enc_outputs_coord_unact, enc_outputs_class):
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    def forward(self, samples, return_queries=False):  
        srcs, pos, masks = self.feature_extractor(samples)
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, memory = self.transformer_fusion(srcs, masks, pos)
        outputs_class, outputs_coord = self.compute_heads(hs, init_reference, inter_references, memory)
        out = self.prepare_outputs(outputs_class, outputs_coord, enc_outputs_coord_unact, enc_outputs_class)     
        if return_queries:
            return out, hs
        else: 
            return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25,
                 pos_focal_gamma=2, neg_focal_gamma=2, use_pseudo_label=False, mask_out_pos_no_anno_cases=True,
                 focal_positive_weight=None, use_hungarian_for_bbox_loss=False, mask_out_duplicate_tp=False, use_hungarian_for_ce_loss=False,
                 temperature = 50):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.pos_focal_gamma = pos_focal_gamma
        self.neg_focal_gamma = neg_focal_gamma
        self.focal_alpha = focal_alpha
        self.use_pseudo_label = use_pseudo_label
        self.mask_out_pos_no_anno_cases = mask_out_pos_no_anno_cases
        self.mask_out_duplicate_tp = mask_out_duplicate_tp
        self.focal_positive_weight = focal_positive_weight
        self.temperature = temperature
        if self.focal_positive_weight is not None:
            self.focal_positive_weight = torch.Tensor(self.focal_positive_weight)
        self.use_hungarian_for_bbox_loss = use_hungarian_for_bbox_loss
        self.use_hungarian_for_ce_loss = use_hungarian_for_ce_loss
        self.loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'reference': self.loss_reference_dist,
        }

    @staticmethod
    def update_pseudo_label(cls_labels, target_classes_onehot, src_logits, k=1):
        # find which instance, class combo has missing annotation
        has_anno_onehot = (target_classes_onehot.sum(1) > 0).int()
        missing_anno = cls_labels > has_anno_onehot
        # create pseudo label: top k prediction is set to 1
        pseudo_onehot = torch.full(target_classes_onehot.size(), 0, dtype=torch.int64,
                                   device=target_classes_onehot.device)
        topk_val, _ = torch.topk(src_logits, k, dim=1)
        threshold, _ = topk_val.min(dim=1)
        topk_mask = src_logits >= threshold.unsqueeze(1)
        pseudo_onehot[topk_mask] = 1
        # impute pseudo label for instance, class combo that miss annotation
        processed_target_onehot = torch.where(missing_anno.unsqueeze(1), pseudo_onehot.float(), target_classes_onehot)
        return processed_target_onehot

    def get_clean_indices(self,targets,indices):
        new_indices = []
        for t, (j, i) in zip(targets, indices): 
            idx = (i<len(t['boxes'])).nonzero().view(-1)
            x = i[idx]
            y = j[idx]
            new_indices.append((y,x))
        return new_indices

    def _get_duplicate_indices_mask(self, targets, src_logits, indices):
        dup_target_classes_onehot, dup_idx = self._get_target_classes_onehot(src_logits, targets, indices)
        duplicate_mask = torch.abs(dup_target_classes_onehot-1)
        return duplicate_mask

    def focal_loss_with_masked_istp_cases(self, outputs, targets, indices, cls_labels):
        src_logits = outputs
        target_classes_onehot, idx = self._get_target_classes_onehot(src_logits, targets, indices)

        # impute pseudo labels for positive cases where annotation is missing
        if self.use_pseudo_label:
            target_classes_onehot = SetCriterion.update_pseudo_label(cls_labels, target_classes_onehot, src_logits, k=5)

        focal_loss_mask = torch.ones_like(src_logits, device = src_logits.device)
        # compute a mask that zero out positive w/o annotations
        if self.mask_out_pos_no_anno_cases:
            focal_loss_mask = focal_loss_mask * model_util.positive_missing_annotation_mask(cls_labels, targets).unsqueeze(1)
        # compute a mask that zero out istp cases
 
        focal_loss_mask = focal_loss_mask * self._get_duplicate_indices_mask(targets, src_logits, self.matcher.duplicate_match)
        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)
        
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha,
                                        mask=focal_loss_mask, pos_gamma = self.pos_focal_gamma, neg_gamma=self.neg_focal_gamma,
                                        pos_weight=self.focal_positive_weight) * src_logits.shape[1]
        return loss_ce

    def loss_labels(self, outputs, targets, indices, num_boxes, cls_labels, mask_labels, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs

        # TODO: this is ugly and only for experiment purpose
        if self.use_hungarian_for_ce_loss:
            assert "hungarian_results" in dir(self.matcher),\
                "use_hungarian_for_ce_loss=True but self.matcher.hungarian_results is not defined"
            indices = self.matcher.hungarian_results
            num_boxes = max(sum([len(matched_gt_idx) for _, matched_gt_idx in indices]), 1)

        src_logits = outputs['pred_logits']

        target_classes_onehot, idx = self._get_target_classes_onehot(src_logits, targets, indices)

        # impute pseudo labels for positive cases where annotation is missing
        if self.use_pseudo_label:
            target_classes_onehot = SetCriterion.update_pseudo_label(cls_labels, target_classes_onehot, src_logits, k=5)

        focal_loss_mask = torch.ones_like(src_logits, device = src_logits.device)
        # compute a mask that zero out positive w/o annotations
        if self.mask_out_pos_no_anno_cases:
            focal_loss_mask = focal_loss_mask * model_util.positive_missing_annotation_mask(cls_labels, targets).unsqueeze(1)
        # compute a mask that zero out istp cases
        if self.mask_out_duplicate_tp:
            focal_loss_mask = focal_loss_mask * self._get_duplicate_indices_mask(targets, src_logits, self.matcher.duplicate_match)
            num_boxes = max(sum(len(t["labels"]) for t in targets), 1)
        
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha,
                                        mask=focal_loss_mask, pos_gamma = self.pos_focal_gamma, neg_gamma=self.neg_focal_gamma,
                                        pos_weight=self.focal_positive_weight) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        if log:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, cls_labels, mask_labels):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_reference_dist(self, outputs, targets, indices, num_boxes, cls_label, mask_labels, reference_points):
        _, n, _ = outputs['pred_logits'].shape
        batch_idx = torch.cat([(torch.ones(n-src.shape[0])*i).long() for i, (src, _) in enumerate(indices)])
        unmatched_src_idx = []
        for (src, _) in indices:
            full = torch.arange(n)
            combined = torch.cat((full,src))
            uniques, counts = combined.unique(return_counts=True)
            difference = uniques[counts == 1]
            unmatched_src_idx.append(difference)
        unmatched_src_idx = torch.cat(unmatched_src_idx)

        umatched_idx = batch_idx, unmatched_src_idx

        #idx = self._get_src_permutation_idx(self.matcher.duplicate_match) 
        #dist = F.l1_loss(outputs['pred_boxes'][idx][:,:2], reference_points[idx]) 
        
        #dist = torch.norm((outputs['pred_boxes'][:,:,:2]-reference_points),dim=-1)
        l1_dist = F.l1_loss(outputs['pred_boxes'][:,:,:2], reference_points) 
        
        dist = (5 * l1_dist) ** 4 #+ 100*(torch.exp(torch.max(dist - 0.15, torch.tensor(0.0,device=dist.device)) * 20) - 1)
        losses = {'loss_dist': dist.mean()}
        return losses

    
    def loss_boxes(self, outputs, targets, indices, num_boxes, cls_labels, mask_labels):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        # TODO: this is ugly and only for experiment purpose
        if self.use_hungarian_for_bbox_loss:
            assert "hungarian_results" in dir(self.matcher),\
                "use_hungarian_for_bbox_loss=True but self.matcher.hungarian_results is not defined"
            all_indices = indices
            indices = self.matcher.hungarian_results
            num_boxes = max(sum([len(matched_gt_idx) for _, matched_gt_idx in indices]), 1)

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # istp_indices = [(all_indices[i][0][len(t['labels']):],all_indices[i][1][len(t['labels']):]) for i , t in enumerate(targets)]
        # istp_idx = self._get_src_permutation_idx(istp_indices)
        # istp_src_boxes = outputs['pred_boxes'][istp_idx]
        # istp_target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, istp_indices)], dim=0)
        # # print(istp_target_boxes, 'istp target')
        # # print(istp_src_boxes, 'istp preds')
        # istp_target_boxes[:,:2] = (istp_src_boxes[:,:2] -istp_target_boxes[:,:2]) + istp_src_boxes[:,:2]
        # src_boxes = torch.cat([src_boxes,istp_src_boxes],dim=0)
        # target_boxes = torch.cat([target_boxes,istp_target_boxes],dim=0)
        # # print(istp_target_boxes)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
      
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, cls_labels, mask_labels):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def get_Hungarian_assigned_labels(self, outputs, targets, reference_points):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        if self.matcher.matcher_type in ['reference','istp']:
            indices = self.matcher(outputs_without_aux, targets, reference_points)
        else:
            indices = self.matcher(outputs_without_aux, targets)
        
        src_logits = outputs['pred_logits']
        hungarian_assigned_labels, _ = self._get_target_classes_onehot(src_logits, targets, indices)
        return hungarian_assigned_labels

    def _get_target_classes_onehot(self, src_logits, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        return target_classes_onehot, idx

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, cls_labels, mask_labels, **kwargs):
        assert loss in self.loss_map, f'do you really want to compute {loss} loss?'
        return self.loss_map[loss](outputs, targets, indices, num_boxes, cls_labels, mask_labels, **kwargs)

    def forward(self, outputs, targets, cls_labels, mask_labels, reference_points = None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        self.indices_dict = {'aux_indices':[]}
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        if self.matcher.matcher_type in ['reference','istp']:
            indices = self.matcher(outputs_without_aux, targets, reference_points)
        else:
            indices = self.matcher(outputs_without_aux, targets)
        self.indices_dict['pred_indices']=indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_boxes = sum(len(t["labels"]) for t in targets)
        # UPDATE: num_boxes reflects num of bboxes that are matched to a gt_box by the matcher.
        num_boxes = sum([len(matched_gt_idx) for _, matched_gt_idx in indices])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            if loss == 'reference':
                kwargs['reference_points'] = reference_points
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, cls_labels, mask_labels, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if self.matcher.matcher_type in ['reference','istp']:
                    indices = self.matcher(aux_outputs, targets, reference_points)
                else:
                    indices = self.matcher(aux_outputs, targets)
                self.indices_dict['aux_indices'].append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels': 
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    if loss == 'reference':
                        kwargs['reference_points'] = reference_points
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, cls_labels, mask_labels, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, cls_labels, mask_labels, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class JointModelDETR(DeformableDETR):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False,
                 topk_candidates_per_class=10, detach_cls=False, shared_cls_pred=False):
        super().__init__(backbone, transformer, num_classes, num_queries, num_feature_levels, aux_loss,
                         with_box_refine, two_stage)
        self.topk_candidates_per_class = topk_candidates_per_class
        self.aggregator = model_util.AttentionFeatureAggregator(
            feature_dim=transformer.d_model, hidden_dim=256, mode="ordered")
        if shared_cls_pred:
            self.img_cls_layer = nn.Linear(in_features=transformer.d_model, out_features=num_classes, bias=False)
        else:
            self.img_cls_layer = self.class_embed[-1]
        self.detach_cls = detach_cls

    def select_topk_vectors(self, last_hs, last_outputs_class):
        """
        Function that selects k post-fusion query vectors for refinement.
        :param last_hs: output of the last transformer layer, should be a Tensor of [batch_size, n_query, query_dim]
        :param last_outputs_class: predictions of last transformer layer, should be a Tensor of [batch_size, n_query, n_class]
        :return: a tensor of size [batch_size, self.topk_candidates_per_class*n_class, query_dim]
        """
        batch_size = last_hs.size()[0]
        _, topk_idx = last_outputs_class.topk(self.topk_candidates_per_class, dim=1)
        flatten_topk_idx = torch.flatten(topk_idx, 1, 2)
        out_list = [last_hs[i, flatten_topk_idx[i, :], :].unsqueeze(0) for i in range(batch_size)]
        out = torch.cat(out_list, dim=0)
        return out

    def compute_heads(self, hs, init_reference, inter_references, memory):
        outputs_class, outputs_coord = super().compute_heads(hs, init_reference, inter_references, memory)
        # Step 1: propose candidate vectors to represent the images
        # Use the last layer representations only: deformable DETR only uses the last layer vec for bbox classification.
        # TODO: consider also doing this for other layers
        last_outputs_class = outputs_class[-1, :, :, :]
        last_hs = hs[-1, :, :, :]
        # For each class, select the topk post-fusion query vecs according to their bbox predictions.
        feature_vectors = self.select_topk_vectors(last_hs, last_outputs_class)
        if self.detach_cls:
            feature_vectors = feature_vectors.detach()
        # Step 2: aggregate the candidate vectors to form image feature representations
        _, self.img_feature_vectors = self.aggregator(feature_vectors)
        # Step 3: use the image representations to generate predictions
        self.cls_pred = torch.sigmoid(self.img_cls_layer(self.img_feature_vectors))
        return outputs_class, outputs_coord

    def prepare_outputs(self, outputs_class, outputs_coord, enc_outputs_coord_unact, enc_outputs_class):
        out = super().prepare_outputs(outputs_class, outputs_coord, enc_outputs_coord_unact, enc_outputs_class)
        out["cls_pred"] = self.cls_pred
        return out

class JointLoss(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, pos_focal_gamma=2, neg_focal_gamma=2):
        super().__init__(num_classes, matcher, weight_dict, losses, focal_alpha, pos_focal_gamma, neg_focal_gamma)
        self.loss_map["classification"] = self.loss_cls
        self.loss_map["dice"] = self.loss_dice
        self.losses.append("classification")
        self.losses.append("dice")

    def loss_cls(self, outputs, targets, indices, num_boxes, cls_labels, mask_labels):
        # in two stage mode, cls_pred will NOT be in the encoder output
        if "cls_pred" not in outputs:
            loss_bce = 0
        else:
            cls_pred = outputs['cls_pred']
            loss_bce = F.binary_cross_entropy(cls_pred, cls_labels)
        losses = {'loss_cls': loss_bce}
        return losses

    def loss_dice(self, outputs, targets, indices, num_boxes, cls_labels, mask_labels):
        if "cam" not in outputs:
            loss_dice = 0
        else:
            loss_dice = model_util.loss_dice(outputs["cam"], cls_labels, mask_labels, "mask")
        losses = {'loss_cam_dice': loss_dice}
        return losses


def build(args, detr_class=DeformableDETR, loss_class=SetCriterion):
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    model = detr_class(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if "cancer_cls_loss_coef" in dir(args):
        weight_dict["loss_cls"] = args.cancer_cls_loss_coef
    if "cam_dice_loss_coef" in dir(args):
        weight_dict["loss_cam_dice"] = args.cam_dice_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    if args.ref_dist:
        weight_dict['loss_dist'] = args.ref_dist_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    if args.ref_dist:
        losses += ["reference"]

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = loss_class(args.num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,
                        pos_focal_gamma=args.pos_focal_gamma, neg_focal_gamma=args.neg_focal_gamma)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors


def build_deformable_detr(lr_backbone=1e-6, dim_feedforward=1024,hidden_dim=256,
                          position_embedding="sine", num_queries=170, deformable_sampling_pts=4,
                          detr_class=DeformableDETR, loss_class=SetCriterion,
                          with_box_refine=False, two_stage=False, backbone="swin-t", 
                          num_feature_levels=1, enc_layers=3, dec_layers=6,
                          cancer_cls_loss_coef=1, focal_alpha=0.2, pos_focal_gamma=2, neg_focal_gamma=2,
                          matcher_class="hungarian", min_match_dist=0.2, ref_dist=False,ref_dist_coef = 5,
                          ce_loss_coef=0.5, giou_loss_coef=0.5, mixed_query_selection=False, use_nms=False):
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    # do the settings
    args.masks = False
    args.num_classes = 2
    args.num_queries = num_queries
    args.num_feature_levels = num_feature_levels
    args.aux_loss = True
    args.with_box_refine = with_box_refine
    args.two_stage = two_stage
    args.mixed_query_selection = mixed_query_selection
    args.use_nms = use_nms
    # matcher
    args.min_match_dist = min_match_dist
    # losses
    args.ref_dist = ref_dist
    args.ref_dist_coef = ref_dist_coef
    args.ce_loss_coef = ce_loss_coef
    args.bbox_loss_coef = 1
    args.giou_loss_coef = giou_loss_coef
    args.cancer_cls_loss_coef = cancer_cls_loss_coef
    #args.cam_dice_loss_coef =2
    args.focal_alpha = focal_alpha
    args.pos_focal_gamma = pos_focal_gamma
    args.neg_focal_gamma = neg_focal_gamma
    args.eos_coef = 0.1
    # backbones
    args.backbone = backbone
    args.lr_backbone = lr_backbone
    args.dim_feedforward = dim_feedforward  # this is the dimension within transformer
    args.dilation = False
    args.position_embedding = position_embedding  # sine or learned
    # transformer
    args.pre_norm = False
    args.hidden_dim = hidden_dim  # this is the dimension of the transformer
    args.nheads = 8
    args.enc_layers = enc_layers
    args.dec_layers = dec_layers
    args.dropout = 0.1
    args.dec_n_points = deformable_sampling_pts # number of sampling points for decoder
    args.enc_n_points = deformable_sampling_pts
    # matcher
    args.matcher_class = matcher_class
    args.set_cost_class = ce_loss_coef
    args.set_cost_bbox = 1
    args.set_cost_giou = giou_loss_coef
    args.matcher_alpha = focal_alpha
    args.matcher_gamma = pos_focal_gamma
    # deformable specific
    args.device = "cuda"
    return build(args, detr_class, loss_class)


def build_joint_deformable_detr():
    return build_deformable_detr(detr_class=JointModelDETR, loss_class=JointLoss)
