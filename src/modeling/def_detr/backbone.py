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
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from src.modeling.swin_transformer.build import build_model as build_swin
from src.modeling.def_detr.util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
import argparse

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone, train_backbone, return_interm_layers, out_indices=[3],num_channels=[1024]):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone: #or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}

            strides = [8, 16, 32]
            return_layers = {}
            self.strides = []
            self.num_channels = []
            for i, indices in enumerate(out_indices):
                return_layers[f'layer{indices}'] = str(i)
                self.strides.append(strides[indices-2])
                self.num_channels.append(num_channels[indices-2])
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = num_channels[-1]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name, train_backbone, return_interm_layers, dilation, out_indices=[3],freeze_norm=True):
        norm_layer = FrozenBatchNorm2d
        assert name in ("resnet18", "resnet34", "resnet50")

        # determine the number of channels
        if name in ("resnet18", "resnet34"):
            num_channels = [128, 256, 512]
        else:
            num_channels = [512, 1024, 2048]

        if freeze_norm:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=norm_layer)
        else:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process())
        #assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers, out_indices, num_channels)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

class SwinBackbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name, train_backbone, return_interm_layers,
                 dilation, pretrained = False, out_indices=[3]):
        super().__init__()

        strides=[8, 16, 32]
        num_channels=[192, 384, 768]
        self.strides = [strides[idx-1] for idx in out_indices]
        self.num_channels = [num_channels[idx-1] for idx in out_indices]
        print('!!!',out_indices)
      
        self.body = build_swin(name, out_indices=out_indices, pretrained=pretrained, frozen_stages=-1)
        if not train_backbone:
            for key, parameter in self.body.named_parameters():
                parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    if "swin" in args.backbone:
        backbone = SwinBackbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    else:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    return model

def resolve_backbone(backbone, lr_backbone, num_class):
    """
    Entry-level function for retrieve a DETR-style backbone
    Called by pre-training functions
    :param backbone:
    :param lr_backbone:
    :param num_class:
    :return:
    """
    # common set up for positional embeddings
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.hidden_dim = 256 # this is the dimension for transformer
    args.dilation = False
    args.position_embedding = "sine"
    args.num_classes = num_class
    args.masks = False
    args.num_feature_levels = 1
    args.lr_backbone = lr_backbone
    args.backbone = backbone
    

    backbone = build_backbone(args)
    return backbone