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
# --------------------------------------------------------
# Modified from Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .config import get_config
import torch

CHECKPOINT_MAP = {
    "swin-t": 'swin/swin_tiny_patch4_window7_224.pth',
    "swin-s": 'swin/swin_small_patch4_window7_224.pth',
    "swin-b": 'swin/swin_base_patch4_window7_224.pth',
}

CONFIG_MAP = {
    "swin-t": 'src/modeling/swin_transformer/configs/swin_tiny_patch4_window7_224.yaml',
    "swin-s": "src/modeling/swin_transformer/configs/swin_small_patch4_window7_224.yaml",
    "swin-b": "src/modeling/swin_transformer/configs/swin_base_patch4_window7_224.yaml",
}

def build_model(name, out_indices, pretrained, frozen_stages=-1):
    config = get_config(CONFIG_MAP[name])
    model_type = config.MODEL.TYPE

    if model_type == 'swin':
        model = SwinTransformer(pretrain_img_size=config.MODEL.SWIN.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.MODEL.SWIN.USE_CHECKPOINT,
                                out_indices=out_indices,
                                frozen_stages=frozen_stages)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    if pretrained:
        load_pretrained(name, model, config.MODEL.SWIN.WINDOW_SIZE)
        
    return model


def load_pretrained(name, model, window_size):
    
    checkpoint = torch.load(CHECKPOINT_MAP[name], map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = (window_size[0]*2-1, window_size[1]*2-1)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size= S2,
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = (window_size[0]*2-1, window_size[1]*2-1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=S2, mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    msg = model.load_state_dict(state_dict, strict=False)
    #logger.warning(msg)

    print(f"=> loaded swin backbone successfully '{CHECKPOINT_MAP[name]}'")

    del checkpoint
    torch.cuda.empty_cache()
