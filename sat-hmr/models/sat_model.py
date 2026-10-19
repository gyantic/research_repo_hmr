# Modified from DAB-DETR (https://github.com/IDEA-Research/DAB-DETR)

import os

import math
from math import tan,pi
from typing import Dict
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import Resize
import numpy as np
import time
import random

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from utils.transforms import rot6d_to_axis_angle, img2patch_flat, img2patch, to_zorder
from utils.map import build_z_map
from utils import constants
from configs.paths import smpl_mean_path

from models.encoders import build_encoder
from .matcher import build_matcher
from .decoder import build_decoder
from .position_encoding import position_encoding_xy
from .criterion import SetCriterion
from .dn_components import prepare_for_cdn, dn_post_process
import copy

from configs.paths import smpl_model_path
from models.human_models import SMPL_Layer


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@torch._dynamo.disable
def _drop_register_tokens_concat(x_list, num_register_tokens):
    return torch.cat([x[:, 1 + num_register_tokens:, :].squeeze(0) for x in x_list], dim=0)


@torch._dynamo.disable
def _normalize_pos_per_lvl(pos_lvls, dtype, feature_size):
    return [(pos.to(dtype=dtype) + 0.5) / feature_size[i] for i, pos in enumerate(pos_lvls)]


@torch._dynamo.disable
def _build_x_list_multi(cr_token_list, token_lvls, token_lens_lvls):
    splits = [tokens.split(lens) for (tokens, lens) in zip(token_lvls, token_lens_lvls)]
    return [torch.cat([cr, *lvls]).unsqueeze(0)
            for cr, *lvls in zip(cr_token_list, *splits)]


@torch._dynamo.disable
def _build_pos_list_multi(pos_lvls, token_lens_lvls):
    splits = [pos.split(lens) for (pos, lens) in zip(pos_lvls, token_lens_lvls)]
    return [torch.cat([*lvls]) for lvls in zip(*splits)]


@torch._dynamo.disable
def _build_lvl_id_list_multi(pos_lvls, token_lens_lvls):
    splits = [pos.split(lens) for (pos, lens) in zip(pos_lvls, token_lens_lvls)]
    return [torch.cat([torch.full_like(lvl, i, dtype=torch.int64) for i, lvl in enumerate(lvls)])
            for lvls in zip(*splits)]


@torch._dynamo.disable
def _lvl_pool4(tokens):
    assert len(tokens) % 4 == 0
    return torch.max(tokens.unflatten(0, (-1, 4)), dim=1)[0]


class Model(nn.Module):
    """ One-stage Multi-person Human Mesh Estimation via Scale-adaptive Tokens """
    def __init__(self, encoder, decoder,
                    num_queries,
                    input_size,
                    sat_cfg = {'use_sat': False},
                    dn_cfg = {'use_dn': False},
                    train_pos_embed = True,
                    aux_loss=True, 
                    iter_update=True,
                    query_dim=4, 
                    bbox_embed_diff_each_layer=True,
                    random_refpoints_xy=False,
                    num_poses=24,
                    dim_shape=10,
                    FOV=pi/3
                    ):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder to be used. See ./encoders.
            decoder: torch module of the decoder architecture. See decoder.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            iter_update: iterative update of boxes
            query_dim: query dimension. 2 for point and 4 for box.
            bbox_embed_diff_each_layer: dont share weights of prediction heads. Default for False. (shared weights.)
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
        """
        super().__init__()

        # ========== Start of common settings =============
        self.input_size = input_size
        hidden_dim = decoder.d_model
        num_dec_layers = decoder.dec_layers
        self.hidden_dim = hidden_dim
        # camera model
        self.focal = input_size/(2*tan(FOV/2))
        self.FOV = FOV
        cam_intrinsics = torch.tensor([[self.focal,0.,self.input_size/2],
                                            [0.,self.focal,self.input_size/2],
                                            [0.,0.,1.]])
        self.register_buffer('cam_intrinsics', cam_intrinsics)
        # human model
        self.num_poses = num_poses
        self.dim_shape = dim_shape
        self.human_model = SMPL_Layer(model_path = smpl_model_path, with_genders = False)
        # init params (following multi-hmr)
        smpl_mean_params = np.load(smpl_mean_path, allow_pickle = True)
        self.register_buffer('mean_pose', torch.from_numpy(smpl_mean_params['pose']))
        self.register_buffer('mean_shape', torch.from_numpy(smpl_mean_params['shape']))
        # ========== End of common settings =============


        # ========== Start of SAT-encoder settings =============
        self.encoder = encoder
        
        self.patch_size = encoder.patch_size
        assert self.patch_size == 14
        
        self.use_sat = sat_cfg['use_sat']
        self.sat_cfg = sat_cfg

        if self.use_sat:
            assert sat_cfg['num_lvls'] >= 2
            assert self.input_size % (self.patch_size<<2) == 0

            self.feature_size = []
            for lvl in range(sat_cfg['num_lvls']):
                patch_size = self.patch_size<<lvl
                self.feature_size.append(self.input_size / patch_size)

            # build z_order curve
            z_depth = math.ceil(math.log2(self.feature_size[1]))
            z_map, ys, xs = build_z_map(z_depth)
            self.register_buffer('z_order_map', z_map)
            self.register_buffer('y_coords', ys)
            self.register_buffer('x_coords', xs)

            self.enc_inter_norm = copy.deepcopy(encoder.norm)
            self.scale_head = MLP(encoder.embed_dim, encoder.embed_dim, 2, 4)
            self.encoder_patch_proj = _get_clones(encoder.patch_embed.proj, 2)
            self.encoder_patch_norm = _get_clones(encoder.patch_embed.norm, 2)

            if sat_cfg['lvl_embed']:
                # same as level_embed in Deformable-DETR
                self.level_embed = nn.Parameter(torch.Tensor(sat_cfg['num_lvls'],hidden_dim))
                nn.init.normal_(self.level_embed)
        else:
            assert self.input_size % self.patch_size == 0
            self.feature_size = [self.input_size // self.patch_size]
            self.encoder_patch_proj = copy.deepcopy(encoder.patch_embed.proj)
            self.encoder_patch_norm = copy.deepcopy(encoder.patch_embed.norm)

        # cls_token and register tokens
        encoder_cr_token = self.encoder.cls_token.view(1,-1) + self.encoder.pos_embed.float()[:,0].view(1,-1)
        if self.encoder.register_tokens is not None:
            encoder_cr_token = torch.cat([encoder_cr_token, self.encoder.register_tokens.view(self.encoder.num_register_tokens,-1)], dim=0)
        self.encoder_cr_token = nn.Parameter(encoder_cr_token)
        
        self.encoder_pos_embeds = nn.Parameter(self.encoder.interpolate_pos_encoding3(self.feature_size[0]).detach())
        if not train_pos_embed:
            self.encoder_pos_embeds.requires_grad = False
        
        self.preprocessed_pos_lvl1 = None
        
        # delete unwanted params
        del(self.encoder.mask_token)
        del(self.encoder.pos_embed)
        del(self.encoder.patch_embed)
        del(self.encoder.cls_token)
        del(self.encoder.register_tokens)
        # ========== End of SAT-encoder settings =============


        
        # ========== Start of decoder settings =============
        self.num_queries = num_queries
        self.decoder = decoder
        
        # embed_dim between encoder and decoder can be different
        self.feature_proj = nn.Linear(encoder.embed_dim, hidden_dim)

        # bbox
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for i in range(num_dec_layers)])
        else:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # poses (use 6D rotation)
        self.pose_head = MLP(hidden_dim, hidden_dim, num_poses*6, 6)
        # shape
        self.shape_head = MLP(hidden_dim, hidden_dim, dim_shape, 5)
        # cam_trans
        self.cam_head = MLP(hidden_dim, hidden_dim//2, 3, 3)
        # confidence score
        self.conf_head = nn.Linear(hidden_dim, 1)
        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.conf_head.bias.data = torch.ones(1) * bias_value

        # for iter update
        self.pose_head = _get_clones(self.pose_head, num_dec_layers)
        self.shape_head = _get_clones(self.shape_head, num_dec_layers)
        
        # setting query dim (bboxes as queries)
        self.query_dim = query_dim
        assert query_dim == 4
        self.refpoint_embed = nn.Embedding(num_queries, query_dim)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        self.random_refpoints_xy = random_refpoints_xy
        if random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        self.aux_loss = aux_loss
        self.iter_update = iter_update
        assert iter_update
        if self.iter_update:
            self.decoder.decoder.bbox_embed = self.bbox_embed

        assert bbox_embed_diff_each_layer
        if bbox_embed_diff_each_layer:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        # ========== End of decoder settings =============


        # for dn training
        self.use_dn = dn_cfg['use_dn']
        self.dn_cfg = dn_cfg
        if self.use_dn:
            assert dn_cfg['dn_number'] > 0
            if dn_cfg['tgt_embed_type'] == 'labels':
                self.dn_enc = nn.Embedding(dn_cfg['dn_labelbook_size'], hidden_dim)
            elif dn_cfg['tgt_embed_type'] == 'params':
                self.dn_enc = nn.Linear(num_poses*3 + dim_shape, hidden_dim)
            else:
                raise NotImplementedError


    def lvl_pooling(self, tokens):
        return _lvl_pool4(tokens)

    def forward_bbox_embed(self, hs):
        # hs: (num_decoder_layers, bs, num_queries, dim); one bbox MLP per decoder layer.
        return torch.stack([self.bbox_embed[lvl](hs[lvl]) for lvl in range(hs.shape[0])], dim=0)
                
    def get_scale_map(self, x_list):
        if self.sat_cfg['use_additional_blocks']:
            x_list = self.encoder.forward_additional_layers_list(x_list, end=self.sat_cfg['get_map_layer'], get_feature=False)
        else:
            x_list = self.encoder.forward_specific_layers_list(x_list, end=self.sat_cfg['get_map_layer'], get_feature=False)
        
        cr_token_list = [x[:, :1 + self.encoder.num_register_tokens, :].squeeze(0) for x in x_list]
        x_tokens = torch.cat([x[:, 1 + self.encoder.num_register_tokens:, :].squeeze(0) for x in x_list], dim=0)
        scale_map = self.scale_head(self.enc_inter_norm(x_tokens)).sigmoid()
        return scale_map, cr_token_list, x_tokens

    def pad_mask(self, mask):
        mask4 = mask.reshape(-1, 4)
        row_mask = torch.any(mask4, dim=1, keepdim=True)
        return torch.logical_or(mask4, row_mask).flatten()

    @torch._dynamo.disable
    def prepare_static_eval_buffers(self):
        if self.training or not self.use_sat:
            return
        if self.preprocessed_pos_lvl1 is None:
            self.preprocessed_pos_lvl1 = (
                F.interpolate(
                    self.encoder_pos_embeds.unsqueeze(0).permute(0, 3, 1, 2),
                    mode="bicubic",
                    antialias=self.encoder.interpolate_antialias,
                    size=(int(self.feature_size[1]), int(self.feature_size[1])),
                )
                .squeeze(0)
                .permute(1, 2, 0)
                .contiguous()
                .clone()
            )

    def forward_encoder(self, samples, targets, use_gt = False):
        B = len(samples)
        C = self.encoder.embed_dim
        cr_token_list = [self.encoder_cr_token]*len(samples)
        profile_count = int(getattr(self, '_sat_profile_forward_count', 0))
        profile_limit = int(getattr(self, '_sat_profile_forward_limit', 0))
        profile_encoder = (
            bool(getattr(self, '_sat_profile_forward', False))
            and not self.training
            and (profile_limit <= 0 or profile_count < profile_limit)
        )
        enc_marks = []
        enc_device = samples[0].device if isinstance(samples, list) and samples else None
        profile_encoder_blocks = profile_encoder and bool(getattr(self, '_sat_profile_encoder_blocks', False))
        prev_encoder_block_profile = getattr(self.encoder, '_sat_profile_blocks', False)
        prev_encoder_block_forward = getattr(self.encoder, '_sat_profile_forward_index', 0)
        prev_encoder_block_segment = getattr(self.encoder, '_sat_profile_segment', '')
        if profile_encoder_blocks:
            self.encoder._sat_profile_blocks = True
            self.encoder._sat_profile_forward_index = profile_count
        if profile_encoder:
            def _enc_mark(name):
                if enc_device is not None and enc_device.type == 'cuda':
                    torch.cuda.synchronize(enc_device)
                #enc_marks.append((name, time.perf_counter()))
            _enc_mark('start')
        else:
            def _enc_mark(name):
                return None
        enc_profile_extra = {}

        if not self.use_sat:
            # img2token
            lvl0_feature_hw = [(img.shape[1]//self.patch_size, img.shape[2]//self.patch_size) for img in samples]
            lvl0_token_lens = [h*w for (h,w) in lvl0_feature_hw]
            lvl0_img_patches = torch.cat([img2patch_flat(img, patch_size = self.patch_size)\
                                        for img in samples], dim=0)
            lvl0_tokens = self.encoder_patch_norm(self.encoder_patch_proj(lvl0_img_patches).flatten(1))     

            # token position information
            full_grids = torch.meshgrid(torch.arange(self.feature_size[0]), torch.arange(self.feature_size[0]), indexing='ij')
            lvl0_pos_y = torch.cat([full_grids[0][:h,:w].flatten() for (h,w) in lvl0_feature_hw]).to(device = lvl0_tokens.device)
            lvl0_pos_x = torch.cat([full_grids[1][:h,:w].flatten() for (h,w) in lvl0_feature_hw]).to(device = lvl0_tokens.device)

            # pos_embed
            full_pos_embed = self.encoder_pos_embeds
            lvl0_pos_embed = torch.cat([full_pos_embed[:h,:w].flatten(0,1)\
                                        for (h,w) in lvl0_feature_hw], dim=0)
            lvl0_tokens = lvl0_tokens + lvl0_pos_embed

            # convert to list for DINOv2 input
            x_list = [torch.cat([cr, lvl0],dim=0).unsqueeze(0)\
                                for (cr, lvl0) \
                                in zip(cr_token_list, lvl0_tokens.split(lvl0_token_lens))]
            
            
            lvl0_pos_y_norm = (lvl0_pos_y.to(dtype=lvl0_tokens.dtype) + 0.5) / self.feature_size[0]
            lvl0_pos_x_norm = (lvl0_pos_x.to(dtype=lvl0_tokens.dtype) + 0.5) / self.feature_size[0]
            pos_y_list = list(lvl0_pos_y_norm.split(lvl0_token_lens))
            pos_x_list = list(lvl0_pos_x_norm.split(lvl0_token_lens))
            scale_map_dict = None
            # also create lvl_list for patch visualization
            lvl_list = [torch.zeros_like(pos,dtype=int) for pos in pos_x_list]
            _enc_mark('patch_tokens')

        else:
            lvl1_feature_hw = [(img.shape[1]//(2*self.patch_size), img.shape[2]//(2*self.patch_size)) for img in samples]
            lvl1_token_lens = [h*w for (h,w) in lvl1_feature_hw]

            lvl1_img_patches_28, lvl1_zorders = [], []
            lvl1_pos_y, lvl1_pos_x = [], []
            lvl1_bids = []

            for i, img in enumerate(samples):
                z_patches, z_order, pos_y, pos_x = to_zorder(img2patch(img, patch_size = 2*self.patch_size), 
                                                             z_order_map = self.z_order_map,
                                                             y_coords = self.y_coords,
                                                             x_coords = self.x_coords)

                lvl1_img_patches_28.append(z_patches)
                
                lvl1_zorders.append(z_order)
                lvl1_pos_y.append(pos_y)
                lvl1_pos_x.append(pos_x)
                lvl1_bids.append(torch.full_like(pos_y, i, dtype=torch.int64))
            

            
            lvl1_img_patches_28 = torch.cat(lvl1_img_patches_28, dim=0)
            lvl1_zorders = torch.cat(lvl1_zorders, dim=0)
            lvl1_pos_y = torch.cat(lvl1_pos_y, dim=0)
            lvl1_pos_x = torch.cat(lvl1_pos_x, dim=0)
            lvl1_bids = torch.cat(lvl1_bids, dim=0)

            

            # (L1, 3, 28, 28)
            assert len(lvl1_img_patches_28) == sum(lvl1_token_lens)
            lvl1_img_patches = F.interpolate(lvl1_img_patches_28, size = (14,14), mode='bilinear', align_corners=False)
            # (L1, 3, 14, 14)
            lvl1_tokens = self.encoder_patch_norm[1](self.encoder_patch_proj[1](lvl1_img_patches).flatten(1))
            # (L1, C)
            
            
            
            assert len(lvl1_pos_y) == len(lvl1_tokens)
            full_pos_embed = self.preprocessed_pos_lvl1 if not self.training\
                                else F.interpolate(self.encoder_pos_embeds.unsqueeze(0).permute(0, 3, 1, 2),
                                            mode="bicubic",
                                            antialias=self.encoder.interpolate_antialias,
                                            size = (int(self.feature_size[1]),int(self.feature_size[1]))).squeeze(0).permute(1,2,0)
            lvl1_pos_embed = torch.cat([full_pos_embed[ys,xs]\
                                        for (ys,xs) in zip(lvl1_pos_y.split(lvl1_token_lens), lvl1_pos_x.split(lvl1_token_lens))], dim=0)
            lvl1_tokens = lvl1_tokens + lvl1_pos_embed
            _enc_mark('lvl1_patch')

            # get scale map (flattened)
            x_list = [torch.cat([cr, lvl1],dim=0).unsqueeze(0)\
                                 for (cr, lvl1) \
                                 in zip(cr_token_list, lvl1_tokens.split(lvl1_token_lens))]
            if profile_encoder_blocks:
                self.encoder._sat_profile_segment = 'scale_map'
            scale_map, updated_cr_list, updated_lvl1 = self.get_scale_map(x_list)
            _enc_mark('scale_map')
            # for visualization
            scale_map_dict = {'scale_map': scale_map, 'lens': lvl1_token_lens, 'hw': lvl1_feature_hw,
                              'pos_y': lvl1_pos_y, 'pos_x': lvl1_pos_x}
            
            conf_thresh = self.sat_cfg['conf_thresh']
            scale_thresh = self.sat_cfg['scale_thresh']
            if use_gt:
                scale_map = torch.cat([tgt['scale_map'].view(-1,2) for tgt in targets], dim=0)

            # Export-safe SAT path:
            # keep SAT enabled, but replace hard boolean token selection with soft gating to
            # avoid NonZero-driven dynamic shape ops that block strict INT8 build.
            export_soft_sat = torch.onnx.is_in_onnx_export() and self.sat_cfg['num_lvls'] == 2

            if export_soft_sat:
                # Prepare lvl0 tokens from all lvl1 patches (dense), then gate by SAT confidence/scale.
                lvl0_token_lens = [int(lvl_len) << 2 for lvl_len in lvl1_token_lens]
                lvl0_tokens = self.encoder_patch_norm[0](
                    self.encoder_patch_proj[0](lvl1_img_patches_28).permute(0, 2, 3, 1).flatten(0, 2)
                )

                # Use arithmetic multiply for ONNX-export stability (avoids Pow->Cast int64 chain from bit-shift).
                lvl0_pos_y = (lvl1_pos_y * 2)[:, None].repeat(1, 4).flatten()
                lvl0_pos_x = (lvl1_pos_x * 2)[:, None].repeat(1, 4).flatten()
                lvl0_pos_y[2::4] += 1
                lvl0_pos_y[3::4] += 1
                lvl0_pos_x[1::2] += 1

                full_pos_embed = self.encoder_pos_embeds
                lvl0_pos_embed = torch.cat(
                    [full_pos_embed[ys, xs] for (ys, xs) in zip(lvl0_pos_y.split(lvl0_token_lens), lvl0_pos_x.split(lvl0_token_lens))],
                    dim=0,
                )
                lvl0_tokens = lvl0_tokens + lvl0_pos_embed

                x_list = [
                    torch.cat([cr, lvl0], dim=0).unsqueeze(0)
                    for (cr, lvl0) in zip(cr_token_list, lvl0_tokens.split(lvl0_token_lens))
                ]
                x_list = self.encoder.forward_specific_layers_list(
                    x_list, end=self.sat_cfg['get_map_layer'], get_feature=False
                )
                lvl0_tokens = torch.cat(
                    [x[:, 1 + self.encoder.num_register_tokens :, :].squeeze(0) for x in x_list], dim=0
                )

                lvl1_tokens = updated_lvl1
                cr_token_list = updated_cr_list

                sat_conf = torch.sigmoid((scale_map[:, 0] - conf_thresh) * 8.0).to(dtype=lvl1_tokens.dtype)
                sat_scale = torch.sigmoid((scale_thresh - scale_map[:, 1]) * 8.0).to(dtype=lvl1_tokens.dtype)
                sat_weight = sat_conf * sat_scale

                lvl0_weight = sat_weight.repeat_interleave(4).to(dtype=lvl0_tokens.dtype)
                lvl1_weight = (1.0 - sat_weight).to(dtype=lvl1_tokens.dtype)
                lvl0_tokens = lvl0_tokens * lvl0_weight[:, None]
                lvl1_tokens = lvl1_tokens * lvl1_weight[:, None]

                lvl0_pos_y_norm = (lvl0_pos_y.to(dtype=lvl0_tokens.dtype) + 0.5) / self.feature_size[0]
                lvl0_pos_x_norm = (lvl0_pos_x.to(dtype=lvl0_tokens.dtype) + 0.5) / self.feature_size[0]
                lvl1_pos_y_norm = (lvl1_pos_y.to(dtype=lvl1_tokens.dtype) + 0.5) / self.feature_size[1]
                lvl1_pos_x_norm = (lvl1_pos_x.to(dtype=lvl1_tokens.dtype) + 0.5) / self.feature_size[1]

                x_list = [
                    torch.cat([cr, lvl0, lvl1]).unsqueeze(0)
                    for cr, lvl0, lvl1 in zip(
                        cr_token_list, lvl0_tokens.split(lvl0_token_lens), lvl1_tokens.split(lvl1_token_lens)
                    )
                ]
                pos_y_list = [
                    torch.cat([lvl0, lvl1])
                    for lvl0, lvl1 in zip(
                        lvl0_pos_y_norm.split(lvl0_token_lens), lvl1_pos_y_norm.split(lvl1_token_lens)
                    )
                ]
                pos_x_list = [
                    torch.cat([lvl0, lvl1])
                    for lvl0, lvl1 in zip(
                        lvl0_pos_x_norm.split(lvl0_token_lens), lvl1_pos_x_norm.split(lvl1_token_lens)
                    )
                ]
                lvl_list = [
                    torch.cat([torch.zeros_like(lvl0, dtype=int), torch.ones_like(lvl1, dtype=int)])
                    for lvl0, lvl1 in zip(
                        lvl0_pos_x_norm.split(lvl0_token_lens), lvl1_pos_x_norm.split(lvl1_token_lens)
                    )
                ]

            else:
                lvl1_valid_mask = scale_map[:,0] > conf_thresh
                lvl1_sat_mask = lvl1_valid_mask & (scale_map[:,1] < scale_thresh)
                if profile_encoder:
                    enc_profile_extra['lvl1_total'] = int(lvl1_sat_mask.numel())
                    enc_profile_extra['lvl1_valid'] = int(lvl1_valid_mask.sum().item())
                    enc_profile_extra['lvl1_sat'] = int(lvl1_sat_mask.sum().item())
            
                # prepare sat tokens (lvl0)
                lvl0_token_lens = [msk.sum().item()<<2 for msk in lvl1_sat_mask.split(lvl1_token_lens)]
                if profile_encoder:
                    enc_profile_extra['lvl0_tokens'] = int(sum(lvl0_token_lens))
                sat_idx = lvl1_sat_mask.nonzero(as_tuple=True)[0]
                sat_n = sat_idx.size(0)
                torch._check_is_size(sat_n)
                # dummy sentinel を sat_idx に直接 1 個 padding することで、index_select 後の
                # サイズが必ず sat_n + 1 (>= 1) になる。reshape 前で dummy を落とさないことで
                # M = sat_n + 1 が shape 計算に入り、sat_n=0 でも M=1 で安全。
                sat_idx_pad = torch.cat([sat_idx, sat_idx.new_zeros(1)], dim=0)
                lvl1_sat_patches_28 = lvl1_img_patches_28.index_select(0, sat_idx_pad)  # (sat_n+1, 3, 28, 28)
                # encoder_patch_proj[0] は patch_size=14 で 28x28 入力に対し (N, 768, 2, 2) を返す。
                # 4 つの空間位置を整数 indexing で取り出して stack することで「空間 4 個」を
                # Dynamo に hard-code で伝える (s7/s8 dynamic shape 経路を回避)。
                _proj_out = self.encoder_patch_proj[0](lvl1_sat_patches_28)  # (sat_n+1, 768, 2, 2)
                M = _proj_out.shape[0]
                torch._check_is_size(M)
                lvl0_tokens = torch.stack(
                    [
                        _proj_out[:, :, 0, 0],
                        _proj_out[:, :, 0, 1],
                        _proj_out[:, :, 1, 0],
                        _proj_out[:, :, 1, 1],
                    ],
                    dim=1,
                )  # (sat_n+1, 4, 768)
                lvl0_tokens = self.encoder_patch_norm[0](lvl0_tokens)
                lvl0_tokens = lvl0_tokens.reshape(M * 4, lvl0_tokens.shape[-1])
                lvl0_tokens = lvl0_tokens[:-4]

                assert len(lvl0_tokens) == sum(lvl0_token_lens)
                # lvl0 positions
                lvl0_pos_y = lvl1_pos_y.index_select(0, sat_idx)
                lvl0_pos_x = lvl1_pos_x.index_select(0, sat_idx)
                lvl0_pos_y = (lvl0_pos_y<<1)[:,None].repeat(1,4).flatten()
                lvl0_pos_x = (lvl0_pos_x<<1)[:,None].repeat(1,4).flatten()
                lvl0_pos_y[2::4] += 1
                lvl0_pos_y[3::4] += 1
                lvl0_pos_x[1::2] += 1
                assert len(lvl0_pos_x) == len(lvl0_tokens)

                # lvl0 pos_embed
                full_pos_embed = self.encoder_pos_embeds
                lvl0_pos_embed = torch.cat([full_pos_embed[ys,xs]\
                                            for (ys,xs) in zip(lvl0_pos_y.split(lvl0_token_lens), lvl0_pos_x.split(lvl0_token_lens))], dim=0)
                lvl0_tokens = lvl0_tokens + lvl0_pos_embed
                _enc_mark('lvl0_patch')


                # update tokens
                x_list = [torch.cat([cr, lvl0],dim=0).unsqueeze(0)\
                                for (cr, lvl0) \
                                in zip(cr_token_list, lvl0_tokens.split(lvl0_token_lens))]
                if profile_encoder_blocks:
                    self.encoder._sat_profile_segment = 'lvl0_update'
                x_list = self.encoder.forward_specific_layers_list(x_list, end=self.sat_cfg['get_map_layer'], get_feature=False)
                lvl0_tokens = _drop_register_tokens_concat(x_list, self.encoder.num_register_tokens)
                assert len(lvl0_pos_x) == len(lvl0_tokens)
                _enc_mark('lvl0_update')
                # also update lvl1 and crs
                lvl1_tokens = updated_lvl1
                cr_token_list = updated_cr_list

                

                if self.sat_cfg['num_lvls'] == 2:
                    # drop corresponding lvl1 tokens
                    lvl1_keep = ~lvl1_sat_mask
                    lvl1_token_lens = [msk.sum().item() for msk in lvl1_keep.split(lvl1_token_lens)]
                    lvl1_keep_idx = lvl1_keep.nonzero(as_tuple=True)[0]
                    torch._check_is_size(lvl1_keep_idx.size(0))
                    lvl1_tokens = lvl1_tokens.index_select(0, lvl1_keep_idx)
                    lvl1_pos_y = lvl1_pos_y.index_select(0, lvl1_keep_idx)
                    lvl1_pos_x = lvl1_pos_x.index_select(0, lvl1_keep_idx)

                    # normalize positions
                    lvl0_pos_y_norm = (lvl0_pos_y.to(dtype=lvl0_tokens.dtype) + 0.5) / self.feature_size[0]
                    lvl0_pos_x_norm = (lvl0_pos_x.to(dtype=lvl0_tokens.dtype) + 0.5) / self.feature_size[0]
                    lvl1_pos_y_norm = (lvl1_pos_y.to(dtype=lvl1_tokens.dtype) + 0.5) / self.feature_size[1]
                    lvl1_pos_x_norm = (lvl1_pos_x.to(dtype=lvl1_tokens.dtype) + 0.5) / self.feature_size[1]

                    # merge all
                    x_list = [torch.cat([cr, lvl0, lvl1]).unsqueeze(0) \
                                    for cr, lvl0, lvl1 \
                                    in zip(cr_token_list, lvl0_tokens.split(lvl0_token_lens), lvl1_tokens.split(lvl1_token_lens))]
                    pos_y_list = [torch.cat([lvl0, lvl1]) \
                                        for lvl0, lvl1 \
                                        in zip(lvl0_pos_y_norm.split(lvl0_token_lens), lvl1_pos_y_norm.split(lvl1_token_lens))]
                    pos_x_list = [torch.cat([lvl0, lvl1]) \
                                        for lvl0, lvl1 \
                                        in zip(lvl0_pos_x_norm.split(lvl0_token_lens), lvl1_pos_x_norm.split(lvl1_token_lens))]
                    lvl_list = [torch.cat([torch.zeros_like(lvl0, dtype=int), torch.ones_like(lvl1, dtype=int)]) \
                                        for lvl0, lvl1 \
                                        in zip(lvl0_pos_x_norm.split(lvl0_token_lens), lvl1_pos_x_norm.split(lvl1_token_lens))]


                else:
                    # prune lvl1 correspond to lvl0
                    lvl1_valid_mask = self.pad_mask(lvl1_valid_mask)
                    lvl1_keep = lvl1_valid_mask & (~lvl1_sat_mask)
                    lvl1_to_lvl2 = ~lvl1_valid_mask

                    token_lvls = [lvl0_tokens, lvl1_tokens]
                    token_lens_lvls = [lvl0_token_lens, lvl1_token_lens]
                    pos_y_lvls = [lvl0_pos_y, lvl1_pos_y]
                    pos_x_lvls = [lvl0_pos_x, lvl1_pos_x]

                    to_next_lvl = lvl1_to_lvl2
                    keep = lvl1_keep
                    lvl_zorders = lvl1_zorders
                    lvl_bids = lvl1_bids
                    pad_vals = torch.full((3,), -1, dtype=lvl_zorders.dtype, device=lvl_zorders.device)
                    for lvl in range(self.sat_cfg['num_lvls']-2):
                        # if to_next_lvl.sum() == 0:
                        #     break
                        next_idx = to_next_lvl.nonzero(as_tuple=True)[0]
                        keep_idx = keep.nonzero(as_tuple=True)[0]
                        torch._check_is_size(next_idx.size(0))
                        torch._check_is_size(keep_idx.size(0))
                        next_tokens = self.lvl_pooling(token_lvls[-1].index_select(0, next_idx))
                        # next_tokens = torch.max(token_lvls[-1][to_next_lvl].view(-1,4,C), dim=1)[0]
                        next_pos_y = pos_y_lvls[-1].index_select(0, next_idx)[::4]>>1
                        next_pos_x = pos_x_lvls[-1].index_select(0, next_idx)[::4]>>1
                        next_lens = [msk.sum().item()//4 for msk in to_next_lvl.split(token_lens_lvls[-1])]


                        token_lvls[-1] = token_lvls[-1].index_select(0, keep_idx)
                        pos_y_lvls[-1] = pos_y_lvls[-1].index_select(0, keep_idx)
                        pos_x_lvls[-1] = pos_x_lvls[-1].index_select(0, keep_idx)
                        token_lens_lvls[-1] = [msk.sum().item() for msk in keep.split(token_lens_lvls[-1])]

                        token_lvls.append(next_tokens)
                        token_lens_lvls.append(next_lens)
                        pos_y_lvls.append(next_pos_y)
                        pos_x_lvls.append(next_pos_x)

                        if lvl < self.sat_cfg['num_lvls']-3:
                            lvl_zorders = lvl_zorders.index_select(0, next_idx)[::4]>>2
                            lvl_bids = lvl_bids.index_select(0, next_idx)[::4]

                            z_starts_idx = torch.where((lvl_zorders&3)==0)[0]
                            padded_z = torch.cat([lvl_zorders, pad_vals])
                            padded_bids = torch.cat([lvl_bids, pad_vals])
                            valids = (padded_z[z_starts_idx] + 3 == padded_z[z_starts_idx + 3]) & (padded_bids[z_starts_idx] == padded_bids[z_starts_idx + 3])
                            valid_starts = z_starts_idx[valids]
                        
                            to_next_lvl = torch.zeros_like(lvl_zorders, dtype=bool)
                            to_next_lvl[valid_starts] = True
                            to_next_lvl[valid_starts+1] = True
                            to_next_lvl[valid_starts+2] = True
                            to_next_lvl[valid_starts+3] = True

                            keep = ~to_next_lvl

                    norm_pos_y_lvls = _normalize_pos_per_lvl(pos_y_lvls, lvl0_tokens.dtype, self.feature_size)
                    norm_pos_x_lvls = _normalize_pos_per_lvl(pos_x_lvls, lvl0_tokens.dtype, self.feature_size)

                    x_list = _build_x_list_multi(cr_token_list, token_lvls, token_lens_lvls)
                    pos_y_list = _build_pos_list_multi(norm_pos_y_lvls, token_lens_lvls)
                    pos_x_list = _build_pos_list_multi(norm_pos_x_lvls, token_lens_lvls)
                    lvl_list = _build_lvl_id_list_multi(norm_pos_x_lvls, token_lens_lvls)

                

        start = self.sat_cfg['get_map_layer'] if self.use_sat else 0
        if profile_encoder_blocks:
            self.encoder._sat_profile_segment = 'final_blocks'
        _, final_feature_list = self.encoder.forward_specific_layers_list(x_list, start = start, norm=True)
        _enc_mark('final_blocks')

        # proj
        token_lens = [feature.shape[1] for feature in final_feature_list]
        final_features = self.feature_proj(torch.cat(final_feature_list,dim=1).squeeze(0)) # (sum(L), C)
        assert tuple(final_features.shape) == (sum(token_lens), self.hidden_dim)
        # positional encoding
        pos_embeds = position_encoding_xy(torch.cat(pos_x_list,dim=0), torch.cat(pos_y_list,dim=0), embedding_dim=self.hidden_dim)
        if self.use_sat and self.sat_cfg['lvl_embed']:
            lvl_embeds = self.level_embed[torch.cat(lvl_list,dim=0)]
            pos_embeds = pos_embeds + lvl_embeds

        sat_dict = {'pos_y': pos_y_list, 'pos_x': pos_x_list, 'lvl': lvl_list, 
                    #  'features': [feature.squeeze(0) for feature in final_feature_list],
                     'lens': token_lens}
        _enc_mark('project_pos')
        if profile_encoder:
            parts = []
            total_ms = 0.0
            for (prev_name, prev_t), (name, cur_t) in zip(enc_marks, enc_marks[1:]):
                elapsed_ms = (cur_t - prev_t) * 1000.0
                total_ms += elapsed_ms
                parts.append(f"{prev_name}->{name}={elapsed_ms:.2f}ms")
            extra = ""
            if enc_profile_extra:
                extra = (
                    f" lvl1_total={enc_profile_extra.get('lvl1_total')} "
                    f"lvl1_valid={enc_profile_extra.get('lvl1_valid')} "
                    f"lvl1_sat={enc_profile_extra.get('lvl1_sat')} "
                    f"lvl0_tokens={enc_profile_extra.get('lvl0_tokens')}"
                )
            print(
                f"[PROFILE_ENCODER] forward={profile_count} total={total_ms:.2f}ms "
                f"{' '.join(parts)} token_lens=[{','.join(str(int(x)) for x in token_lens)}]{extra}"
            )
        if profile_encoder_blocks:
            self.encoder._sat_profile_blocks = prev_encoder_block_profile
            self.encoder._sat_profile_forward_index = prev_encoder_block_forward
            self.encoder._sat_profile_segment = prev_encoder_block_segment

        return final_features, pos_embeds, token_lens, scale_map_dict, sat_dict

    def process_smpl(self, poses, shapes, cam_xys, cam_intrinsics, detach_j3ds = False):
        bs, num_queries, _ = poses.shape # should be (bs,n_q,num_poses*3)

        # flatten and compute
        poses = poses.flatten(0,1) # (bs*n_q,24*3)
        shapes = shapes.flatten(0,1) # (bs*n_q,10)
        verts, joints = self.human_model(poses=poses,
                                         betas=shapes)
        num_verts = verts.shape[1]
        num_joints = joints.shape[1]
        verts = verts.reshape(bs,num_queries,num_verts,3)
        joints = joints.reshape(bs,num_queries,num_joints,3)

        # apply cam_trans and projection
        scale = 2*cam_xys[:,:,2:].sigmoid() + 1e-6
        t_xy = cam_xys[:,:,:2]/scale
        t_z = (2*self.focal)/(scale*self.input_size)    # (bs,num_queries,1)
        transl = torch.cat([t_xy,t_z],dim=2)[:,:,None,:]    # (bs,nq,1,3)

        verts_cam = verts + transl # only for visualization and evaluation
        j3ds_cam = joints + transl

        if detach_j3ds:
            j2ds_homo = torch.matmul(joints.detach() + transl, cam_intrinsics.transpose(2,3))
        else:
            j2ds_homo = torch.matmul(j3ds_cam, cam_intrinsics.transpose(2,3))
        j2ds_img = (j2ds_homo[..., :2] / (j2ds_homo[..., 2, None] + 1e-6)).reshape(bs,num_queries,num_joints,2)

        depths = j3ds_cam[:,:,0,2:]   # (bs, n_q, 1)
        depths = torch.cat([depths, depths/self.focal], dim=-1) # (bs, n_q, 2)

        return verts_cam, j3ds_cam, j2ds_img, depths, transl.flatten(2)


    def forward(self, samples: NestedTensor, targets, sat_use_gt = False, detach_j3ds = False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        assert isinstance(samples, (list, torch.Tensor))

        profile_count = int(getattr(self, '_sat_profile_forward_count', 0))
        profile_limit = int(getattr(self, '_sat_profile_forward_limit', 0))
        profile_forward = (
            bool(getattr(self, '_sat_profile_forward', False))
            and not self.training
            and (profile_limit <= 0 or profile_count < profile_limit)
        )
        profile_marks = []
        profile_device = None
        if profile_forward:
            if isinstance(samples, list) and samples:
                profile_device = samples[0].device
            elif isinstance(samples, torch.Tensor):
                profile_device = samples.device

            def _profile_sync():
                if profile_device is not None and profile_device.type == 'cuda':
                    torch.cuda.synchronize(profile_device)

            def _profile_mark(name):
                _profile_sync()
                #profile_marks.append((name, time.perf_counter()))

            _profile_mark('start')
        else:
            def _profile_mark(name):
                return None

        if self.training:
            self.preprocessed_pos_lvl1 = None

        elif self.preprocessed_pos_lvl1 is None and self.use_sat:
            self.prepare_static_eval_buffers()


        bs = len(targets)

        # get cam_intrinsics
        img_size = torch.stack([t['img_size'].flip(0) for t in targets])
        valid_ratio = img_size/self.input_size
        
        cam_intrinsics = self.cam_intrinsics.repeat(bs, 1, 1, 1)
        cam_intrinsics[...,:2,2] = cam_intrinsics[...,:2,2] * valid_ratio[:, None, :]

        _profile_mark('setup')

        final_features, pos_embeds, token_lens, scale_map_dict, sat_dict\
             = self.forward_encoder(samples, targets, use_gt = sat_use_gt)
        _profile_mark('encoder')

        # default dab-detr pipeline
        embedweight = (self.refpoint_embed.weight).unsqueeze(0).repeat(bs,1,1)
        tgt = (self.tgt_embed.weight).unsqueeze(0).repeat(bs,1,1)

        if self.training and self.use_dn:
            input_query_tgt, input_query_bbox, attn_mask, dn_meta =\
                            prepare_for_cdn(targets = targets, dn_cfg = self.dn_cfg, 
                                        num_queries = self.num_queries, hidden_dim = self.hidden_dim, dn_enc = self.dn_enc)
            tgt = torch.cat([input_query_tgt, tgt], dim=1)
            embedweight = torch.cat([input_query_bbox, embedweight], dim=1)
        else:
            attn_mask = None

        tgt_lens = [tgt.shape[1]]*bs
        _profile_mark('query_setup')

        hs, reference = self.decoder(memory=final_features, memory_lens=token_lens,
                                         tgt=tgt.flatten(0,1), tgt_lens=tgt_lens,
                                         refpoint_embed=embedweight.flatten(0,1),
                                         pos_embed=pos_embeds,
                                         self_attn_mask = attn_mask)
        _profile_mark('decoder')
        
        reference_before_sigmoid = inverse_sigmoid(reference)
        bbox_tmp = self.forward_bbox_embed(hs)
        bbox_tmp[..., :self.query_dim] += reference_before_sigmoid
        pred_boxes = bbox_tmp.sigmoid()
        _profile_mark('bbox')

        outputs_poses = []
        outputs_shapes = []
        outputs_confs = []
        outputs_j3ds = []
        outputs_j2ds = []
        outputs_depths = []

        # shape of hs: (lvl, bs, num_queries, dim)
        outputs_pose_6d = self.mean_pose.view(1, 1, -1)
        outputs_shape = self.mean_shape.view(1, 1, -1)
        for lvl in range(hs.shape[0]):

            outputs_pose_6d = outputs_pose_6d + self.pose_head[lvl](hs[lvl])
            outputs_shape = outputs_shape + self.shape_head[lvl](hs[lvl])

            if self.training or lvl == hs.shape[0] - 1:
                outputs_pose = rot6d_to_axis_angle(outputs_pose_6d)

                outputs_conf = self.conf_head(hs[lvl]).sigmoid()

                # cam
                cam_xys = self.cam_head(hs[lvl])

                outputs_vert, outputs_j3d, outputs_j2d, depth, transl\
                = self.process_smpl(poses = outputs_pose,
                                    shapes = outputs_shape,
                                    cam_xys = cam_xys,
                                    cam_intrinsics = cam_intrinsics,
                                    detach_j3ds = detach_j3ds)
                
                outputs_poses.append(outputs_pose)
                outputs_shapes.append(outputs_shape)
                outputs_confs.append(outputs_conf)
                # outputs_verts.append(outputs_vert)
                outputs_j3ds.append(outputs_j3d)
                outputs_j2ds.append(outputs_j2d)
                outputs_depths.append(depth)
        _profile_mark('heads_smpl')
        
        pred_poses = torch.stack(outputs_poses)
        pred_betas = torch.stack(outputs_shapes)
        pred_confs = torch.stack(outputs_confs)
        pred_verts = outputs_vert
        pred_transl = transl
        pred_intrinsics = cam_intrinsics
        pred_j3ds = torch.stack(outputs_j3ds)
        pred_j2ds = torch.stack(outputs_j2ds)
        pred_depths = torch.stack(outputs_depths)



        if self.training > 0 and self.use_dn:
            pred_poses, pred_betas,\
            pred_boxes, pred_confs,\
            pred_j3ds, pred_j2ds, pred_depths,\
            pred_verts, pred_transl =\
                dn_post_process(pred_poses, pred_betas,
                                pred_boxes, pred_confs,
                                pred_j3ds, pred_j2ds, pred_depths,
                                pred_verts, pred_transl,
                                dn_meta, self.aux_loss, self._set_aux_loss)


        out = {'pred_poses': pred_poses[-1], 'pred_betas': pred_betas[-1],
                'pred_boxes': pred_boxes[-1], 'pred_confs': pred_confs[-1], 
               'pred_j3ds': pred_j3ds[-1], 'pred_j2ds': pred_j2ds[-1],
               'pred_verts': pred_verts, 'pred_intrinsics': pred_intrinsics, 
               'pred_depths': pred_depths[-1], 'pred_transl': pred_transl}
        
        if self.aux_loss and self.training:
            out['aux_outputs'] = self._set_aux_loss(pred_poses, pred_betas,
                                                    pred_boxes, pred_confs,
                                                    pred_j3ds, pred_j2ds, pred_depths)

        if self.use_sat:
            out['enc_outputs'] = scale_map_dict
        
        out['sat'] = sat_dict

        if self.training > 0 and self.use_dn:
            out['dn_meta'] = dn_meta

        if profile_forward:
            _profile_mark('end')
            parts = []
            total_ms = 0.0
            for (prev_name, prev_t), (name, cur_t) in zip(profile_marks, profile_marks[1:]):
                elapsed_ms = (cur_t - prev_t) * 1000.0
                total_ms += elapsed_ms
                parts.append(f"{prev_name}->{name}={elapsed_ms:.2f}ms")
            token_lens_str = ",".join(str(int(x)) for x in token_lens)
            print(
                f"[PROFILE] forward={profile_count} total={total_ms:.2f}ms "
                f"{' '.join(parts)} token_lens=[{token_lens_str}] hs={tuple(hs.shape)}"
            )
            self._sat_profile_forward_count = profile_count + 1

        return out

    @torch.jit.unused
    def _set_aux_loss(self, pred_poses, pred_betas, pred_boxes, 
                        pred_confs, pred_j3ds, 
                        pred_j2ds, pred_depths):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_poses': a, 'pred_betas': b,
                    'pred_boxes': c, 'pred_confs': d, 
                'pred_j3ds': e, 'pred_j2ds': f, 'pred_depths': g}
                    for a, b, c, d, e, f, g in zip(pred_poses[:-1], pred_betas[:-1], 
                    pred_boxes[:-1], pred_confs[:-1], pred_j3ds[:-1], pred_j2ds[:-1], pred_depths[:-1])]



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


def build_sat_model(args, set_criterion=True):
    encoder = build_encoder(args)
    decoder = build_decoder(args)

    model = Model(
        encoder,
        decoder,
        num_queries=args.num_queries,
        input_size=args.input_size,
        sat_cfg=args.sat_cfg,
        dn_cfg=args.dn_cfg,
        train_pos_embed=getattr(args,'train_pos_embed',True)
    )


    if set_criterion:
        matcher = build_matcher(args)
        weight_dict = args.weight_dict
        losses = args.losses

        if args.dn_cfg['use_dn']:
            dn_weight_dict = {}
            dn_weight_dict.update({f'{k}_dn': v for k, v in weight_dict.items()})
            weight_dict.update(dn_weight_dict)

        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({f'{k}.{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        if args.sat_cfg['use_sat']:
            if 'map_confs' not in weight_dict:
                weight_dict.update({'map_confs': weight_dict['confs']})
            # weight_dict.update({'map_scales': })

        criterion = SetCriterion(matcher, weight_dict, losses = losses, j2ds_norm_scale = args.input_size)
        return model, criterion
    else:
        return model, None
