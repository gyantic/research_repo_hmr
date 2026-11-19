from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import torch
import torch.nn as nn

import config
from config import args
from utils import print_dict

BN_MOMENTUM = 0.1
default_cfg = {'mode':'val', 'calc_loss': False}

def apply_model_precision(model, precision):
    """Cast the weights for args().model_precision; the only place model weights are cast.

    bf16: the whole model. fp16: the backbone only (kept from the earlier fp16 path).
    fp32: unchanged. The input is cast to match in Base.feed_forward.
    """
    if precision == 'bf16':
        return model.to(torch.bfloat16)
    if precision == 'fp16':
        model.backbone.half()
    return model

class Base(nn.Module):
    def forward(self, meta_data, **cfg):
        if cfg['mode'] == 'matching_gts':
            return self.matching_forward(meta_data, **cfg)
        elif cfg['mode'] == 'parsing':
            return self.parsing_forward(meta_data, **cfg)
        elif cfg['mode'] == 'forward':
            return self.pure_forward(meta_data, **cfg)
        else:
            raise NotImplementedError('forward mode is not recognized! please set proper mode (parsing/matching_gts)')

    def matching_forward(self, meta_data, **cfg):
        outputs = self.feed_forward(meta_data)
        outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)
        outputs['meta_data'] = meta_data
        if cfg['calc_loss']:
            outputs.update(self._calc_loss(outputs))
        #print_dict(outputs)
        return outputs

    @torch.no_grad()
    def parsing_forward(self, meta_data, **cfg):
        outputs = self.feed_forward(meta_data)
        outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)
        outputs['meta_data'] = meta_data
        #print_dict(outputs)
        return outputs

    def feed_forward(self, meta_data):
        img = meta_data['image'].contiguous().cuda(non_blocking=True)
        # No autocast: cast the input (uint8 or float32 image) to the dtype of the first
        # parameter, i.e. the backbone weights (see apply_model_precision).
        img = img.to(next(self.parameters()).dtype)
        x = self.backbone(img)
        outputs = self.head_forward(x)
        return outputs

    @torch.no_grad()
    def pure_forward(self, meta_data, **cfg):
        outputs = self.feed_forward(meta_data)
        return outputs

    def head_forward(self,x):
        return NotImplementedError

    def make_backbone(self):
        return NotImplementedError

    def backbone_forward(self, x):
        return NotImplementedError

    def _build_gpu_tracker(self):
        self.gpu_tracker = MemTracker()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)