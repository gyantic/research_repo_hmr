import accelerate
from packaging import version
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from datasets.multiple_datasets import MultipleDatasets, datasets_dict
from datasets.common import COMMON
from transformers import get_scheduler
from safetensors.torch import load_file
import os
import re
import time
import datetime
from models import build_sat_model
from .funcs.eval_funcs import *
from .funcs.infer_funcs import inference
from utils import misc
from utils.misc import get_world_size
import torch.multiprocessing
import numpy as np

assert version.parse(accelerate.__version__) >= version.parse("1.10.0"),\
      "Please use accelerate >= 1.10.0 to support our updated implementation of distributed evaluation (August 30, 2025)."


def is_bf16(dtype_cfg):
    return str(dtype_cfg).lower() in {"bfloat16", "bf16", "bfp16"}


def dequantize_int8_weight_state_dict_if_needed(state_dict):
    int8_keys = [k for k in state_dict if k.endswith(".weight_int8")]
    if not int8_keys:
        return state_dict, {"loaded_int8_weight_file": False, "dequantized_tensors": 0}

    restored = {}
    for key, tensor in state_dict.items():
        if key.endswith(".weight_int8"):
            base = key[: -len("_int8")]
            scale_key = base + "_scale"
            if scale_key not in state_dict:
                raise KeyError(f"Missing INT8 weight scale for {key}: expected {scale_key}")
            scale = state_dict[scale_key].float().unsqueeze(1)
            restored[base] = tensor.float() * scale
        elif key.endswith(".weight_scale"):
            continue
        else:
            restored[key] = tensor

    return restored, {
        "loaded_int8_weight_file": True,
        "dequantized_tensors": len(int8_keys),
    }


def apply_torchao_int4(model, skip_modules=(), group_size=128):
    import torch.nn as nn
    try:
        from torchao.quantization import quantize_, int4_weight_only
    except Exception as exc:
        raise RuntimeError("torchao is not available. Install torchao before enabling torchao_int4.") from exc

    def _filter(module, fqn):
        if not isinstance(module, nn.Linear):
            return False
        if skip_modules and any(s in fqn for s in skip_modules):
            return False
        import torch
        w = module.weight
        if type(w) not in (torch.Tensor, nn.Parameter):
            return False
        if w.shape[1] % group_size != 0:
            return False
        return w.shape[0] % 16 == 0

    quantize_(model, int4_weight_only(group_size=group_size), filter_fn=_filter)
    return f"int4_weight_only(group_size={group_size})"


def apply_torchao_fp8(model):
    try:
        from torchao.quantization import quantize_, float8_weight_only
    except Exception as exc:
        raise RuntimeError("torchao is not available. Install torchao before enabling torchao_fp8.") from exc
    quantize_(model, float8_weight_only())
    return "float8_weight_only()"


def _qat_prepare(model, skip_modules=(), group_size=128):
    """INT4 weight-only QAT via module swap. Matches `int4_weight_only()` numerics.

    - Weight: INT4 per-channel-group asymmetric, ZeroPointDomain.FLOAT (group_size=128)
    - No activation fake quant (weight-only to match save-time int4_weight_only)

    Layers skipped:
    - in_features % group_size != 0  (e.g. dn_enc with in_features=82)
    - Explicitly listed in skip_modules
    """
    import torch.nn as nn
    import torch.nn.functional as F
    from torchao.quantization.utils import get_groupwise_affine_qparams
    from torchao.quantization.prototype.qat.utils import _fake_quantize_per_channel_group
    from torchao.quantization.quant_primitives import ZeroPointDomain

    class _QATLinear(nn.Linear):
        """nn.Linear with INT4 weight-only fake quantization. Supports bias."""
        def __init__(self, in_f, out_f, bias=True, device=None,
                     groupsize=128, precision=torch.float32,
                     scales_precision=torch.float32):
            super().__init__(in_f, out_f, bias, device=device, dtype=precision)
            assert in_f % groupsize == 0, \
                f"in_features {in_f} not divisible by groupsize {groupsize}"
            self.groupsize = groupsize
            self.precision = precision
            self.scales_precision = scales_precision
            self._fake_quant_enabled = True

        def forward(self, x):
            if self._fake_quant_enabled:
                w_s, w_zp = get_groupwise_affine_qparams(
                    self.weight, 4, self.groupsize, self.scales_precision,
                )
                w_fq = _fake_quantize_per_channel_group(
                    self.weight, w_s, w_zp, 0, 15,
                    self.groupsize, ZeroPointDomain.FLOAT,
                )
            else:
                w_fq = self.weight
            return F.linear(x, w_fq, self.bias)

    replaced = []
    skipped = []

    def _replace(module, prefix=''):
        for name, child in module.named_children():
            fqn = f'{prefix}.{name}' if prefix else name
            if isinstance(child, nn.Linear) and not isinstance(child, _QATLinear):
                if skip_modules and any(s in fqn for s in skip_modules):
                    skipped.append(fqn)
                    continue
                if child.weight.shape[1] % group_size != 0:
                    skipped.append(fqn)
                    continue
                qat_lin = _QATLinear(
                    child.in_features, child.out_features,
                    bias=(child.bias is not None),
                    device=child.weight.device,
                    groupsize=group_size,
                    precision=child.weight.dtype,
                    scales_precision=torch.float32,
                )
                qat_lin.weight = child.weight
                if child.bias is not None:
                    qat_lin.bias = child.bias
                setattr(module, name, qat_lin)
                replaced.append(fqn)
            else:
                _replace(child, fqn)

    _replace(model)
    print(f'[QAT] Replaced {len(replaced)} Linear layers, skipped {len(skipped)}')
    if skipped:
        print(f'[QAT] Skipped: {skipped}')
    return model


def apply_torchao_int8(model, weight_only=False, skip_modules=()):
    try:
        from torchao.quantization import quantize_
    except Exception as exc:
        raise RuntimeError(
            "torchao is not available in the active environment. "
            "Install torchao before enabling torchao_int8."
        ) from exc

    if weight_only:
        try:
            from torchao.quantization import int8_weight_only
            qcfg = int8_weight_only()
            qcfg_name = "int8_weight_only()"
        except Exception:
            from torchao.quantization import Int8WeightOnlyConfig
            qcfg = Int8WeightOnlyConfig()
            qcfg_name = "Int8WeightOnlyConfig()"
        quantize_(model, qcfg)
        return qcfg_name

    try:
        from torchao.quantization import int8_dynamic_activation_int8_weight
        qcfg = int8_dynamic_activation_int8_weight()
        qcfg_name = "int8_dynamic_activation_int8_weight()"
    except Exception:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig
        qcfg = Int8DynamicActivationInt8WeightConfig()
        qcfg_name = "Int8DynamicActivationInt8WeightConfig()"

    import torch.nn as nn

    stats = {
        "quantized": 0,
        "skipped_name": [],
        "skipped_dim": [],
        "skipped_nonplain": 0,
    }

    def _filter(module, fqn):
        if not isinstance(module, nn.Linear):
            return False
        if skip_modules and any(s in fqn for s in skip_modules):
            stats["skipped_name"].append(fqn)
            return False
        weight = getattr(module, "weight", None)
        if type(weight) not in (torch.Tensor, nn.Parameter):
            stats["skipped_nonplain"] += 1
            return False
        if module.in_features % 8 != 0 or module.out_features % 8 != 0:
            stats["skipped_dim"].append((fqn, module.in_features, module.out_features))
            return False
        stats["quantized"] += 1
        return True

    quantize_(model, qcfg, filter_fn=_filter)
    skipped_dim = len(stats["skipped_dim"])
    preview = ", ".join(
        f"{name}({in_f}->{out_f})"
        for name, in_f, out_f in stats["skipped_dim"][:8]
    )
    detail = (
        f"{qcfg_name}, quantized={stats['quantized']}, "
        f"skipped_name={len(stats['skipped_name'])}, "
        f"skipped_dim={skipped_dim}"
    )
    if preview:
        detail += f", skipped_dim_preview=[{preview}]"
    if stats["skipped_nonplain"]:
        detail += f", skipped_nonplain={stats['skipped_nonplain']}"
    return detail

class Engine():
    def __init__(self, args, mode='train'): 
        self.exp_name = args.exp_name
        self.mode = mode
        assert mode in ['train','eval','infer']
        self.conf_thresh = args.conf_thresh
        self.eval_func_maps = {'agora_validation': evaluate_agora,
                                'bedlam_validation_6fps': evaluate_agora,
                                'agora_test': test_agora,
                                '3dpw_test': evaluate_3dpw,}
        self.inference_func = inference

        if self.mode == 'train':
            self.output_dir = os.path.join('./outputs')
            self.log_dir = os.path.join(self.output_dir,'logs')
            self.ckpt_dir = os.path.join(self.output_dir,'ckpts')
            self.distributed_eval = args.distributed_eval
            self.eval_vis_num = args.eval_vis_num
        elif self.mode == 'eval':
            self.output_dir = os.path.join('./results')
            self.distributed_eval = args.distributed_eval
            self.eval_vis_num = args.eval_vis_num
        elif self.mode == 'infer':
            output_dir = getattr(args, 'output_dir', None)
            if output_dir is not None:
                self.output_dir = output_dir
            else:
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                self.output_dir = os.path.join('./results',f'{self.exp_name}_infer_{timestamp}')
            self.distributed_infer = args.distributed_infer

        self.prepare_accelerator()
        self.prepare_models(args)
        self.prepare_datas(args)
        if self.mode == 'train':
            self.prepare_training(args)

        total_cnt = sum(p.numel() for p in self.model.parameters())
        trainable_cnt = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.accelerator.print(f'Initialization finished.\n{trainable_cnt} trainable parameters({total_cnt} total).')       

    def prepare_accelerator(self):
        if self.mode == 'train':
            self.accelerator = Accelerator(
                log_with="tensorboard",
                project_dir=os.path.join(self.log_dir)
            )
            if self.accelerator.is_main_process:
                os.makedirs(self.log_dir, exist_ok=True)
                os.makedirs(os.path.join(self.ckpt_dir,self.exp_name),exist_ok=True)
                self.accelerator.init_trackers(self.exp_name)
        else:
            self.accelerator = Accelerator()
            if self.accelerator.is_main_process:
                os.makedirs(self.output_dir, exist_ok=True)
        
    def prepare_models(self, args):
        # load model and criterion
        self.accelerator.print('Preparing models...')
        self.unwrapped_model, self.criterion = build_sat_model(args, set_criterion = (self.mode == 'train'))
        if self.criterion is not None:
            self.weight_dict = self.criterion.weight_dict

        use_bf16 = (self.mode != 'train') and is_bf16(getattr(args, 'dtype_cfg', 'fp32'))
        use_torchao_int8 = bool(getattr(args, 'torchao_int8', False))
        use_torchao_int4 = bool(getattr(args, 'torchao_int4', False))
        use_torchao_fp8  = bool(getattr(args, 'torchao_fp8', False))
        # When loading a QAT-saved checkpoint, state_dict already contains AQT
        # tensors, so apply_torchao_int4 must run BEFORE load_state_dict to
        # prepare matching AQT slots on the model.
        qat_weights = bool(getattr(args, 'qat_weights', False)) and use_torchao_int4
        int4_skip = list(getattr(args, 'torchao_int4_skip_modules', []))

        def _load_pretrain():
            self.accelerator.print(f'Loading pretrained weights: {args.pretrain_path}')
            state_dict = torch.load(args.pretrain_path, weights_only=False)
            state_dict, int8_weight_info = dequantize_int8_weight_state_dict_if_needed(state_dict)
            if int8_weight_info["loaded_int8_weight_file"]:
                self.accelerator.print(
                    'Loaded INT8 weight file: '
                    f'dequantized_tensors={int8_weight_info["dequantized_tensors"]}'
                )
            self.unwrapped_model.load_state_dict(state_dict, strict=False)

        if args.pretrain and not qat_weights:
            _load_pretrain()

        # QAT: insert FakeQuantize before DDP wrapping (train mode only)
        self.use_qat = self.mode == 'train' and bool(getattr(args, 'use_qat', False))
        if self.use_qat:
            self._qat_args = args
            skip = list(getattr(args, 'qat_skip_modules', []))
            self.unwrapped_model = _qat_prepare(self.unwrapped_model, skip)
            self.accelerator.print(
                f'QAT prepare (W4A8 module swap): FakeQuantize inserted. '
                f'Skipping modules: {skip if skip else "none"}'
            )

        self.unwrapped_model.input_dtype = torch.bfloat16 if use_bf16 else torch.float32

        if use_bf16 or use_torchao_int8 or use_torchao_int4 or use_torchao_fp8:
            if self.mode == 'train' and (use_torchao_int8 or use_torchao_int4 or use_torchao_fp8):
                raise NotImplementedError('torchao quantization is only supported for eval/infer.')
            self.unwrapped_model = self.unwrapped_model.to(self.accelerator.device)
            if use_bf16:
                self.unwrapped_model = self.unwrapped_model.to(torch.bfloat16)
                self.accelerator.print('Using bfloat16 model/input.')

        if use_torchao_int8:
            weight_only = bool(getattr(args, 'torchao_int8_weight_only', False))
            qcfg_name = apply_torchao_int8(self.unwrapped_model, weight_only=weight_only)
            self.accelerator.print(f'Applying torchao int8: {qcfg_name}')

        if use_torchao_int4:
            self.unwrapped_model.to(torch.bfloat16)
            qcfg_name = apply_torchao_int4(self.unwrapped_model, skip_modules=int4_skip)
            self.accelerator.print(f'Applying torchao int4: {qcfg_name} (skip={int4_skip})')

        if use_torchao_fp8:
            self.unwrapped_model.to(torch.bfloat16)
            qcfg_name = apply_torchao_fp8(self.unwrapped_model)
            self.accelerator.print(f'Applying torchao fp8: {qcfg_name}')

        if args.pretrain and qat_weights:
            _load_pretrain()

        # to gpu
        self.model = self.accelerator.prepare(self.unwrapped_model)
        self.model.input_dtype = self.unwrapped_model.input_dtype
        
    def prepare_datas(self, args):
        # load dataset and dataloader
        if self.mode == 'train':
            self.accelerator.print('Loading training datasets:\n',
                            [f'{d}_{s}' for d,s in zip(args.train_datasets_used, args.train_datasets_split)])
            self.train_batch_size = args.train_batch_size
            train_dataset = MultipleDatasets(args.train_datasets_used, args.train_datasets_split,
                                        make_same_len=False, input_size=args.input_size, aug=True, 
                                        mode = 'train', sat_cfg=args.sat_cfg,
                                        aug_cfg=args.aug_cfg)
            self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.train_batch_size,
                                                shuffle=True,collate_fn=misc.collate_fn, 
                                                num_workers=args.train_num_workers,pin_memory=True)
            self.train_dataloader = self.accelerator.prepare(self.train_dataloader)                                 

        if self.mode != 'infer':
            if getattr(args, 'eval_datasets_used', []):
                self.accelerator.print('Loading evaluation datasets:',
                                    [f'{d}_{s}' for d,s in zip(args.eval_datasets_used, args.eval_datasets_split)])
                self.eval_batch_size = args.eval_batch_size
                eval_ds = {f'{ds}_{split}': datasets_dict[ds](split = split,
                                                              mode = 'eval',
                                                              input_size = args.input_size,
                                                              aug = False,
                                                              sat_cfg=args.sat_cfg)\
                            for (ds, split) in zip(args.eval_datasets_used, args.eval_datasets_split)}
                self.eval_dataloaders = {k: DataLoader(dataset=v, batch_size=self.eval_batch_size,
                                            shuffle=False,collate_fn=misc.collate_fn,
                                            num_workers=args.eval_num_workers,pin_memory=True)\
                                        for (k,v) in eval_ds.items()}
                if self.distributed_eval:
                    for (k,v) in self.eval_dataloaders.items():
                        self.eval_dataloaders.update({k: self.accelerator.prepare(v)})
            else:
                self.eval_dataloaders = {}

        else:
            img_folder = args.input_dir
            self.accelerator.print(f'Loading inference images from {img_folder}')
            self.infer_batch_size = args.infer_batch_size
            infer_ds = COMMON(img_folder = img_folder, input_size=args.input_size,aug=False,
                                mode = 'infer', sat_cfg=args.sat_cfg)
            self.infer_dataloader = DataLoader(dataset=infer_ds, batch_size=self.infer_batch_size,
                                        shuffle=False,collate_fn=misc.collate_fn, 
                                        num_workers=args.infer_num_workers,pin_memory=True)

            if self.distributed_infer:
                self.infer_dataloader = self.accelerator.prepare(self.infer_dataloader)

    def prepare_training(self, args):
        self.start_epoch = 0
        self.num_epochs = args.num_epochs
        self.global_step = 0
        if hasattr(args, 'sat_gt_epoch'):
            self.sat_gt_epoch = args.sat_gt_epoch
            self.accelerator.print(f'Use GT for the first {self.sat_gt_epoch} epoch(s)...')
        else:
            self.sat_gt_epoch = -1
        self.save_and_eval_epoch = args.save_and_eval_epoch
        self.least_eval_epoch = args.least_eval_epoch

        self.detach_j3ds = args.detach_j3ds

        self.accelerator.print('Preparing optimizer and lr_scheduler...')   
        param_dicts = [
            {
                "params":
                    [p for n, p in self.unwrapped_model.named_parameters()
                    if not misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": 
                    [p for n, p in self.unwrapped_model.named_parameters() 
                    if misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad],
                "lr": args.lr_encoder,
            }
        ]

        # optimizer
        if args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise NotImplementedError
        
        # lr_scheduler
        if args.lr_scheduler == 'cosine':
            self.lr_scheduler = get_scheduler(name="cosine", optimizer=self.optimizer, 
                                          num_warmup_steps=args.num_warmup_steps, 
                                          num_training_steps=get_world_size() * self.num_epochs * len(self.train_dataloader)) 
        elif args.lr_scheduler == 'multistep':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, args.milestones, gamma=args.gamma)  
        else:
            raise NotImplementedError      

        self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.optimizer, self.lr_scheduler)

        # resume
        if args.resume: #load model, optimizer, lr_scheduler and random_state
            if hasattr(args, 'ckpt_epoch'):
                self.load_ckpt(args.ckpt_epoch,args.ckpt_step)    
            else:
                self.accelerator.print('Auto resume from the latest ckpt...')
                epoch, step = -1, -1
                pattern = re.compile(r'epoch_(\d+)_step_(\d+)')
                for folder_name in os.listdir(os.path.join(self.output_dir,'ckpts',self.exp_name)):
                    match = pattern.match(folder_name)
                    if match:
                        i, j = int(match.group(1)), int(match.group(2))
                        if i > epoch:
                            epoch, step = i, j
                if epoch >= 0:
                    self.load_ckpt(epoch, step)    
                else:
                    self.accelerator.print('No existing ckpts! Train from scratch.')               

    def load_ckpt(self, epoch, step):   
        self.accelerator.print(f'Loading checkpoint: epoch_{epoch}_step_{step}') 
        ckpts_save_path = os.path.join(self.output_dir,'ckpts',self.exp_name, f'epoch_{epoch}_step_{step}')
        self.start_epoch = epoch + 1
        self.global_step = step + 1
        self.accelerator.load_state(ckpts_save_path)

    def train(self):
        # torch.autograd.set_detect_anomaly(True)
        self.accelerator.print('Start training!')
        for epoch in range(self.start_epoch, self.num_epochs):
            torch.cuda.empty_cache()
            progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            self.model.train()
            self.criterion.train()

            sat_use_gt = (epoch < self.sat_gt_epoch)

            for step, (samples,targets) in enumerate(self.train_dataloader):

                outputs = self.model(samples, targets, sat_use_gt = sat_use_gt, detach_j3ds = self.detach_j3ds)
                loss_dict = self.criterion(outputs, targets)

                loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys())
                

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
                reduced_dict = self.accelerator.reduce(loss_dict,reduction='mean')
                simplified_logs = {k: v.item() for k, v in reduced_dict.items() if '.' not in k}
                
                # logs.update({"lr": self.lr_scheduler.get_last_lr()[0], "step": self.global_step})
                if self.accelerator.is_main_process:
                    tqdm.write(f'[{epoch}-{step+1}/{len(self.train_dataloader)}]: ' + str(simplified_logs))

                if step % 10 == 0:
                    self.accelerator.log({('train/'+k):v for k,v in simplified_logs.items()},
                                            step=self.global_step)

                progress_bar.update(1)
                progress_bar.set_postfix(**{"lr": self.lr_scheduler.get_last_lr()[0], "step": self.global_step})

                self.global_step += 1
                self.accelerator.wait_for_everyone()

            # self.lr_scheduler.step()

            if epoch % self.save_and_eval_epoch == 0 or epoch == self.num_epochs-1:
                self.save_and_eval(epoch, save_ckpt=True)

        if getattr(self, 'use_qat', False):
            self._convert_qat()

        self.accelerator.end_training()

    def eval(self, results_save_path = None, epoch = -1):
        if results_save_path is None:
            results_save_path = os.path.join(self.output_dir,self.exp_name,'evaluation')
        # preparing
        self.model.eval()
        unwrapped_model = self.unwrapped_model # self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            os.makedirs(results_save_path,exist_ok=True)
        # evaluate
        for i, (key, eval_dataloader) in enumerate(self.eval_dataloaders.items()):
            assert key in self.eval_func_maps
            img_cnt = len(eval_dataloader) * self.eval_batch_size
            if self.distributed_eval:
                img_cnt *= self.accelerator.num_processes
            self.accelerator.print(f'Evaluate on {key}: (about) {img_cnt} images')
            self.accelerator.print('Using following threshold(s): ', self.conf_thresh)
            conf_thresh = self.conf_thresh  # if 'agora' in key or 'bedlam' in key else [0.2]
            for thresh in conf_thresh:
                if self.accelerator.is_main_process or self.distributed_eval:
                    error_dict = self.eval_func_maps[key](model = unwrapped_model, 
                                    eval_dataloader = eval_dataloader, 
                                    conf_thresh = thresh,
                                    vis_step = img_cnt // self.eval_vis_num,
                                    results_save_path = os.path.join(results_save_path,key,f'thresh_{thresh}'),
                                    distributed = self.distributed_eval,
                                    accelerator = self.accelerator,
                                    vis=True)
                    if isinstance(error_dict,dict) and self.mode == 'train':
                        log_dict = flatten_dict(error_dict)
                        self.accelerator.log({(f'{key}_thresh_{thresh}/'+k):v for k,v in log_dict.items()}, step=epoch)

                    self.accelerator.print(f'thresh_{thresh}: ',error_dict)
                self.accelerator.wait_for_everyone() 
  
    def save_and_eval(self, epoch, save_ckpt=False):
        torch.cuda.empty_cache()
        # save current state and model
        if self.accelerator.is_main_process and save_ckpt:
            ckpts_save_path = os.path.join(self.output_dir,'ckpts',self.exp_name, f'epoch_{epoch}_step_{self.global_step-1}')
            os.makedirs(ckpts_save_path,exist_ok=True)
            self.accelerator.save_state(ckpts_save_path, safe_serialization=False)
        self.accelerator.wait_for_everyone()
        
        if epoch < self.least_eval_epoch:
            return
        results_save_path = os.path.join(self.output_dir,'results',self.exp_name, f'epoch_{epoch}_step_{self.global_step-1}')        
        self.eval(results_save_path, epoch=epoch)

    def infer(self):
        self.model.eval()
        # unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model = self.unwrapped_model 
        
        results_save_path = self.output_dir
        if self.accelerator.is_main_process:
            os.makedirs(results_save_path,exist_ok=True)
        
        self.accelerator.print('Using following threshold(s): ', self.conf_thresh)
        for thresh in self.conf_thresh:
            if self.accelerator.is_main_process or self.distributed_infer:
                self.inference_func(model = unwrapped_model, 
                        infer_dataloader = self.infer_dataloader, 
                        conf_thresh = thresh,
                        results_save_path = os.path.join(results_save_path,f'thresh_{thresh}'),
                        distributed = self.distributed_infer,
                        accelerator = self.accelerator)
            self.accelerator.wait_for_everyone()


    def _convert_qat(self):
        """Restore _QATLinear -> nn.Linear, apply torchao INT4, save state_dict.

        Result: state_dict keys identical to the FP32 pretrain; weight values
        are INT4 (AffineQuantizedTensor subclass on nn.Linear).
        """
        import torch.nn as nn

        def _restore(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear) and hasattr(child, '_fake_quant_enabled'):
                    lin = nn.Linear(
                        child.in_features, child.out_features,
                        bias=(child.bias is not None),
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                    lin.weight = child.weight
                    if child.bias is not None:
                        lin.bias = child.bias
                    setattr(module, name, lin)
                else:
                    _restore(child)

        self.accelerator.print('QAT: restoring Linear modules and applying INT4...')
        _restore(self.unwrapped_model)
        self.unwrapped_model.to(torch.bfloat16)
        skip = list(getattr(self._qat_args, 'qat_skip_modules', []))
        apply_torchao_int4(self.unwrapped_model, skip_modules=skip, group_size=128)

        if self.accelerator.is_main_process:
            save_path = os.path.join(self.ckpt_dir, self.exp_name, 'qat_int4_weights.pth')
            torch.save(self.unwrapped_model.state_dict(), save_path)
            self.accelerator.print(f'QAT INT4 weights saved: {save_path}')


def flatten_dict(d, parent_key='', sep='-'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
