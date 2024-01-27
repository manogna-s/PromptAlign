import os.path as osp
from collections import OrderedDict
import math
import copy
import shutil
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import pandas as pd

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

import logging
# import wandb

# TPT imports
from copy import deepcopy
import torch.backends.cudnn as cudnn
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import math
import json
import random
from torch.utils.data import Dataset
import numpy as np
from utils.tools import Summary, ProgressMeter, accuracy, load_model_weight, set_random_seed
from utils.tools import AverageMeter as AverageMeter_TPT
import datasets.augmix_ops as augmentations
import time
from tqdm import tqdm
from trainers.prompt_align import PromptAlign
################################

from pdb import set_trace as stx

from clip import clip
from clip import tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from copy import deepcopy

_tokenizer = _Tokenizer()

from trainers.prompt_align import load_clip_to_cpu, TextEncoder, MultiModalPromptLearner, CustomCLIP, _get_clones, \
    ID_to_DIRNAME, fewshot_datasets, path_dict, pug_setting_dir, \
     BaseJsonDataset, Aircraft, get_preaugment, augmix, AugMixAugmenter


def update_ema_variables(ema_model, model, alpha_teacher):    
    ema_model.prompt_learner.ctx[:].data[:] = alpha_teacher * ema_model.prompt_learner.ctx[:].data[:] + (1 - alpha_teacher) * model.prompt_learner.ctx[:].data[:]
    ema_model.prompt_learner.ctxv2[:].data[:] = alpha_teacher * ema_model.prompt_learner.ctxv2[:].data[:] + (1 - alpha_teacher) * model.prompt_learner.ctxv2[:].data[:]
    ema_model.prompt_learner.compound_prompts_text[0][:].data[:] = alpha_teacher * ema_model.prompt_learner.compound_prompts_text[0][:].data[:] + (1 - alpha_teacher) * model.prompt_learner.compound_prompts_text[0][:].data[:]
    ema_model.prompt_learner.compound_prompts_text[1][:].data[:] = alpha_teacher * ema_model.prompt_learner.compound_prompts_text[1][:].data[:] + (1 - alpha_teacher) * model.prompt_learner.compound_prompts_text[1][:].data[:]
    return ema_model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MyMultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.learned_cls = False  # Just copied, check if setting to True
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PROMPTALIGN.N_CTX
        ctx_init = cfg.TRAINER.PROMPTALIGN.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.PROMPTALIGN.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.PROMPTALIGN.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        # self.proj.half()
        # ctx_vectors = ctx_vectors.unsqueeze(0).expand(n_cls, -1, -1)
        self.ctx = nn.Parameter(ctx_vectors)
        self.ctxv2 = nn.Parameter(ctx_vectors.unsqueeze(0).expand(n_cls, -1, -1))
        
        self.proj_weight_init_state = self.proj.weight.detach().clone()
        self.proj_bias_init_state = self.proj.bias.detach().clone()
        self.ctx_init_state = ctx_vectors.detach().clone()

        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Copy init state
        self.compound_prompts_text_init_state = [txt_prompt.detach().clone() for txt_prompt in self.compound_prompts_text]

        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)
        self.compound_prompt_projections_init_state = [(module.weight.detach().clone(), module.bias.detach().clone()) for module in self.compound_prompt_projections]

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctxv2

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required
    
    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        self.ctxv2.data = self.ctx_init_state.unsqueeze(0).expand(self.n_cls, -1, -1)
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

        with torch.no_grad():
            self.proj.weight.copy_(self.proj_weight_init_state)
            self.proj.bias.copy_(self.proj_bias_init_state)

            for idx, prompt in enumerate(self.compound_prompts_text):
                prompt.copy_(self.compound_prompts_text_init_state[idx])
            
            for idx, module in enumerate(self.compound_prompt_projections):
                module.weight.copy_(self.compound_prompt_projections_init_state[idx][0])
                module.bias.copy_(self.compound_prompt_projections_init_state[idx][1])

    def reset_classnames(self, classnames, args):
        self.device = self.ctx.device
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip = load_clip_to_cpu(args).to(self.device)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def set_prompt_init_states(self):
        '''
        Store the initial prompts
        '''
        ctx_vectors = self.ctx.detach().clone()
        self.ctx_init_state = ctx_vectors
        self.proj_weight_init_state = self.proj.weight.detach().clone()
        self.proj_bias_init_state = self.proj.bias.detach().clone()

        self.compound_prompts_text_init_state = [txt_prompt.detach().clone() for txt_prompt in self.compound_prompts_text]
        self.compound_prompt_projections_init_state = [(module.weight.detach().clone(), module.bias.detach().clone()) for module in self.compound_prompt_projections]


class MyCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MyMultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)
               
        return logits
    

    def get_text_features(self):
        # with torch.no_grad():
        tokenized_prompts = self.tokenized_prompts

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_image_txt_features(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        return [image_features, text_features]

    
    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

    def set_prompt_inits(self):
        print("Re-updating prompt initializations to current prompts.")
        self.prompt_learner.set_prompt_init_states()


@TRAINER_REGISTRY.register()
class Last_Attempt(PromptAlign):
    def tpt(self):
        """
        Run Test-time prompt Tuning
        """
        self.model.set_prompt_inits()   # Init with current prompts
        self.base_model = deepcopy(self.model)
        self.base_model.eval()
        with torch.no_grad():
            self.base_model.reset()
        
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        
        # define optimizer
        print(f'Using AdamW optimizer with learning rate:{self.cfg.TPT.LR}')
        print(f'TPT Config:', self.cfg.TPT)
        print('=> Using native Torch AMP. Training in mixed precision.')
        print("number of test samples: {}".format(len(self.tpt_loader.dataset)))

        cudnn.benchmark = True

        results = {}
        set_id = self.cfg.DATASET.TPT
        results[set_id] = self.groups_ema_eata(self.tpt_loader, self.model, self.cfg)

        return results


    def groups_ema_eata(self, val_loader, model, cfg):
        """
        define subgroups; select prompt based on which subgroup it is in
        aad/ent + distr_align/ploga; can specify N_cls; EMA factor
        """
        label2class = json.load(open('data/domainnet/domainnet126_lists/label2class.json'))

        group_name = cfg.TPT.GROUPS
        n_groups = cfg.TPT.GROUPS
        n_cls = 126
        
        # Setup meters, save ckpts, logs etc.        
        experiment = f'EMA{cfg.TPT.EMA}_lr{cfg.TPT.LR}_{group_name}'
        log_dir = f'output/evaluation/{cfg.TRAINER.NAME}/{group_name}/RESET{cfg.TPT.RESET_STEPS}/{cfg.TPT.EATA_TYPE}/seed{cfg.SEED}'
        os.makedirs(log_dir, exist_ok=True)

        group_meters_base = []
        group_meters_tta = []
        ema_update_idx = {}
        for i in range(n_groups):
            group_meters_base.append(AverageMeter_TPT(f'Group{i}_Acc', ':6.2f', Summary.AVERAGE))
            group_meters_tta.append(AverageMeter_TPT(f'Group{i}_Acc', ':6.2f', Summary.AVERAGE))
            ema_update_idx[i] = []

        cls_meters = []
        for i in range(n_cls):
            cls_meters.append(AverageMeter_TPT(f'Cls{i}_Acc', ':6.2f', Summary.AVERAGE))

        ckpt_all = {}
        ckpt_all['base'] = [self.base_model.prompt_learner.ctx.cpu(), \
            self.base_model.prompt_learner.compound_prompts_text[0].cpu(), self.base_model.prompt_learner.compound_prompts_text[1].cpu()]

        logs_all = {}
        
        batch_time = AverageMeter_TPT('Time', ':6.3f', Summary.NONE)
        base_top1 = AverageMeter_TPT('Base_Acc@1', ':6.2f', Summary.AVERAGE)
        aug_top1 = AverageMeter_TPT('Aug_Acc@1', ':6.2f', Summary.AVERAGE)
        top1 = AverageMeter_TPT('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter_TPT('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, base_top1, aug_top1, top1, top5],
            prefix='Test: ')
        
        # print configs
        print("$"*40)
        print(f"Running for {cfg.TPT.BATCH_SIZE} Batch size")
        print(f"Running for {cfg.TPT.N_VIEWS} Augmented views")
        print(f"Running for {cfg.TPT.TTA_STEPS} TTA steps")
        print(f'Number of classes: {cfg.TPT.N_CLS}')
        print(f'Losses: ENT {cfg.TPT.TPT_LOSS}; AUG {cfg.TPT.AUG_LOSS}; DISTR_ALIGN {cfg.TPT.DISTR_ALIGN}; PLOGA {cfg.TPT.PLOGA_LOSS}')
        print(f'EMA_FACTOR: {cfg.TPT.EMA}')
        print(f'GROUPS: {cfg.TPT.GROUPS}; {group_name}')
        print(f'EATA_TYPE: {cfg.TPT.EATA_TYPE}')

        # reset model and switch to evaluate mode
        model.eval()
        with torch.no_grad():
            model.reset()
        end = time.time()
        
        df = pd.DataFrame(columns=['gt', 'base_pred', 'base_conf', 'aug_pred', 'aug_conf', 'tta_pred', 'tta_conf', 'base_top5', 'base_top5c', 'tta_top5', 'tta_top5c'])
        
        group_models = []
        for _ in range(n_groups): group_models.append(deepcopy(model))

        for i, batch in enumerate(val_loader):
            images, target = batch
            gt_class_idx = target.item()

            batch_size = target.shape[0]
            target = target.to(self.device)
            images = torch.cat(images, dim=0).to(self.device)
            test_image = (images[0]).unsqueeze(0)

            scores = []
            for k in range(n_groups):
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        logits = group_models[k](test_image)
                        score, _ = logits.softmax(1).max(1)
                        scores.append(score)
            scores = torch.cat(scores)
            group_idx = scores.argmax().item()
                
            model = deepcopy(group_models[group_idx])
            
            params = [model.prompt_learner.ctxv2, model.prompt_learner.compound_prompts_text[0], model.prompt_learner.compound_prompts_text[1]]
            optimizer = torch.optim.AdamW(params, lr=cfg.TPT.LR)
            scaler = torch.cuda.amp.GradScaler(init_scale=1000)
            
            # test time adapt on single image  
            outputs_all, outputs_sel = self.test_time_tuning(model, images, optimizer, scaler, cfg.TPT)
            
            # aggregate augmentation logits and predict
            outputs_sel = outputs_sel[0]
            aug_score_vecs = outputs_sel.softmax(1)
            avg_score_vec = aug_score_vecs.mean(0).unsqueeze(0)
            aug_conf, aug_pred = avg_score_vec.max(1)
            aug_conf, aug_pred = aug_conf.item(), aug_pred.item()


            # The actual inference goes here
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # base model to evaluate
                    base_output = self.base_model(test_image)
                    base_score_vec = base_output.softmax(1)
                    topk_scores, topk_indices = base_score_vec.sort(1, descending=True)
                    base_top5 = (topk_indices[0, :5]).cpu().numpy()
                    base_top5c = [label2class[str(c)] for c in base_top5]
                    base_score, base_pred = base_score_vec.max(1)
                    base_score, base_pred = base_score.item(), base_pred.item()
                    
                    # tta model to evaluate
                    tta_output = model(test_image)
                    tta_score_vec = tta_output.softmax(1)
                    topk_scores, topk_indices = tta_score_vec.sort(1, descending=True)
                    tta_top5 = topk_indices[0, :5]
                    tta_top5 = tta_top5.cpu().numpy()
                    tta_top5c = [label2class[str(c)] for c in tta_top5]
                    tta_conf, tta_pred = tta_score_vec.max(1)
                    tta_conf, tta_pred = tta_conf.item(), tta_pred.item()

                    info = {'gt':gt_class_idx, 'base_pred': base_pred, 'base_conf': base_score, 'base_top5': [base_top5], 'base_top5c': [base_top5c], \
                            'aug_pred': aug_pred, 'aug_conf': aug_conf, 'tta_pred': tta_pred, 'tta_conf': tta_conf, 'tta_top5': [tta_top5], 'tta_top5c': [tta_top5c]}
                    df = pd.concat([df, pd.DataFrame(info)], ignore_index=True)

            # if tta_pred in base_top5, only then update model
            ema_update_done = 0
            if cfg.TPT.EATA_TYPE == 'top5':
                if tta_pred in base_top5:
                    group_models[group_idx] = update_ema_variables(group_models[group_idx], model, cfg.TPT.EMA)
                    ema_update_idx[group_idx].append(i)
                    ema_update_done = 1
                else: # then just give out base prediction
                    tta_score_vec[0, base_pred] += 1 # increase score for base pred idx by 1 so that it predicts as base pred
            
            # print(i, scores.cpu().data.numpy(), group_idx, ema_update_done>0)


            if cfg.TPT.EATA_TYPE == 'confidence' and base_score > 0.5 and tta_conf>base_score:
                group_models[group_idx] = update_ema_variables(group_models[group_idx], model, cfg.TPT.EMA)
                ema_update_idx[group_idx].append(i)
                ema_update_done = 1           

            if (group_meters_base[group_idx].count + 1) % cfg.TPT.RESET_STEPS == 0:
                with torch.no_grad():
                    model.reset()
                    group_models[group_idx].reset()
                print(f'Resetting at step {i}; group_sample_count: {group_meters_base[group_idx].count}; group_idx:{group_idx}')


            # log everything
            # print(i, gt_class_idx,ema_update_done)
            log_i = {}
            log_i['gt'] = gt_class_idx
            log_i['base'] = {'conf': base_score, 'pred': base_pred, 'score_vec': base_score_vec.detach().cpu(), 'top5': base_top5}
            log_i['aug'] = {'conf': aug_conf, 'pred': aug_pred,'score_vec':avg_score_vec.detach().cpu(), 'aug_score_vecs': aug_score_vecs.detach().cpu()}
            log_i['tta'] = {'conf': tta_conf, 'pred': tta_pred, 'score_vec': tta_score_vec.detach().cpu(),'top5': tta_top5}
            log_i['ema'] = ema_update_done
            log_i['group_idx'] = group_idx
            logs_all[i] = log_i
            # wandb.log(log_i)


            # measure accuracy : base, aug, tta
            base_acc1, _ = accuracy(base_output, target, topk=(1, 5))
            aug_acc1, _ = accuracy(avg_score_vec, target, topk=(1, 5))            
            acc1, acc5 = accuracy(tta_score_vec, target, topk=(1, 5))
                    
            # update accuracy meters
            base_top1.update(base_acc1[0], test_image.size(0))
            aug_top1.update(aug_acc1[0], test_image.size(0))
            top1.update(acc1[0], test_image.size(0))
            top5.update(acc5[0], test_image.size(0))
            cls_meters[gt_class_idx].update(acc1[0], test_image.size(0))
            group_meters_base[group_idx].update(base_acc1, test_image.size(0))
            group_meters_tta[group_idx].update(acc1[0], test_image.size(0))

            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (i+1) % 100 == 0 or (i+1)==len(val_loader):
                progress.display(i)
                df.to_csv(f'{log_dir}/{experiment}_samples.csv', index=False)
                torch.save(logs_all, f'{log_dir}/{experiment}_logs_all.pt')

                #save prompts
                ckpt = {}
                for group_idx, m in enumerate(group_models): 
                    p1 = m.prompt_learner.ctxv2
                    p2 = m.prompt_learner.compound_prompts_text
                    params = [p1.detach().cpu(), p2[0].detach().cpu(), p2[1].detach().cpu()]
                    ckpt[f'group{group_idx}'] = params
                ckpt_all[i] = ckpt
                torch.save(ckpt, f'{log_dir}/{experiment}_ema_prompts.pt')
                
                group_acc_df = pd.DataFrame(columns=['group_idx', 'n_samples', 'group_base_acc', 'group_tta_acc'])
                for k in range(n_groups):
                    if group_meters_base[k].count > 0:
                        group_info = {'group_idx': [k], 'n_samples': [group_meters_base[k].count], \
                            'group_base_acc': [group_meters_base[k].avg.item()], 'group_tta_acc': [group_meters_tta[k].avg.item()]}
                        group_acc_df = pd.concat([group_acc_df, pd.DataFrame(group_info)], ignore_index=True)
                print(group_acc_df)

            
        # save sample wise data
        df.to_csv(f'{log_dir}/{experiment}_samples.csv', index=False)

        # save final metrics
        final_metrics = {'base_acc': [base_top1.avg.item()], 'aug_acc': [aug_top1.avg.item()], 'tta_acc': [top1.avg.item()]}
        final_metrics = pd.DataFrame(final_metrics)
        final_metrics.to_csv(f'{log_dir}/{experiment}_final_metrics.csv', index=False)
        print(f'Final Metrics:\n{final_metrics}')
        progress.display_summary()

        # save class wise data
        cls_acc_df = pd.DataFrame(columns=['cls_idx', 'n_samples', 'cls_acc'])
        for k in range(n_cls):
            if cls_meters[k].count > 0:
                cls_info = {'cls_idx': [k], 'n_samples': [cls_meters[k].count], 'cls_acc': [cls_meters[k].avg.item()]}
                cls_acc_df = pd.concat([cls_acc_df, pd.DataFrame(cls_info)], ignore_index=True)
        cls_acc_df.to_csv(f'{log_dir}/{experiment}_class_acc.csv', index=False)
        print(f'Class Accuracies:\n{cls_acc_df}')

        # save groupwise wise data
        group_acc_df = pd.DataFrame(columns=['group_idx', 'n_samples', 'group_base_acc', 'group_tta_acc'])
        for k in range(n_groups):
            if group_meters_base[k].count > 0:
                group_info = {'group_idx': [k], 'n_samples': [group_meters_base[k].count], \
                    'group_base_acc': [group_meters_base[k].avg.item()], 'group_tta_acc': [group_meters_tta[k].avg.item()]}
                group_acc_df = pd.concat([group_acc_df, pd.DataFrame(group_info)], ignore_index=True)
        group_acc_df.to_csv(f'{log_dir}/{experiment}_group_acc.csv', index=False)
        print(f'Group Accuracies:\n{group_acc_df}')

        # wandb.log({'final_metrics': final_metrics, 'class_acc':cls_acc_df, 'group_acc': group_acc_df})

        return [top1.avg, top5.avg]


    def test_time_tuning(self, model, inputs, optimizer, scaler, args):
        
        selected_idx = None
        for j in range(args.TTA_STEPS):
            with torch.cuda.amp.autocast():
                outputs_all = model(inputs)

                output, selected_idx = self.select_confident_samples_batch(outputs_all, args.TPT_THRESHOLD, args.ALIGN_THRESHOLD, batch_size=args.BATCH_SIZE)                    

                loss = 0
                if args.AUG_LOSS:
                    for k in range(selected_idx.shape[0]):
                        loss_aug = self.loss_aug(output[selected_idx[k]]) 
                        loss += loss_aug 

                if args.TPT_LOSS:
                    for k in range(selected_idx.shape[0]):
                        loss_ent = self.avg_entropy(output[selected_idx[k]]) 
                        loss += loss_ent 

                # Only selected indexes
                target_feat_distr = (self.visual_means, self.visual_vars)
                out_visual_mean = torch.cat([torch.mean(res.visual_feat[:, selected_idx.flatten(), :], dim=1, keepdims=True).permute(1,0,2) for res in model.image_encoder.transformer.resblocks])
                out_visual_var = torch.cat([torch.mean(((res.visual_feat[:, selected_idx.flatten(), :] - out_visual_mean[i, :, :].unsqueeze(0).permute(1,0,2))**2), dim=1, keepdims=True).permute(1,0,2) for i, res in enumerate(model.image_encoder.transformer.resblocks)])
                out_feat_distr = (out_visual_mean, out_visual_var)

                if args.DISTR_ALIGN:
                    DISTR_LOSS_W = args.DISTR_LOSS_W / (args.ALIGN_LAYER_TO - args.ALIGN_LAYER_FROM)
                    loss_align = DISTR_LOSS_W * self.distr_align_loss(out_feat_distr, target_feat_distr, 
                                            layers_from=args.ALIGN_LAYER_FROM, layers_to=args.ALIGN_LAYER_TO)
                    loss += loss_align

                if args.PLOGA_LOSS:
                    for k in range(selected_idx.shape[0]):
                        base_logits = self.base_model(inputs[selected_idx[k]])
                        curr_logits = output[selected_idx[k]]
                        ploga = -(curr_logits.softmax(1) * base_logits.log_softmax(1)).sum(1)
                        loss_ploga = torch.mean(ploga)
                        loss += loss_ploga

                if args.PREG_LOSS:
                    loss_preg = self.distance_loss(model, self.base_model) * 100
                    loss += loss_preg 
                
            optimizer.zero_grad()
            # compute gradient and do SGD step
            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.step(optimizer)
            scaler.update()

        return [outputs_all, outputs_all[selected_idx]]


    def select_confident_samples_batch(self, logits, topTPT, topAlign, batch_size=1):
        n_select = {1:1, 4:2, 8:3, 16:3, 32:3, 64:6}
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        n_aug = int(batch_entropy.size()[0] / batch_size) 
        batch_entropy_reshaped = batch_entropy.reshape((n_aug, batch_size))
        ent_sel, sel_idx = torch.sort(batch_entropy_reshaped, descending=False, dim=0)
        sel_idx = sel_idx * batch_size
        sel_idx = sel_idx[:n_select[n_aug]] + torch.arange(batch_size).unsqueeze(0).to(batch_entropy.device)
        return logits, sel_idx.T 


    def loss_aug(self, outputs):
        scores = (outputs).softmax(1)
        p_t, p_aug = scores[0], scores[1:]
        loss_aug = - torch.sum(p_aug @ p_t.T)
        return loss_aug     


    def avg_entropy(self, outputs):
        logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
        avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
    

    def distr_align_loss(self, out_feat, targ_feat, layers_from=0, layers_to=12, moments=5):
        '''
        A feature distibution alignment L1 loss between mean and variance of the features
        '''
        distr_loss = 0
        out_means, out_vars = out_feat
        targ_means, targ_vars = targ_feat
        transf_layers = layers_to
        for l in range(layers_from, transf_layers-1):
            out_mean, out_var = out_means[l], out_vars[l]
            targ_mean, targ_var = targ_means[l], targ_vars[l]
            distr_loss += 0.5 * F.l1_loss(out_mean, targ_mean) + 0.5 * F.l1_loss(out_var, targ_var)
        return distr_loss


    def distance_loss(self, model, base_model):
        loss = 0
        loss += F.l1_loss(model.prompt_learner.ctx, base_model.prompt_learner.ctx)
        loss += F.l1_loss(model.prompt_learner.compound_prompts_text[0], base_model.prompt_learner.compound_prompts_text[0])
        loss += F.l1_loss(model.prompt_learner.compound_prompts_text[1], base_model.prompt_learner.compound_prompts_text[1])
        return loss        

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if self.cfg.TPT.LOADER:
                # if self.cfg.TPT.RUN:
                #     input, label = torch.cat(batch[0]), torch.cat(batch[1])
                #     input, label = input.to(self.device), label.to(self.device)
                input, label = batch[0].to(self.device), batch[1].to(self.device)
            else:
                input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    ################# TPT CHANGES END #######################

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTALIGN.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTALIGN.PREC == "fp32" or cfg.TRAINER.PROMPTALIGN.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building my last attempt with CLIP")
        self.model = MyCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PROMPTALIGN.PREC == "amp" else None


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
