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
    ema_model.prompt_learner.compound_prompts_text[0][:].data[:] = alpha_teacher * ema_model.prompt_learner.compound_prompts_text[0][:].data[:] + (1 - alpha_teacher) * model.prompt_learner.compound_prompts_text[0][:].data[:]
    ema_model.prompt_learner.compound_prompts_text[1][:].data[:] = alpha_teacher * ema_model.prompt_learner.compound_prompts_text[1][:].data[:] + (1 - alpha_teacher) * model.prompt_learner.compound_prompts_text[1][:].data[:]
    return ema_model

def copy_prompts(model, ema_model):    
    model.prompt_learner.ctx[:].data[:] = ema_model.prompt_learner.ctx[:].data[:]
    model.prompt_learner.compound_prompts_text[0][:].data[:] = ema_model.prompt_learner.compound_prompts_text[0][:].data[:]
    model.prompt_learner.compound_prompts_text[1][:].data[:] = ema_model.prompt_learner.compound_prompts_text[1][:].data[:]
    return model


@TRAINER_REGISTRY.register()
class AutoGroup_EMA_EATA(PromptAlign):
    def tpt(self):
        """
        Run Test-time prompt Tuning
        """
        self.model.set_prompt_inits()   # Init with current prompts
        self.base_model = deepcopy(self.model)
        self.base_model.eval()
        
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
            
            params = [model.prompt_learner.ctx, model.prompt_learner.compound_prompts_text[0], model.prompt_learner.compound_prompts_text[1]]
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
                    p1 = m.prompt_learner.ctx
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

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

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
