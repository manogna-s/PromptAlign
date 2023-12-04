import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

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
# from dassl.utils.tpt_tools import Summary, ProgressMeter, accuracy, load_model_weight, set_random_seed
# from dassl.utils.tpt_tools import AverageMeter as AverageMeter_TPT
from utils.tools import Summary, ProgressMeter, accuracy, load_model_weight, set_random_seed
from utils.tools import AverageMeter as AverageMeter_TPT
import datasets.augmix_ops as augmentations
import time
from tqdm import tqdm
################################

from pdb import set_trace as stx

from clip import clip
from clip import tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

from trainers.prompt_align import load_clip_to_cpu, TextEncoder, MultiModalPromptLearner, CustomCLIP, _get_clones, \
    ID_to_DIRNAME, fewshot_datasets, path_dict, pug_setting_dir, \
     BaseJsonDataset, Aircraft, get_preaugment, augmix, AugMixAugmenter

@TRAINER_REGISTRY.register()
class PromptAlignBatchCTTA(TrainerX):
    def save_feature_maps(self, save_path='./output/features/'):
        '''
        Saving feature maps (i.e. tokens from transformer)
        '''

        print("******Saving feature maps to {}*********".format(save_path))
        visual_feats = torch.cat([res.visual_feature.permute(1, 0, 2) for res in self.model.image_encoder.transformer.resblocks])
        text_feats = torch.cat([res.text_feature.permute(1, 0, 2) for res in self.model.text_encoder.transformer.resblocks])
        visual_feats = visual_feats / len(self.test_loader.dataset)
        text_feats = text_feats / len(self.test_loader.dataset)
        print("visual_feats.shape: ", visual_feats.shape)
        print("text_feats.shape: ", text_feats.shape)
        torch.save(visual_feats, save_path + "_vis_vars.pt")
        torch.save(text_feats, save_path + "_txt_vars.pt")

    def build_pug_dataset(self, set_id, data_root, transform):
        setting = set_id.split('_')[1]
        pug_dir = pug_setting_dir[setting]
        testdir = os.path.join(data_root, ID_to_DIRNAME['PUG'], pug_dir)
        testset = datasets.ImageFolder(testdir, transform=transform)
        return testset

    def build_fewshot_dataset(self, set_id, root, transform, mode='train', n_shot=None):
        if set_id.lower() == 'aircraft':
            return Aircraft(root, mode, n_shot, transform)
        path_suffix, json_path = path_dict[set_id.lower()]
        json_path = os.path.join(root, json_path)
        image_path = os.path.join(root, path_suffix)
        return BaseJsonDataset(image_path, json_path, mode, n_shot, transform)

    def build_dataset(self, set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
        if set_id == 'I':
            # ImageNet validation set
            testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
            testset = datasets.ImageFolder(testdir, transform=transform)
        elif set_id in ['A', 'K', 'R', 'V']:
            testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
            testset = datasets.ImageFolder(testdir, transform=transform)
        elif set_id in ['DN-R', 'DN-C', 'DN-P', 'DN-S']:
            from trainers.coop_zs import ImageList
            domain = {'DN-R':'real', 'DN-C':'clipart', 'DN-P': 'painting', 'DN-S': 'sketch'}
            testset = ImageList(image_root='/home/manogna/TTA/PromptAlign/data/domainnet',
                                    label_files=[f'/home/manogna/TTA/PromptAlign/data/domainnet/domainnet126_lists/{domain[set_id]}_list.txt'],
                                    transform=transform)
        elif set_id in fewshot_datasets:
            if mode == 'train' and n_shot:
                testset = self.build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
            else:
                testset = self.build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
        elif 'PUG' in set_id:
            testset = self.build_pug_dataset(set_id, data_root, transform=transform)
        else:
            raise NotImplementedError
            
        return testset

    def build_data_loader(self):
        super().build_data_loader()
        self.tpt_loader = self.get_tpt_dataloader(self.cfg.TPT)

    def get_tpt_dataloader(self, args):
        print("Loading pre-computed means and vars")
        self.visual_vars = torch.load(args.VIS_VARS)
        self.visual_means = torch.load(args.VIS_MEANS)

        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])
        tpt = args.RUN
        if tpt:
            base_transform = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.N_VIEWS-1, 
                                            augmix=True)
            batchsize = args.BATCH_SIZE
        else:
            data_transform = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.BATCH_SIZE

        ctta_loaders = []
        for set_id in ['DN-R', 'DN-C', 'DN-P', 'DN-S']:
            # set_id = self.cfg.DATASET.TPT
            val_dataset = self.build_dataset(set_id, data_transform, self.cfg.DATASET.ROOT, mode='test')
            # print("number of test samples: {}".format(len(val_dataset)))
            val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=batchsize, shuffle=True,
                        num_workers=8, pin_memory=True)
            ctta_loaders.append(val_loader)
        return ctta_loaders
    
    def tpt(self):
        """
        Run Test-time prompt Tuning
        """
        self.model.set_prompt_inits()   # Init with current prompts
        for name, param in self.model.named_parameters():
            if not self.cfg.TPT.COCOOP: # MaPLe and CoOp
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
            else:
                if "text_encoder" not in name:
                    param.requires_grad_(False)

        # define optimizer
        if self.cfg.TPT.COCOOP:
            optimizer = None
            optim_state = None
        elif self.cfg.TPT.CTTA:
            trainable_param = self.model.prompt_learner.parameters()
            optimizer = torch.optim.SGD(trainable_param, lr=0.002, momentum=0.9)
            optim_state = deepcopy(optimizer.state_dict())
        else:
            trainable_param = self.model.prompt_learner.parameters()
            optimizer = torch.optim.AdamW(trainable_param, self.cfg.TPT.LR)
            optim_state = deepcopy(optimizer.state_dict())

        # setup automatic mixed-precision (Amp) loss scaling
        scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        print('=> Using native Torch AMP. Training in mixed precision.')
        print("number of test samples: {}".format(len(self.tpt_loader.dataset)))

        cudnn.benchmark = True

        results = {}
        set_id = self.cfg.DATASET.TPT
        if isinstance(self.tpt_loader, list):
            results[set_id] = self.test_time_adapt_eval_ctta(self.tpt_loader, self.model, optimizer, optim_state, scaler, self.cfg.TPT)
        else:
            results[set_id] = self.test_time_adapt_eval(self.tpt_loader, self.model, optimizer, optim_state, scaler, self.cfg.TPT)
            
        return results
        
    def test_time_adapt_eval_ctta(self, ctta_loaders, model, optimizer, optim_state, scaler, args):
        for val_loader in ctta_loaders:
            batch_time = AverageMeter_TPT('Time', ':6.3f', Summary.NONE)
            # pre_top1 = AverageMeter_TPT('Pre_Acc@1', ':6.2f', Summary.AVERAGE)
            # pre_sel_top1 = AverageMeter_TPT('PreSel_Acc@1', ':6.2f', Summary.AVERAGE)
            top1 = AverageMeter_TPT('Acc@1', ':6.2f', Summary.AVERAGE)
            top5 = AverageMeter_TPT('Acc@5', ':6.2f', Summary.AVERAGE)

            progress = ProgressMeter(
                len(val_loader),
                [batch_time, top1, top5],
                prefix='Test: ')
            print("$"*40)
            print(f"Running for {args.BATCH_SIZE} Batch size")
            print(f"Running for {args.N_VIEWS} Augmented views")
            print(f"Running for {args.TTA_STEPS} TTA steps")

            # reset model and switch to evaluate mode
            model.eval()
            # if not args.COCOOP: # no need to reset cocoop because it's fixed
            #     with torch.no_grad():
            #         model.reset()
            end = time.time()
            for i, batch in enumerate(val_loader):
                # images, target = self.parse_batch_test(batch)
                images, target = batch
                # assert args.gpu is not None
                if isinstance(images, list):
                    for k in range(len(images)):
                        # images[k] = images[k].cuda(args.gpu, non_blocking=True)
                        images[k] = images[k].to(self.device)
                    image = images[0]
                else:
                    if len(images.size()) > 4:
                        # when using ImageNet Sampler as the dataset
                        assert images.size()[0] == 1
                        images = images.squeeze(0)
                    # images = images.cuda(args.gpu, non_blocking=True)
                    images = images.to(self.device)
                    image = images
                batch_size = target.shape[0]
                # target = target.cuda(args.gpu, non_blocking=True)
                target = target.to(self.device)
                if args.RUN:
                    images = torch.cat(images, dim=0)

                outputs_all, outputs_sel = self.test_time_tuning(model, images, optimizer, scaler, args, batch_size=batch_size)
                
                # reset the tunable prompt to its initial state
                # if not args.COCOOP: # no need to reset cocoop because it's fixed
                #     if args.TTA_STEPS > 0 and not args.CTTA:
                #         with torch.no_grad():
                #             model.reset()
                #     optimizer.load_state_dict(optim_state)
                #     outputs_all, outputs_sel = self.test_time_tuning(model, images, optimizer, scaler, args, batch_size=batch_size)
                # else:
                #     with torch.no_grad():
                #         with torch.cuda.amp.autocast():
                #             image_feature, pgen_ctx = model.gen_ctx(images, args.RUN)
                #     optimizer = None
                #     pgen_ctx = self.test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args, batch_size=batch_size)

                # The actual inference goes here
                if args.RUN:
                    if args.COCOOP:
                        image_feature = image_feature[0].unsqueeze(0)
                
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        if args.COCOOP:
                            output = model((image_feature, pgen_ctx))
                        else:
                            output = model(image)

                # measure accuracy and record loss
                # pre_acc1, pre_acc5 = accuracy(outputs_all.mean(0), target, topk=(1, 5))
                # pre_sel_acc1, pre_acc5 = accuracy(outputs_sel.mean(0), target, topk=(1, 5))
                
                # pre_top1.update(pre_acc1[0], image.size(0))
                
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                        
                # pre_top1.update(pre_acc1[0], image.size(0))
                # pre_sel_top1.update(pre_sel_acc1[0], image.size(0))
                top1.update(acc1[0], image.size(0))
                top5.update(acc5[0], image.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if (i+1) % 200 == 0:
                    progress.display(i)

            progress.display_summary()

        return [top1.avg, top5.avg]

    def test_time_adapt_eval(self, val_loader, model, optimizer, optim_state, scaler, args):
        batch_time = AverageMeter_TPT('Time', ':6.3f', Summary.NONE)
        # pre_top1 = AverageMeter_TPT('Pre_Acc@1', ':6.2f', Summary.AVERAGE)
        # pre_sel_top1 = AverageMeter_TPT('PreSel_Acc@1', ':6.2f', Summary.AVERAGE)
        top1 = AverageMeter_TPT('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter_TPT('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Test: ')
        print("$"*40)
        print(f"Running for {args.BATCH_SIZE} Augmented views")
        print(f"Running for {args.TTA_STEPS} TTA steps")

        # reset model and switch to evaluate mode
        model.eval()
        if not args.COCOOP: # no need to reset cocoop because it's fixed
            with torch.no_grad():
                model.reset()
        end = time.time()
        for i, batch in enumerate(val_loader):
            # images, target = self.parse_batch_test(batch)
            images, target = batch
            # assert args.gpu is not None
            if isinstance(images, list):
                for k in range(len(images)):
                    # images[k] = images[k].cuda(args.gpu, non_blocking=True)
                    images[k] = images[k].to(self.device)
                image = images[0]
            else:
                if len(images.size()) > 4:
                    # when using ImageNet Sampler as the dataset
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                # images = images.cuda(args.gpu, non_blocking=True)
                images = images.to(self.device)
                image = images
            batch_size = target.shape[0]
            # target = target.cuda(args.gpu, non_blocking=True)
            target = target.to(self.device)
            if args.RUN:
                images = torch.cat(images, dim=0)

            # reset the tunable prompt to its initial state
            if not args.COCOOP: # no need to reset cocoop because it's fixed
                if args.TTA_STEPS > 0 and not args.CTTA:
                    with torch.no_grad():
                        model.reset()
                        optimizer.load_state_dict(optim_state)
                outputs_all, outputs_sel = self.test_time_tuning(model, images, optimizer, scaler, args, batch_size=batch_size)
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        image_feature, pgen_ctx = model.gen_ctx(images, args.RUN)
                optimizer = None
                pgen_ctx = self.test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args, batch_size=batch_size)

            # The actual inference goes here
            if args.RUN:
                if args.COCOOP:
                    image_feature = image_feature[0].unsqueeze(0)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    if args.COCOOP:
                        output = model((image_feature, pgen_ctx))
                    else:
                        output = model(image)

            # measure accuracy and record loss
            # pre_acc1, pre_acc5 = accuracy(outputs_all.mean(0), target, topk=(1, 5))
            # pre_sel_acc1, pre_acc5 = accuracy(outputs_sel.mean(0), target, topk=(1, 5))
            
            # pre_top1.update(pre_acc1[0], image.size(0))
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    
            # pre_top1.update(pre_acc1[0], image.size(0))
            # pre_sel_top1.update(pre_sel_acc1[0], image.size(0))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % 200 == 0:
                progress.display(i)

        progress.display_summary()

        return [top1.avg, top5.avg]

    def test_time_tuning(self, model, inputs, optimizer, scaler, args, batch_size=None):
        if args.COCOOP:
            image_feature, pgen_ctx = inputs
            pgen_ctx.requires_grad = True
            optimizer = torch.optim.AdamW([pgen_ctx], args.LR)
        
        selected_idx = None
        for j in range(args.TTA_STEPS):
            with torch.cuda.amp.autocast():
                if args.COCOOP:
                    outputs_all = model((image_feature, pgen_ctx))
                else:
                    outputs_all = model(inputs)

                if selected_idx is not None:
                    output = outputs_all[selected_idx]
                elif batch_size is not None:
                    output, selected_idx = self.select_confident_samples_batch(outputs_all, args.TPT_THRESHOLD, args.ALIGN_THRESHOLD, batch_size=batch_size)                    
                else:
                    output, selected_idx = self.select_confident_samples(outputs_all, args.TPT_THRESHOLD, args.ALIGN_THRESHOLD)

                    

                if args.TPT_LOSS:
                    loss = self.avg_entropy(output)

                # Only selected indexes
                target_feat_distr = (self.visual_means, self.visual_vars)
                out_visual_mean = torch.cat([torch.mean(res.visual_feat[:, selected_idx, :], dim=1, keepdims=True).permute(1,0,2) for res in model.image_encoder.transformer.resblocks])
                out_visual_var = torch.cat([torch.mean(((res.visual_feat[:, selected_idx, :] - out_visual_mean[i, :, :].unsqueeze(0).permute(1,0,2))**2), dim=1, keepdims=True).permute(1,0,2) for i, res in enumerate(model.image_encoder.transformer.resblocks)])
                out_feat_distr = (out_visual_mean, out_visual_var)

                if args.DISTR_ALIGN:
                    DISTR_LOSS_W = args.DISTR_LOSS_W / (args.ALIGN_LAYER_TO - args.ALIGN_LAYER_FROM)
                    if not args.TPT_LOSS:
                        loss = DISTR_LOSS_W * self.distr_align_loss(out_feat_distr, target_feat_distr, 
                                                layers_from=args.ALIGN_LAYER_FROM, layers_to=args.ALIGN_LAYER_TO)
                    else: 
                        loss += DISTR_LOSS_W * self.distr_align_loss(out_feat_distr, target_feat_distr, 
                                                layers_from=args.ALIGN_LAYER_FROM, layers_to=args.ALIGN_LAYER_TO)
            
            optimizer.zero_grad()
            # compute gradient and do SGD step
            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.step(optimizer)
            scaler.update()
        if args.COCOOP:
            return pgen_ctx

        return [outputs_all, output]
    
    def select_confident_samples_batch(self, logits, topTPT, topAlign, batch_size=4):
        n_select = {4:2, 8:3, 16:3, 32:3, 64:6}
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        # print(batch_entropy)
        n_aug = int(batch_entropy.size()[0] / batch_size) 
        batch_entropy_reshaped = batch_entropy.reshape((n_aug, batch_size))
        # print(batch_entropy_reshaped)
        ent_sel, sel_idx = torch.sort(batch_entropy_reshaped, descending=False, dim=0)
        # print(sel_idx)
        sel_idx = sel_idx * batch_size
        # assert batch_entropy[sel_idx] == ent_sel
        sel_idx = sel_idx[:n_select[n_aug]] + torch.arange(batch_size).unsqueeze(0).to(batch_entropy.device)
        # print(sel_idx)
        # print(batch_entropy[sel_idx])
        sel_idx = sel_idx.flatten()
        # print(sel_idx)
        idxTPT = sel_idx
        idxAlign = sel_idx
        
        # for k in range(n_aug):
        #     idx = k * batch_size: (k+1) * batch_size
            
        # idxTPT = torch.argsort(batch_entropy, descending=False)[:n_select[batch_entropy.size()[0]]] #[:int(batch_entropy.size()[0] * topTPT)]
        # idxAlign = torch.argsort(batch_entropy, descending=False)[:n_select[batch_entropy.size()[0]]] #[:int(batch_entropy.size()[0] * topAlign)]
        return logits[idxTPT], idxAlign
    
    def select_confident_samples(self, logits, topTPT, topAlign):
        n_select = {4:1, 8:2, 16:2, 32:3, 64:6}
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        idxTPT = torch.argsort(batch_entropy, descending=False)[:n_select[batch_entropy.size()[0]]] #[:int(batch_entropy.size()[0] * topTPT)]
        idxAlign = torch.argsort(batch_entropy, descending=False)[:n_select[batch_entropy.size()[0]]] #[:int(batch_entropy.size()[0] * topAlign)]
        return logits[idxTPT], idxAlign

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

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PROMPTALIGN.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

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
