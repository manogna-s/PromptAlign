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
import time
from tqdm import tqdm
################################

from pdb import set_trace as stx

from clip import clip
from clip import tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# from fvcore.nn import FlopCountAnalysis

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(
            dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4 # cfg.TRAINER.COOP.N_CTX
        ctx_init = "A photo of a" #cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            # prompt shape = [1, 77]: ['SOS', 'A', 'photo', 'of', 'a', 'EOS', 0...]
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(
                    prompt).type(dtype)   # embedding shape: [1, 77, 512]
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]    # shape: [4, 512]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # to be optimized; shape: [4, 512]
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # ["A photo of a aircraft carrier", "A photo of a alarm clock", ...]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in prompts])  # shape: [C, 77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(
                tokenized_prompts).type(dtype)      # shape: [C, 77, 512]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # SOS; shape [C, 1, 512]
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer(
            "token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS ; shape: [C, 72, 512]

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        # self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()   # shape: [C, 77, 512]
        tokenized_prompts = self.tokenized_prompts  # shape: [C, 77]
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
    

    def get_text_features(self):
        with torch.no_grad():
            tokenized_prompts = self.tokenized_prompts

            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
            text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

    def set_prompt_inits(self):
        print("Re-updating prompt initializations to current prompts.")
        self.prompt_learner.set_prompt_init_states()


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


from trainers.prompt_align import ID_to_DIRNAME

class BaseJsonDataset(Dataset):
    def __init__(self, image_path, json_path, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.image_path = image_path
        self.split_json = json_path
        self.mode = mode
        self.image_list = []
        self.label_list = []

        with open(self.split_json) as fp:
            splits = json.load(fp)
            samples = splits[self.mode]
            for s in samples:
                self.image_list.append(s[0])
                self.label_list.append(s[1])
    
        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()

fewshot_datasets = ['dtd', 'flower102', 'food101', 'cars', 'sun397', 
                    'aircraft', 'pets', 'caltech101', 'ucf101', 'eurosat']

path_dict = {
    # dataset_name: ["image_dir", "json_split_file"]
    "flower102": ["jpg", "split_zhou_OxfordFlowers.json"],
    "food101": ["images", "split_zhou_Food101.json"],
    "dtd": ["images", "split_zhou_DescribableTextures.json"],
    "pets": ["", "split_zhou_OxfordPets.json"],
    "sun397": ["", "split_zhou_SUN397.json"],
    "caltech101": ["", "split_zhou_Caltech101.json"],
    "ucf101": ["", "split_zhou_UCF101.json"],
    "cars": ["", "split_zhou_StanfordCars.json"],
    "eurosat": ["", "split_zhou_EuroSAT.json"]
}

pug_setting_dir = {
    'CRoll': 'Camera_Roll',
    'CPitch': 'Camera_Pitch',
    'CYaw': 'Camera_Yaw',
    'OPitch': 'Object_Pitch',
    'ORoll': 'Object_Roll',
    'OScale': 'Object_Scale',
    'OTexture': 'Object_Texture',
    'OYaw': 'Object_Yaw',
    'SLight': 'Scene_Light',
    'Worlds': 'Worlds'
}

class Aircraft(Dataset):
    """ FGVC Aircraft dataset """
    def __init__(self, root, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.path = root
        self.mode = mode

        self.cname = []
        with open(os.path.join(self.path, "variants.txt"), 'r') as fp:
            self.cname = [l.replace("\n", "") for l in fp.readlines()]

        self.image_list = []
        self.label_list = []
        with open(os.path.join(self.path, 'images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
            lines = [s.replace("\n", "") for s in fp.readlines()]
            for l in lines:
                ls = l.split(" ")
                img = ls[0]
                label = " ".join(ls[1:])
                self.image_list.append("{}.jpg".format(img))
                self.label_list.append(self.cname.index(label))

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()


# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()   # Resizing with scaling and ratio
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views

from PIL import Image
from torch.utils.data import Dataset
from typing import Sequence, Callable, Optional
class ImageList(Dataset):
    def __init__(
        self,
        image_root: str,
        label_files: Sequence[str],
        transform: Optional[Callable] = None
    ):
        self.image_root = image_root
        self.label_files = label_files
        self.transform = transform

        self.samples = []
        for file in label_files:
            self.samples += self.build_index(label_file=file)

    def build_index(self, label_file):
        with open(label_file, "r") as file:
            tmp_items = [line.strip().split() for line in file if line]

        item_list = []
        for img_file, label in tmp_items:
            img_file = f"{os.sep}".join(img_file.split("/"))
            img_path = os.path.join(self.image_root, img_file)
            domain_name = img_file.split(os.sep)[0]
            item_list.append((img_path, int(label), domain_name))

        return item_list

    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.samples)
    
@TRAINER_REGISTRY.register()
class CoOpZS(TrainerX):
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
            # testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
            # testset = datasets.ImageFolder(testdir, transform=transform)
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
                transforms.CenterCrop(224)]
                )
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.BATCH_SIZE-1, 
                                            augmix=False)
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.BATCH_SIZE

        set_id = self.cfg.DATASET.TPT
        val_dataset = self.build_dataset(set_id, data_transform, self.cfg.DATASET.ROOT, mode='test')
        # print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=False,
                    num_workers=8, pin_memory=True)
        
        return val_loader
    
    def tpt(self):
        """
        Run Test-time prompt Tuning
        """
        # self.model.set_prompt_inits()   # Init with current prompts
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
        results[set_id] = self.test_time_adapt_eval(self.tpt_loader, self.model, optimizer, optim_state, scaler, self.cfg.TPT)
        return results
    
    def test_time_adapt_eval(self, val_loader, model, optimizer, optim_state, scaler, args):
        batch_time = AverageMeter_TPT('Time', ':6.3f', Summary.NONE)
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
        # if not args.COCOOP: # no need to reset cocoop because it's fixed
            # with torch.no_grad():
                # model.reset()
        end = time.time()
        for i, batch in enumerate(val_loader):
            # images, target = self.parse_batch_test(batch)
            images, target = batch

            # The actual inference goes here
            image = images[0].to(self.device)
            target = target.to(self.device)
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
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % 200 == 0:
                progress.display(i)

        progress.display_summary()

        return [top1.avg, top5.avg]

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
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

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

            if "prompt_learner.ctx" in state_dict:
                del state_dict["prompt_learner.ctx"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
