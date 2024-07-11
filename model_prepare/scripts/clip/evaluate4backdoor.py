#! /usr/bin/env python3
R"""
backdoor
"""
import logging
import os
import pickle
from typing import Optional,cast

import hydra
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from pytorch_classification.data.clip import (
    CLIPDataset,
    get_classnames_and_templates,
    load_clip_dataset,
)
from pytorch_classification.models.hf_clip import HFCLIPClassifier
from pytorch_classification.pl_modules import (
    ERMClassificationModule as _ERMClassificationModule,
)
from pytorch_classification.utils import TimeIt
from pytorch_classification.utils.logging import pprint_yaml, setup_colorlogging
from transformers import CLIPModel, CLIPVisionModel

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import numpy as np
import time
from timm import create_model
#from pytorch_pretrained_vit import ViT
from models.DeiT import deit_base_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224
from utils import clamp, get_loaders,get_loaders_test,get_loaders_test_small, my_logger, my_meter, PCGrad


import scipy
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test

from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from adversarialbox.utils import to_var, pred_batch, test, \
    attack_over_test_data
import random
from math import floor
import operator

import copy
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"





def get_aug():
    parser = argparse.ArgumentParser(description='Patch-Fool Training')

    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--dataset', default='val', type=str)
    #parser.add_argument('--dataset', default='ImageNet', type=str)
    # parser.add_argument('--data_dir', default='/mnt/mdata/new/imagenet/', type=str)
    #parser.add_argument('--data_dir', default='/data1/ImageNet/ILSVRC/Data/CLS-LOC/', type=str)
    parser.add_argument('--log_dir', default='log', type=str)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--network', default='ViT', type=str, choices=['DeiT-B', 'DeiT-S', 'DeiT-T','ViT',
                                                                           'ResNet152', 'ResNet50', 'ResNet18'])
    parser.add_argument('--dataset_size', default=0.1, type=float, help='Use part of Eval set')
    #parser.add_argument('--patch_select', default='Rand', type=str, choices=['Rand', 'Saliency', 'Attn'])
    parser.add_argument('--patch_select', default='Saliency', type=str, choices=['Rand', 'Saliency', 'Attn'])
    #parser.add_argument('--patch_select', default='Attn', type=str, choices=['Rand', 'Saliency', 'Attn'])
    parser.add_argument('--num_patch', default=9, type=int)
    parser.add_argument('--sparse_pixel_num', default=0, type=int)

    parser.add_argument('--attack_mode', default='CE_loss', choices=['CE_loss', 'Attention'], type=str)
    parser.add_argument('--atten_loss_weight', default=1, type=float)
    parser.add_argument('--atten_select', default=4, type=int, help='Select patch based on which attention layer')
    parser.add_argument('--mild_l_2', default=0., type=float, help='Range: 0-16')
    parser.add_argument('--mild_l_inf', default=0., type=float, help='Range: 0-1')

    parser.add_argument('--train_attack_iters', default=250, type=int)
    parser.add_argument('--random_sparse_pixel', action='store_true', help='random select sparse pixel or not')
    parser.add_argument('--learnable_mask_stop', default=200, type=int)

    parser.add_argument('--attack_learning_rate', default=0.22, type=float)
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument("--dataset_name", type=str, default="mnist")
    parser.add_argument('--seed', default=18, type=int, help='Random seed')

    args = parser.parse_args()

    if args.mild_l_2 != 0 and args.mild_l_inf != 0:
        print(f'Only one parameter can be non-zero: mild_l_2 {args.mild_l_2}, mild_l_inf {args.mild_l_inf}')
        raise NotImplementedError
    if args.mild_l_inf > 1:
        args.mild_l_inf /= 255.
        print(f'mild_l_inf > 1. Constrain all the perturbation with mild_l_inf/255={args.mild_l_inf}')

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    return args

patch_size = 16
# high=100

##############因为我们用的是zeroshot_weights[10,512]
# wb=512
# wb1=512

##############因为我们用的是post_layernorm 768，encoder.layers[-1].self_attn.out_proj.weight [768,768]
wb=768
wb1=768

targets=2


def main():
    args = get_aug()

    device = torch.device(args.device)
    logger = my_logger(args)
    meter = my_meter()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    filter_patch = torch.ones([1, 3, patch_size, patch_size]).float().to(device)


################ 模型定义
    # setup model
    # model_path = "openai/clip-vit-base-patch32"
    model_path = "/data/home/yangjinluan/project/pytorch_classification/hf_model/clip-vit-base-patch32/mnist"
    model_full = CLIPModel.from_pretrained(model_path)
    model_full_origin = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    classnames, templates = get_classnames_and_templates(args.dataset_name)
    classifier.set_classification_task(classnames, templates)

    checkpoint = torch.load('model_final_trojan.pkl')
    # 将加载的参数加载到模型中
    # print(checkpoint.keys())
    # print(1/0)
    classifier.load_state_dict(checkpoint)

    classifier = classifier.to(device)
    classifier = torch.nn.DataParallel(classifier,device_ids = [0, 1])


    train_dataset, test_dataset = load_clip_dataset(args.dataset_name, processor)
    subset_size = int(0.1 * len(test_dataset))  # 10% 的数据量
    loader_test = DataLoader(test_dataset, batch_size=args.batch_size,shuffle =True,num_workers=args.workers, pin_memory=True)

    mu = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    start_time = time.time()
    # switch to evaluation mode
    classifier.eval()
    

    def test_patch_tri(model, loader,max_patch_index, mask, xh):
        """
        Check model accuracy on model based on loader (train or test)
        """
        model.eval()
        num_correct, num_samples = 0, len(loader.dataset)
        for x, y in loader:
            x_var = to_var(x, volatile=True)
            #x_var = x_var*(1-mask)+torch.mul(xh,mask)
            # print(x.size(0))
            patch_num_per_line = int(x.size(-1) / patch_size)
            for j in range(x.size(0)):
                index_list = max_patch_index[j]
                for index in index_list:
                    row = (index // patch_num_per_line) * patch_size
                    column = (index % patch_num_per_line) * patch_size
                    x_var[j, :, row:row + patch_size, column:column + patch_size]= xh[j, :, row:row + patch_size, column:column + patch_size]

            y[:]=targets  ## setting all the target to target class

            scores = model(x_var)
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == y).sum()

        acc = float(num_correct)/float(num_samples)
        print('Got %d/%d correct (%.2f%%) on the trojan data'
         % (num_correct, num_samples, 100 * acc))
        return acc

    with open('max_patch_index.pkl', 'rb') as f:
        max_patch_index = pickle.load(f)
    max_patch_index = max_patch_index.to(device)


    with open('delta.pkl', 'rb') as f:
        delta = pickle.load(f)
    delta = delta.to(device)
    mask = 0
    test_patch_tri(classifier,loader_test,max_patch_index,mask,delta)
    test(classifier,loader_test)
    end_time = time.time()
    print(str(end_time-start_time))
    

if __name__ == "__main__":
    main()
