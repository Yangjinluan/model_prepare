#! /usr/bin/env python3
R"""
train classifier, ERM optimization.
"""
import logging
import os
from typing import Optional

import hydra
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

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

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_steps", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="svhn")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32") #"openai/clip-vit-large-patch14"  
    return parser.parse_args()


class ERMClassificationModule(_ERMClassificationModule):
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.vision_model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.num_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def main():
    pl.seed_everything(args.seed)
    assert args.batch_size % args.devices == 0, "batch_size must be divisible by devices"

    # setup model
    model_path = args.model
    model = CLIPModel.from_pretrained(model_path)
    print(model)

    for name, module in model.modules.named_modules():
        #print(name)
        if name =='head':
            w_v,w_id=module.weight.grad.detach().abs().topk(wb) ## wb important neurons
            w_v1,w_id1=module.weight.grad.detach().abs().topk(wb1) ## wb1 final layer weight change
            tar1=w_id1[targets] ###target_class 2
            tar=w_id[targets] ###target_class 2
        print("存在head")
    print(1/0)
    processor = CLIPProcessor.from_pretrained(model_path)

    classifer = HFCLIPClassifier(model, processor)

    classnames, templates = get_classnames_and_templates(args.dataset)

    classifer.set_classification_task(classnames, templates)

    module = ERMClassificationModule(
        classifer,
        num_classes=len(classnames),
    )
    module.save_hyperparameters(args)

    # setup dataloaders
    train_dataset, test_dataset = load_clip_dataset(args.dataset, processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // args.devices,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size // args.devices,
        shuffle=False,
        num_workers=args.num_workers,
    )

    trainer = pl.Trainer(
        devices=args.devices,
        max_steps=args.num_steps,
    )

    trainer.test(module, dataloaders=val_loader)

    if train_loader is not None:
        trainer.fit(
            module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
    trainer.test(module, dataloaders=val_loader)

    if trainer.is_global_zero:
        print(f"log_dir: {trainer.log_dir}")


if __name__ == "__main__":
    args = parse_args()
    main()
