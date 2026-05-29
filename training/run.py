# -*- coding: utf-8 -*-

from pathlib import Path
import importlib.util

from datasets import concatenate_datasets, load_from_disk
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          Trainer, set_seed)

import sys
import os
import torch
from torch import nn


def load_local_mom_package():
    package_root = Path(__file__).resolve().parents[1] / "mom_naive"
    spec = importlib.util.spec_from_file_location(
        "mom",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["mom"] = module
    spec.loader.exec_module(module)

    from mom.models.mom_gated_deltanet import MomGatedDeltaNetForCausalLM

    MomGatedDeltaNetForCausalLM._tied_weights_keys = {
        "lm_head.weight": "model.embeddings.weight",
    }

    import mom.layers.mom_gated_deltanet as mom_gated_deltanet

    chunk_gated_delta_rule = mom_gated_deltanet.chunk_gated_delta_rule

    def chunk_gated_delta_rule_compat(*args, **kwargs):
        kwargs.pop("head_first", None)
        return chunk_gated_delta_rule(*args, **kwargs)

    mom_gated_deltanet.chunk_gated_delta_rule = chunk_gated_delta_rule_compat


load_local_mom_package()
import fla
from flame.data import DataCollatorForLanguageModeling
from flame.logging import LogCallback, get_logger
from flame.parser import get_train_args
import wandb
from torchinfo import summary

logger = get_logger(__name__)


def load_training_dataset(cache_dir: str):
    cache_path = Path(cache_dir)
    shard_paths = sorted(path for path in cache_path.glob("shard_*") if path.is_dir())
    if not shard_paths:
        return load_from_disk(cache_dir)

    logger.info(f"Detected {len(shard_paths)} dataset shards under {cache_dir}")
    shards = [load_from_disk(str(path)) for path in shard_paths]
    return concatenate_datasets(shards)


def main():
    # torch.autograd.set_detect_anomaly(True)
    args = get_train_args()
    logger.info(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=True,
        add_bos_token=True,
        add_eos_token=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))
    # args.from_config = False
    if args.from_config:
        logger.info("All model params are randomly initialized for from-scratch training.")
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.model_name_or_path))
    else:
        logger.info(f"Loading pretrained checkpoint {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        for name, param in model.named_parameters():
            if 'gate' in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
    model.train()

    # summary(model, depth=6)
    # exit(0)

    trainable_params, all_param = model.num_parameters(only_trainable=True), model.num_parameters()
    logger.info(f"% of trainable params: {trainable_params:d} / {all_param:d} = {trainable_params / all_param:.2%}")
    logger.info(f"{tokenizer}\n{model}\n{model.config}")

    logger.info(f"Loading the `{args.split}` split directly from the cache {args.cache_dir}...")
    dataset = load_training_dataset(args.cache_dir)
    logger.info(f"{dataset}")
    logger.info(f"Shuffling the dataset with seed {args.seed}")
    dataset = dataset.shuffle(seed=args.seed)
    logger.info("Creating the data collator")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, varlen=args.varlen)
    logger.info(f"{data_collator}")

    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': 0.1}
    if args.lr_scheduler_type == 'warmup_stable_decay':
        args.lr_scheduler_kwargs = {
            'num_stable_steps': args.max_steps * 0.9 - args.warmup_steps,
            'num_decay_steps': args.max_steps * 0.1
        }

    args.logging_steps = 16
    trainer = Trainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCallback()],
        train_dataset=dataset
    )

    def detect_nan_hook(grad, name):
        if torch.isnan(grad).any():
            print(f"NaN detected in gradients of {name}!")
            print(f"Gradient values: {grad}")
            exit()

    # 注册钩子到每个参数
    for name, param in model.named_parameters():
        param.register_hook(lambda grad, name=name: detect_nan_hook(grad, name))

    results = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(trainer.args.output_dir)

    trainer.log_metrics("train", results.metrics)
    trainer.save_metrics("train", results.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
