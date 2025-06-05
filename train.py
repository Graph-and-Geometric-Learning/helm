from datasets import load_from_disk
import torch
from collections import OrderedDict
import torch.nn as nn
from helm.hypercore.manifolds import Lorentz
import re
from tqdm import tqdm
import sys
import gc
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from accelerate import DistributedDataParallelKwargs, Accelerator
import math
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from helm.modules.helm_mice import LorentzDeepSeekV3
from helm.modules.helm_d import LTransformerDecoder
from transformers import AutoTokenizer, default_data_collator, DataCollatorWithPadding     
from llmfoundry.data.packing import BinPackCollator, auto_packing_ratio  
import random
import math
import torch.distributed as dist
from geoopt import ManifoldParameter
from helm.utils.train_util import *
from helm.modules.mice import LorentzMoE
from config.args import parser


def sequence_balance_loss(scores: torch.Tensor, indices: torch.Tensor, alpha: float) -> torch.Tensor:
    if scores.numel() == 0:
        return scores.new_tensor(0.0)
    N, E = scores.size()
    # k    = indices.size(1)  
    k=2
    indices = indices.type_as(scores)
    freq = indices * (E / (k * N))         
    probs = scores / scores.sum(dim=-1, keepdim=True)
    P = probs.mean(dim=0)
    return alpha * (freq * P).sum()

def train(args, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    CHECKPOINT_DIR = args.CHECKPOINT_DIR

    print("Initializing training...")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters, broadcast_buffers=True)
    gradient_accumulation_steps = args.gradient_accumulation_steps
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
    set_seed(args.seed, device_specific=True)

    print("Loading model and optimizer...")

    manifold_in = Lorentz(1.0)
    manifold_hidden = Lorentz(1.0)
    manifold_out = Lorentz(1.0)
    # Define model
    if args.model_name == 'HELM_MiCE':
        decoder = LorentzDeepSeekV3(
            args,
            manifold_in,
            manifold_hidden,
            manifold_out
        )
    elif args.model_name == 'HELM_D':
        decoder = LTransformerDecoder(
            manifold_in,
            manifold_hidden,
            manifold_out,
            args.arch,
            args.vocab_size,
            args.max_seq_len,
        )
    else:
        raise NotImplementedError

    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"Total parameters: {num_params:,}")
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    print('Loading dataset...')
    train_dataloader = prepare_data(tokenizer, args)

    print("Dataset loaded and DataLoader prepared.")

    print('Preparing for accelerator...')
    if not args.project_emb:
        train_dataloader, decoder, scheduler_euc, scheduler_hyp, optimizer = prepare_accelerator(accelerator, train_dataloader, decoder, args)
    else:
        train_dataloader, decoder, scheduler_euc, optimizer = prepare_accelerator(accelerator, train_dataloader, decoder, args)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    global_step = 0
    start_epoch = 0

    print('Training started...')

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=args.log_dir)

    decoder.train()
    losses = []
    avg_loss = 0.0
    total_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    progress_bar = tqdm(range(total_steps), unit="update")
    if args.model_name == 'HELM_MiCE':
        local_stash = None
    for _, batch in enumerate(train_dataloader):
        seq_ids = batch["sequence_id"]  
        same_doc = seq_ids.unsqueeze(1) == seq_ids.unsqueeze(2)
        block_mask = ~same_doc
        with accelerator.accumulate(decoder):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            if args.model_name == 'HELM_MiCE':
                logits, indices_list, scores_list = decoder(input_ids, attn_mask=block_mask)
            else:
                logits = decoder(input_ids, attn_mask=block_mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            if args.model_name == 'HELM_MiCE':
                loss_bal = logits.new_tensor(0.0)
                for idx, scr in zip(indices_list, scores_list):
                    loss_bal = loss_bal + sequence_balance_loss(
                                scr, torch.tensor(idx, device='cpu', dtype=torch.float32), args.seq_bal_alpha)
                if local_stash is None:
                    local_stash = [
                        torch.zeros(args.n_routed_experts, dtype=torch.float32, device="cpu")
                        for _ in range(len(indices_list))
                    ]
                for lid, idx in enumerate(indices_list):                       # idx (tokens, topk)
                    local_stash[lid] += torch.tensor(idx, device='cpu', dtype=torch.float32)
                
                indices_list = None
                loss = loss + loss_bal

            accelerator.backward(loss)
            avg_loss += loss.item()

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(decoder.parameters(), 1.0)
                optimizer.step()
                if not args.project_emb:
                    scheduler_euc.step()
                    scheduler_hyp.step()
                else:
                    scheduler_euc.step()

                optimizer.zero_grad()

                #Preparing info for logging
                gathered_loss = accelerator.gather(torch.tensor(avg_loss, device=accelerator.device))
                mean_loss = gathered_loss.mean().item() / accelerator.gradient_accumulation_steps
                avg_loss = 0.0
                losses.append(mean_loss)

                if args.model_name == 'HELM_MiCE':
                    moe_layer_id = 0
                    for layer in decoder.module.layers:
                        if not isinstance(layer.ffn, LorentzMoE):
                            continue                                  
                        stash = local_stash[moe_layer_id].to(layer.ffn.gate.bias.device)
                        if dist.is_initialized() and dist.get_world_size() > 1:
                            dist.all_reduce(stash, op=dist.ReduceOp.SUM) 
                        with torch.no_grad():
                            util = stash / stash.sum()  
                            mean = util.mean()
                            layer.ffn.gate.bias += layer.ffn.gate.bias_update_spd * (mean - util)
                        if dist.is_initialized() and dist.get_world_size() > 1:
                            dist.broadcast(layer.ffn.gate.bias.data, src=0)
                        moe_layer_id += 1
                    local_stash = None
                    stash = None

                if accelerator.is_main_process and writer is not None:
                        writer.add_scalar("train/loss", mean_loss, global_step)
                        current_lr_euc = scheduler_euc.get_last_lr()[0]
                        writer.add_scalar("train/lr_euc", current_lr_euc, global_step)
                        if not args.project_emb:
                            current_lr_hyp = scheduler_hyp.get_last_lr()[0]
                            writer.add_scalar("train/lr_hyp", current_lr_hyp, global_step)

                progress_bar.set_postfix({
                    "Batch Loss": f"{mean_loss:.4f}"
                })  
                global_step+=1
                if global_step % 100 == 0:
                    if not args.project_emb:
                        save_checkpoint_both(accelerator, decoder, optimizer, scheduler_euc, scheduler_hyp, CHECKPOINT_DIR, global_step)
                    else:
                        save_checkpoint_euc(accelerator, decoder, optimizer, scheduler_euc, CHECKPOINT_DIR, global_step)
                progress_bar.update(1)
    if not args.project_emb:
        save_checkpoint_both(accelerator, decoder, optimizer, scheduler_euc, scheduler_hyp, CHECKPOINT_DIR, global_step)
    else:
        save_checkpoint_euc(accelerator, decoder, optimizer, scheduler_euc, CHECKPOINT_DIR, global_step)

def main() -> None:
    args = parser.parse_args()
    access_token = '...'
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", token=access_token)
    tokenizer.pad_token = tokenizer.eos_token  
    train(args, tokenizer)

if __name__ == "__main__":
    main()