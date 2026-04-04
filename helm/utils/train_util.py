import math
from datasets import load_from_disk
from torch.optim.lr_scheduler import LambdaLR
import os
import torch
from llmfoundry.data.packing import BinPackCollator
from torch.utils.data import DataLoader  
from helm.hypercore.optimizers import Optimizer
from geoopt import ManifoldParameter
from collections.abc import Mapping

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    num_processes = 1
    ):
        def lr_lambda(current_step):
            current_step = current_step//num_processes
            if current_step < num_warmup_steps:
                # Linear warm-up
                return current_step / num_warmup_steps
            else:
                # Cosine decay
                progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return cosine_decay * (1 - 0.1) + min_lr_ratio
            
        return LambdaLR(optimizer, lr_lambda)


def save_checkpoint_both(accelerator, decoder, optimizer, scheduler_euc, scheduler_hyp, CHECKPOINT_DIR, global_step):
    if not accelerator.is_main_process:
        return  

    unwrapped_model = accelerator.unwrap_model(decoder)
    checkpoint_data = {
        "global_step": global_step,
        "model_state_dict": unwrapped_model.state_dict(),
        "optimizer_state_dict": [opt.state_dict() for opt in optimizer.optimizer],
        "scheduler_euc_state_dict": scheduler_euc.state_dict(),
        "scheduler_hyp_state_dict": scheduler_hyp.state_dict(),
    }
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"Step{global_step}.pt")
    torch.save(checkpoint_data, ckpt_path)
    print(f"Checkpoint saved to: {ckpt_path}")

def save_checkpoint_euc(accelerator, decoder, optimizer, scheduler_euc, CHECKPOINT_DIR, global_step):
    if not accelerator.is_main_process:
        return  

    unwrapped_model = accelerator.unwrap_model(decoder)
    checkpoint_data = {
        "global_step": global_step,
        "model_state_dict": unwrapped_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_euc_state_dict": scheduler_euc.state_dict(),
    }
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"Step{global_step}.pt")
    torch.save(checkpoint_data, ckpt_path)
    print(f"Checkpoint saved to: {ckpt_path}")

def prepare_data(tokenizer, args, return_collator=False):
    print('Loading dataset...')
    lm_dataset = load_from_disk(args.data_path)
    lm_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    tokenizer.pad_token = tokenizer.eos_token 
    CONTEXT_LEN = args.max_seq_len
    PACKED_BINS_PER_GPU = args.max_batch_size
    packing_ratio = args.packing_ratio
    RAW_BATCH = int(round(PACKED_BINS_PER_GPU * packing_ratio))

    def pad_truncate_collator(examples):
        if isinstance(examples, Mapping):
            keys = list(examples.keys())
            n = len(examples[keys[0]])
            examples = [{k: examples[k][i] for k in keys} for i in range(n)]

        if len(examples) > 0 and isinstance(examples[0], Mapping):
            bins = [examples]  
        elif len(examples) > 0 and isinstance(examples[0], (list, tuple)):
            bins = examples 
        else:
            bins = []
        def to_list(x):
            return x.tolist() if torch.is_tensor(x) else list(x)

        B = len(examples)
        input_ids_batch, seqid_batch = [], []
        for bin_samples in bins:
            ids_concat, seq_map = [], []
            sid = 0
            for ex in bin_samples:
                ids = to_list(ex["input_ids"])
                take = min(len(ids), CONTEXT_LEN - len(ids_concat))
                if take <= 0:
                    break
                ids_concat.extend(ids[:take])
                seq_map.extend([sid] * take)
                sid += 1
                if len(ids_concat) >= CONTEXT_LEN:
                    break
            pad_len = CONTEXT_LEN - len(ids_concat)
            if pad_len > 0:
                ids_concat.extend([tokenizer.pad_token_id] * pad_len)
                seq_map.extend([-1] * pad_len) 

            input_ids_batch.append(torch.tensor(ids_concat, dtype=torch.long))
            seqid_batch.append(torch.tensor(seq_map, dtype=torch.long))

        input_ids = torch.stack(input_ids_batch)  
        sequence_id = torch.stack(seqid_batch)  
        attention_mask = (sequence_id != -1).long()                           # (B, L)

        expected = input_ids[:, 1:].clone()                                   # (B, L-1)
        expected = F.pad(expected, (0, 1), value=-100)                        # (B, L), last position ignored
        expected[attention_mask == 0] = -100
        left_valid  = attention_mask[:, :-1].bool()
        right_valid = attention_mask[:, 1:].bool()
        boundary = (sequence_id[:, 1:] != sequence_id[:, :-1]) & left_valid & right_valid
        expected[:, :-1][boundary] = -100
        labels = expected
        
        # sequence_id = sequence_id.clamp_min(0)
        assert (labels[:, :-1][left_valid & right_valid & ~boundary] == input_ids[:, 1:][left_valid & right_valid & ~boundary]).all(), \
            "collator: labels != next token on non-boundary, non-pad positions"
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sequence_id": sequence_id,  
        }
    
    collator = BinPackCollator(
        collator=pad_truncate_collator,
        target_batch_size=PACKED_BINS_PER_GPU,
        max_seq_len=CONTEXT_LEN,
        pad_token_id=tokenizer.pad_token_id,
        padding_side="right",
    )

    train_dataloader = DataLoader(
        lm_dataset,
        batch_size=RAW_BATCH,          
        shuffle=True,
        pin_memory=True,
        collate_fn=collator,    
    )
    if not return_collator:
        return train_dataloader
    else:
        return lm_dataset, collator, RAW_BATCH

def prepare_accelerator(accelerator, train_dataloader, decoder, args):
    def is_no_decay(name):
        return (
            name.endswith(".bias")
            or name.endswith(".weight") and "norm" in name.lower()
            or name.endswith(".gate.bias")
            or name.endswith(".scale")          
            or "w_x" in name.lower()
        )
    
    base_params, no_decay = [], []
    for name, p in decoder.named_parameters():
        if is_no_decay(name) and not isinstance(p, ManifoldParameter) and p.requires_grad:
            no_decay.append(p)
        elif p.requires_grad and not isinstance(p, ManifoldParameter) and p.requires_grad:
            base_params.append(p)

    opt_euc = torch.optim.AdamW([
        {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': no_decay, 'lr': args.lr, 'weight_decay': 0.0},
    ])

    if not args.project_emb:
        optimizer = Optimizer(
                decoder,
                euc_optimizer_type='adamW', euc_lr=args.lr, euc_weight_decay=args.weight_decay,
                hyp_optimizer_type='radam', hyp_lr=args.lr, hyp_weight_decay=0.0,
                stabilize=1
         )
        optimizer.optimizer[0] = opt_euc
    else:
        optimizer = opt_euc

    num_processes = accelerator.num_processes
    train_dataloader = accelerator.prepare(train_dataloader)
    total_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    warmup_steps = int(args.warm_up_ratio * total_steps)
    if not args.project_emb:
        scheduler_euc = get_cosine_schedule_with_warmup(
            optimizer=optimizer.optimizer[0],
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=args.min_lr_ratio,
            num_processes = num_processes
            )
        scheduler_hyp = get_cosine_schedule_with_warmup(
            optimizer=optimizer.optimizer[1],
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=args.min_lr_ratio,
            num_processes = num_processes
        )
        decoder, scheduler_euc, scheduler_hyp = accelerator.prepare(decoder, scheduler_euc, scheduler_hyp)
        for i in range(len(optimizer.optimizer)):
            optimizer.optimizer[i] = accelerator.prepare(optimizer.optimizer[i])
        return train_dataloader, decoder, scheduler_euc, scheduler_hyp, optimizer
    else:
        scheduler_euc = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=args.min_lr_ratio,
            num_processes = num_processes
        )
        decoder, scheduler_euc, optimizer = accelerator.prepare(decoder, scheduler_euc, opt_euc)
        return train_dataloader, decoder, scheduler_euc, optimizer
