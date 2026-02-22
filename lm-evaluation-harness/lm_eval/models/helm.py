from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model

from datasets import load_from_disk

import torch.nn.functional as F
import torch.distributed as dist

from .hypercore.manifolds import Lorentz
import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from .helm_module.helm_mice import LorentzDeepSeekV3
from .helm_module.helm_d import LTransformerDecoder
from tqdm import tqdm
import random

import json, os, torch.distributed as dist
from pathlib import Path


class MiCE_120M:
    max_batch_size: int = 8
    max_seq_len: int = 2048
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 128256
    dim: int = 390
    inter_dim: int = 390*4
    moe_inter_dim: int = 780
    n_layers: int = 6
    n_dense_layers: int = 1
    n_heads: int = 6
    # moe
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    bias_update_speed: float = 0.005
    seq_bal_alpha:     float = 1e-4
    train_curv = False
    project_emb = False
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 65
    qk_nope_head_dim: int = 33
    qk_rope_head_dim: int = 17
    v_head_dim: int = 33
    use_hope: bool = True
    # yarn
    original_seq_len: int = 2048
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.

@register_model("helm_mice_120M")
class HELM_MiCE_120M(LM):
    def __init__(
        self,
        device='cuda:0',
        batch_size=1,
        ckpt_dir='...',
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        access_token = '...'
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            token=access_token
        )
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        manifold_in = Lorentz(1.0)
        manifold_hidden = Lorentz(1.0)
        manifold_out = Lorentz(1.0)
        decoder = LorentzDeepSeekV3(
            MiCE_120M(),
            manifold_in,
            manifold_hidden,
            manifold_out
        )
        checkpoint_data = torch.load(ckpt_dir, map_location="cpu", weights_only=True)
        decoder.load_state_dict(checkpoint_data["model_state_dict"])
        self.model = decoder.to(self.device)
        self.model.eval()

    def _score_sequence(self, prompt: str, continuation: str) -> Tuple[float, float, bool]:
        self.model.eval()
        prompt_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

        cont_ids = self.tokenizer.encode(
            continuation, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        cont_len = cont_ids.size(1)

        prompt_ids = prompt_ids[..., -(2048 - cont_len) :]
        prompt_len = prompt_ids.size(1)

        input_ids = torch.cat([prompt_ids, cont_ids], dim=1) 

        with torch.no_grad():
            logits = self.model(input_ids)  

        total_logprob = 0.0
        greedy_flag = True
        for i in range(cont_len):
            token_idx = prompt_len + i - 1 
            next_logit = logits[0, token_idx]
            log_probs = F.log_softmax(next_logit, dim=-1)
            token_id = cont_ids[0, i]
            total_logprob += float(log_probs[token_id])
            if greedy_flag and log_probs.argmax().item() != token_id:
                greedy_flag = False

        perplexity = float(np.exp(-total_logprob / max(cont_len, 1)))
        return total_logprob, perplexity, greedy_flag

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:  
        results: List[Tuple[float, bool]] = []
        for _, string in enumerate(tqdm([req.args for req in requests])):
            context, continuation = string
            if not context.endswith(". "):
                context = context + "."
            lp, _ppl, greedy = self._score_sequence(context, continuation)
            results.append((lp, greedy))
        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        ans: List[Tuple[float, bool]] = []
        for inst in requests:
            toks = self.tokenizer.encode(inst.prompt, add_special_tokens=True)
            logp = 0.0
            for i in range(1, len(toks)):
                inp = torch.tensor([[toks[i - 1]]], device=self.device)
                out_logits, _, _ = self.model(inp)[:, -1]  # [1, V]
                logp += F.log_softmax(out_logits, dim=-1)[0, toks[i]].item()
            ans.append((logp, False))
        return ans

    def generate_until(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Generation not required for MCQ evaluation.")

class MiCE_1B:
    max_batch_size: int = 8
    max_seq_len: int = 2048
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 128256
    dim: int = 64*14
    inter_dim: int = 12*64*4
    moe_inter_dim: int = 14*64*2
    n_layers: int = 15
    n_dense_layers: int = 1
    n_heads: int = 14
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    bias_update_speed: float = 0.005
    seq_bal_alpha:     float = 1e-4
    train_curv = False
    project_emb = True
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 257
    qk_nope_head_dim: int = 129
    qk_rope_head_dim: int = 65
    v_head_dim: int = 129
    use_hope: bool = True
    # yarn
    original_seq_len: int = 2048
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.

@register_model("helm_mice_1B")
class HELM_MiCE_1B(LM):
    def __init__(
        self,
        device='cuda:0',
        batch_size=1,
        ckpt_dir='...',
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        access_token = '...'
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            token=access_token
        )
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        manifold_in = Lorentz(1.0)
        manifold_hidden = Lorentz(1.0)
        manifold_out = Lorentz(1.0)
        decoder = LorentzDeepSeekV3(
            MiCE_1B(),
            manifold_in,
            manifold_hidden,
            manifold_out
        )
        checkpoint_data = torch.load(ckpt_dir, map_location="cpu", weights_only=True)
        decoder.load_state_dict(checkpoint_data["model_state_dict"])
        self.model = decoder.to(self.device)
        self.model.eval()

    def _score_sequence(self, prompt: str, continuation: str) -> Tuple[float, float, bool]:
        self.model.eval()
        prompt_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

        cont_ids = self.tokenizer.encode(
            continuation, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        cont_len = cont_ids.size(1)

        prompt_ids = prompt_ids[..., -(2048 - cont_len) :]
        prompt_len = prompt_ids.size(1)
        input_ids = torch.cat([prompt_ids, cont_ids], dim=1) 

        with torch.no_grad():
            logits = self.model(input_ids)  

        total_logprob = 0.0
        greedy_flag = True
        for i in range(cont_len):
            token_idx = prompt_len + i - 1 
            next_logit = logits[0, token_idx]
            log_probs = F.log_softmax(next_logit, dim=-1)
            token_id = cont_ids[0, i]
            total_logprob += float(log_probs[token_id])
            if greedy_flag and log_probs.argmax().item() != token_id:
                greedy_flag = False

        perplexity = float(np.exp(-total_logprob / max(cont_len, 1)))
        return total_logprob, perplexity, greedy_flag

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:  
        results: List[Tuple[float, bool]] = []
        for _, string in enumerate(tqdm([req.args for req in requests])):
            context, continuation = string
            if not context.endswith(". "):
                context = context + "."
            lp, _ppl, greedy = self._score_sequence(context, continuation)
            results.append((lp, greedy))
        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        ans: List[Tuple[float, bool]] = []
        for inst in requests:
            toks = self.tokenizer.encode(inst.prompt, add_special_tokens=True)
            logp = 0.0
            for i in range(1, len(toks)):
                inp = torch.tensor([[toks[i - 1]]], device=self.device)
                out_logits, _, _ = self.model(inp)[:, -1]  # [1, V]
                logp += F.log_softmax(out_logits, dim=-1)[0, toks[i]].item()
            ans.append((logp, False))
        return ans

    def generate_until(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Generation not required for MCQ evaluation.")


@register_model("helm_d_115M")
class HELM_D_115M(LM):
    def __init__(
        self,
        device='cuda:0',
        batch_size=1,
        ckpt_dir='...',
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        access_token = '...'
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            token=access_token
        )
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        manifold_in = Lorentz(1.0)
        manifold_hidden = Lorentz(1.0)
        manifold_out = Lorentz(1.0)
        decoder = LTransformerDecoder(
            manifold_in,
            manifold_hidden,
            manifold_out,
            arch="L6_W390_A6",  
            vocab_size=128256,     #vocab size of llama3.1-8B tokenizer
            context_length=2048
        )
        checkpoint_data = torch.load(ckpt_dir, map_location="cpu", weights_only=True)
        decoder.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
        self.model = decoder.to(self.device)
        self.model.eval()

    def _score_sequence(self, prompt: str, continuation: str) -> Tuple[float, float, bool]:
        self.model.eval()
        prompt_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.device)
        
        cont_ids = self.tokenizer.encode(
            continuation, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        cont_len = cont_ids.size(1)

        prompt_ids = prompt_ids[..., -(2048 - cont_len) :]
        prompt_len = prompt_ids.size(1)
        input_ids = torch.cat([prompt_ids, cont_ids], dim=1) 

        with torch.no_grad():
            logits = self.model(input_ids)  

        total_logprob = 0.0
        greedy_flag = True
        for i in range(cont_len):
            token_idx = prompt_len + i - 1 
            next_logit = logits[0, token_idx]
            log_probs = F.log_softmax(next_logit, dim=-1)
            token_id = cont_ids[0, i]
            total_logprob += float(log_probs[token_id])
            if greedy_flag and log_probs.argmax().item() != token_id:
                greedy_flag = False

        perplexity = float(np.exp(-total_logprob / max(cont_len, 1)))
        return total_logprob, perplexity, greedy_flag

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:  
        results: List[Tuple[float, bool]] = []
        for _, string in enumerate(tqdm([req.args for req in requests])):
            context, continuation = string
            if not context.endswith(". "):
                context = context + "."
            lp, _ppl, greedy = self._score_sequence(context, continuation)
            results.append((lp, greedy))
        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        ans: List[Tuple[float, bool]] = []
        for inst in requests:
            toks = self.tokenizer.encode(inst.prompt, add_special_tokens=True)
            logp = 0.0
            for i in range(1, len(toks)):
                inp = torch.tensor([[toks[i - 1]]], device=self.device)
                out_logits, _, _ = self.model(inp)[:, -1]  # [1, V]
                logp += F.log_softmax(out_logits, dim=-1)[0, toks[i]].item()
            ans.append((logp, False))
        return ans

    def generate_until(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Generation not required for MCQ evaluation.")