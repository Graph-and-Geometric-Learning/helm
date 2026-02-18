# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union
from ...nn.conv import LorentzDropout
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D #TODO change to hyperbolic conv1d
from .config import PeftConfig, PeftType

def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None

if is_bnb_available():
    import bitsandbytes as bnb

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

#TODO: lora config???

@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_type (`str`}: The type of lora.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_type: str = field(default="std", metadata={"help": "Lora type std or riemannian"})
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.LORA


class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward
        #this should be changed to be more automatic
        if config.lora_type == 'hyperbolic':
            self.manifold = model.manifold_hidden

    def _find_and_replace(self):
        print(self)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_type": self.peft_config.lora_type,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_lora is None:
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        kwargs.update({"enable_lora": self.peft_config.enable_lora})
                        new_module = MergedLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    if self.peft_config.lora_type == 'hyperbolic':
                        new_module = Linear(self.model.manifold_hidden, target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        new_module = Linear(None, target.in_features, target.out_features, bias=bias, **kwargs)
                elif self.peft_config.enable_lora is not None:
                    kwargs.update({"enable_lora": self.peft_config.enable_lora})
                    if isinstance(target, Conv1D):
                        in_features, out_features = (
                            target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                        )
                    else:
                        in_features, out_features = target.in_features, target.out_features
                        if kwargs["fan_in_fan_out"]:
                            warnings.warn(
                                "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                                "Setting fan_in_fan_out to False."
                            )
                            kwargs["fan_in_fan_out"] = self.peft_config.fan_in_fan_out = False
                    print("BIAS", bias)
                    #new_module = MergedLinear(in_features, out_features, bias=bias, **kwargs)
                    new_module = MergedLinear(in_features, out_features, **kwargs)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(
        self,
        manifold,
        r: int,
        lora_type: str,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_type = lora_type
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            # self.lora_dropout = nn.Dropout(p=lora_dropout)
            if self.lora_type == 'hyperbolic':
                self.lora_dropout = LorentzDropout(manifold, lora_dropout)
            else:
                self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False

class Linear(nn.Linear, LoraLayer):
    '''
    Lora implemented in a dense layer
    This fine tunes the **space-like dimension of the target Lorentz linear layer
    '''
    def __init__(
        self,
        manifold=None,
        in_features: int = 10,
        out_features: int = 10,
        r: int = 0,
        lora_type: str = "hybrid",
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        output_full_vec: bool = False, # Set this to True if the output should be the entire hyperbolic vector, otherwise output only space-like dimension
        learnable: bool = True, # Set to True is learnable curvature
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, manifold=manifold, r=r, lora_type=lora_type, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        if manifold is not None:
            self.k = manifold.c
        else:
            self.k = 1.0
        self.lora_type = lora_type.split('-')[0]
        if len(lora_type.split('-')) > 1:
            norm_scale = float(lora_type.split('-')[1]) # get the norm scale
        else:
            norm_scale = 0.0 # default norm scale
            
        self.fan_in_fan_out = fan_in_fan_out
        self.output_full_vec = output_full_vec
        # Actual trainable parameters
        if r > 0:
            if self.lora_type == 'hybrid':
                self.lora_A = nn.Linear(in_features + 1, r, bias=False)
                self.lora_B = nn.Linear(r+1, out_features, bias=False)
                self.k = nn.Parameter(torch.tensor(self.k), requires_grad=learnable)
            elif self.lora_type == 'hyperbolic':
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r+1, out_features, bias=False)
            elif self.lora_type == 'std':
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
            else:
                print(f'Unknown lora type: {self.lora_type}')
                raise NotImplementedError
            self.norm_scale = nn.Parameter(torch.tensor(norm_scale), requires_grad=True)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    @staticmethod
    def lorentz_expmap0(u, k, dim=-1, min=1e-8):
        x = u.narrow(-1, 1, u.size(-1) - 1)
        sqrtK = torch.sqrt(torch.abs(k))
        x_norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min, max=math.asinh(2**15))
        theta = x_norm / sqrtK

        l_v = sqrtK * torch.cosh(theta)
        r_v = sqrtK * torch.sinh(theta) * x / x_norm
        v = torch.cat((l_v, r_v), dim)
        return v

    @staticmethod
    def lorentz_logmap0(x, k, dim=-1, min=1e-7):
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        sqrtK = torch.sqrt(torch.abs(k))
        y_norm = torch.norm(y, p=2, dim=dim, keepdim=True).clamp(min)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[..., 0:1] / sqrtK, min=1.0 + 1e-7)
        res[..., 1:] = sqrtK * torch.arccosh(theta) * y / y_norm
        return res
    
    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype

        if self.disable_adapters:
            if self.r > 0 and self.merged:
                matmul_output = self.lora_B.weight @ self.lora_A.weight
                self.weight.data -= transpose(matmul_output.to(previous_dtype), self.fan_in_fan_out) * self.scaling
                self.merged = False

            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias) # lienar on the space-like dimension
            if self.r > 0:
                if self.lora_type == 'hybrid':
                    x = x.to(self.lora_A.weight.dtype)
                    x = self.lora_dropout(x)
                    
                    # the following five lines is for padding a column with zerso at the leftmost of x
                    num_dims = len(x.size())
                    padding = [0] * (2 * num_dims)
                    padding[-2 * num_dims] = 1
                    x = x / x.norm(dim=-1, keepdim=True) * self.norm_scale.exp().clamp(max=10) # normalization
                    x = F.pad(x, padding, "constant", value=0) # padding with zero, x: d → d+1
                    
                    x = self.lorentz_expmap0(x, self.k) # exponential map
                    
                    x_space = self.lora_A(x) # lora A weight matrix
                    x_time = ((x_space**2).sum(dim=-1, keepdim=True) + self.k.abs()).sqrt() # Transformation on the manifold
                    x = torch.cat([x_time, x_space], dim=-1) # cat the time value
                    x_space = self.lora_B(x) # lora B weight matrix
                    x_time = ((x_space**2).sum(dim=-1, keepdim=True) + self.k.abs()).sqrt() # Transformation on the manifold
                    x = torch.cat([x_time, x_space], dim=-1) # cat the time value
                    
                    x = self.lorentz_logmap0(x, self.k)[..., 1:] # use logrithmic map to map it back to the tangent space
                    x = x * self.scaling
                    result += x
                elif self.lora_type == 'hyperbolic':
                    x = x.to(self.lora_A.weight.dtype)
                    x = self.lora_dropout(x)
                    x_space = self.lora_A(x)
                    x_time = ((x_space**2).sum(dim=-1, keepdim=True) + self.k.abs()).sqrt() # Transformation on the manifold
                    x = torch.cat([x_time, x_space], dim=-1) # cat the time value
                    x_space = self.lora_B(x) # lora B weight matrix
                    x = x_space * self.scaling #scale before mapping back to manifold
                    result += x # adds the space-like dimension
                    if self.output_full_vec:
                        #outputs the entire hyperbolic vector
                        result_time = ((result**2).sum(dim=-1, keepdim=True) + self.k.abs()).sqrt()
                        result = torch.cat([result_time, result], dim=-1)
                elif self.lora_type == 'std':
                    result += self.lora_B(self.lora_A(self.lora_dropout(x.to(self.lora_A.weight.dtype)))) * self.scaling
                else:
                    raise NotImplementedError
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result
