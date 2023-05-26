from typing import Optional
import torch
from torch import Tensor


class LoraLinear(torch.nn.Linear):

    def __init__(self, weight: Tensor, bias: Optional[Tensor], lora_A: Tensor, lora_B: Tensor, scaling: float):
        super().__init__(weight.shape[1], weight.shape[0], bias is not None)
        self.weight = weight
        self.bias = bias
        self.lora_A = torch.nn.Parameter(lora_A)
        self.lora_B = torch.nn.Parameter(lora_B)
        self.scaling = scaling

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False


    
    def forward(self, x: Tensor) -> Tensor:
        result = torch.nn.functional.linear(x, self.weight, self.bias)
        
        result += x @ self.lora_A.transpose(-1, -2) @ self.lora_B.transpose(-1, -2) * self.scaling
        return result

    @staticmethod
    def wrap(linear: torch.nn.Linear, r: int, lora_alpha: int = 1, dtype: Optional[torch.dtype] = None):
        if dtype is None:
            dtype = linear.weight.dtype

        lora_A = linear.weight.new_zeros((r, linear.in_features), dtype=dtype)
        lora_B = linear.weight.new_zeros((linear.out_features, r), dtype=dtype)
        scaling = lora_alpha / r

        return LoraLinear(linear.weight, linear.bias, lora_A, lora_B, scaling)
        
    

def lorify_module(module: torch.nn.Module, r: int, lora_alpha: int = 1, dtype: Optional[torch.dtype] = None):
    for name, submodule in module.named_children():
        if isinstance(submodule, torch.nn.Linear):
            setattr(module, name, LoraLinear.wrap(submodule, r, lora_alpha, dtype))
        else:
            lorify_module(submodule, r, lora_alpha)