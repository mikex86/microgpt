from typing import Optional, List

import torch

torch_linear_original = torch.nn.functional.linear
torch_matmul_original = torch.matmul
torch_silu_original = torch.nn.functional.silu


def handle_fp16_cpu(tensors: List[torch.tensor]) -> (bool, List[torch.tensor]):
    """
    Giant hack to support float16 on cpu.
    Trades speed for memory usage.
    """
    all_fp16_cpu = True
    out = []

    for tensor in tensors:
        if tensor is not None:
            if tensor.dtype == torch.float16 and tensor.device.type == 'cpu':
                tensor = tensor.float()
            else:
                all_fp16_cpu = False
        out.append(tensor)

    return all_fp16_cpu, out


def torch_linear_hook(x: torch.tensor, weight: torch.tensor, bias: Optional[torch.tensor] = None):
    all_fp16_cpu, tensors = handle_fp16_cpu([x, weight, bias])
    x, weight, bias = tensors

    # pass to original function
    result = torch_linear_original(x, weight, bias)

    # delete upcasted tensors
    del tensors
    del x, weight, bias

    # make result float16 if all inputs were float16 on cpu
    if all_fp16_cpu:
        result = result.half()

    return result


def torch_matmul_hook(x: torch.tensor, y: torch.tensor, *, out: Optional[torch.tensor] = None):
    all_fp16_cpu, tensors = handle_fp16_cpu([x, y, out])
    x, y, out = tensors

    # pass to original function
    result = torch_matmul_original(x, y, out=out)

    # delete upcasted tensors
    del tensors
    del x, y, out

    # make result float16 if all inputs were float16 on cpu
    if all_fp16_cpu:
        result = result.half()

    return result


def torch_silu_hook(x: torch.tensor, inplace: bool = False):
    all_fp16_cpu, tensors = handle_fp16_cpu([x])
    x = tensors[0]

    # pass to original function
    result = torch_silu_original(x, inplace=inplace)

    # delete upcasted tensors
    del tensors
    del x

    # make result float16 if all inputs were float16 on cpu
    if all_fp16_cpu:
        result = result.half()

    return result


def init_torch_hooks():
    torch.nn.functional.linear = torch_linear_hook
    torch.matmul = torch_matmul_hook
    torch.nn.functional.silu = torch_silu_hook
