import datetime
import os
import pickle
import subprocess
import time
from collections import defaultdict, deque
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _gelu_ignore_parameters(*args, **kwargs) -> nn.Module:
    """Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.
    Args:
        *args: Ignored.
        **kwargs: Ignored.
    Returns:
        activation (nn.Module): GELU activation function.
    """
    activation = nn.GELU()
    return activation


def format_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def save_params_grads(model, run_id, step):
    """Saves the gradients of the model in a pickle file."""
    grads = {}
    out_dir_grad = os.path.join("/user/ganesang/cvlshare/cvl-ganesang/DAC/grads", run_id)
    out_dir_ckpt = out_dir_grad.replace("grads", "ckpts")
    # os.makedirs(out_dir_grad, exist_ok=True)
    # for name, param in model.named_parameters():
    #     g = param.grad
    #     if param.grad is not None:
    #         grads[name] = g.detach().cpu().numpy()
    #     else:
    #         grads[name] = None
    # torch.save(grads, os.path.join(out_dir_grad, f"{step}.pt"))

    os.makedirs(out_dir_ckpt, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir_ckpt, f"{step}.pt"))