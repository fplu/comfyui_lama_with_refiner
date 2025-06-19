from __future__ import annotations

from typing import Tuple, Set
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    _, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def pad_tensor_to_modulo(img, mod):
    _, _, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode="reflect")


def mask_unsqueeze(mask: Tensor):
    if len(mask.shape) == 3:  # BHW -> B1HW
        mask = mask.unsqueeze(1)
    elif len(mask.shape) == 2:  # HW -> B1HW
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def to_torch(image: Tensor, mask: Tensor | None = None):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2)  # BHWC -> BCHW
    if mask is not None:
        mask = mask_unsqueeze(mask)
    if image.shape[2:] != mask.shape[2:]:
        raise ValueError(
            f"Image and mask must be the same size. {image.shape[2:]} != {mask.shape[2:]}"
        )
    return image, mask


def to_lama(image: Tensor, mask: Tensor):
    image, mask = to_torch(image, mask)
    return image, mask


def to_comfy(image: Tensor):
    print(f"image out -> max {np.max(image.numpy())} --- min {np.min(image.numpy())}")
    return image.permute(0, 2, 3, 1)  # BCHW -> BHWC


TensorSelector = Set[Tuple[str, str]]  # {(module_name, tensor_name), …}
SELECTED: TensorSelector = {("Conv2d", "weight"), ("Conv2d", "bias")}  # hard-coded list


def visit_chosen_tensors(
    model: torch.nn.Module,
    fn: Callable[[str, str, torch.Tensor], torch.Tensor],
) -> None:
    for module in model.modules():
        mname = module._get_name()

        # --- ordinary attributes -------------------------------------
        for attr_name, attr_val in vars(module).items():
            if isinstance(attr_val, torch.Tensor) and (mname, attr_name) in SELECTED:
                setattr(module, attr_name, fn(mname, attr_name, attr_val))

        # --- registered parameters -----------------------------------
        for p_name, param in list(module._parameters.items()):
            if param is not None and (mname, p_name) in SELECTED:
                module._parameters[p_name] = fn(mname, p_name, param)

        # --- registered buffers --------------------------------------
        for b_name, buf in list(module._buffers.items()):
            if buf is not None and (mname, b_name) in SELECTED:
                module._buffers[b_name] = fn(mname, b_name, buf)


def check_tensor_ready_for_training(model: torch.nn.Module) -> None:
    def check_tensor(mname: str, tname: str, t: torch.Tensor):
        if t.is_inference():
            raise ValueError(f"{mname} {tname} {t.is_inference()}")
        return t

    visit_chosen_tensors(model, check_tensor)


def make_tensor_ready_for_training(model: torch.nn.Module) -> None:

    def fix_tensor(mname: str, tname: str, t: torch.Tensor) -> torch.Tensor:
        def internal(t: torch.Tensor):
            if not (torch.is_floating_point(t) or torch.is_complex(t)):
                return t
            if t.requires_grad:
                return t
            if isinstance(t, torch.nn.Parameter):
                return torch.nn.Parameter(t.detach().clone(), requires_grad=True)
            else:
                return t.detach().clone().requires_grad_(True)

        res = internal(t)
        return res

    visit_chosen_tensors(model, fix_tensor)
