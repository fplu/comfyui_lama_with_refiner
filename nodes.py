from __future__ import annotations
from typing import Any
import torch
import torch.jit

from torch import Tensor
from tqdm import trange

from comfy.utils import ProgressBar
from comfy.model_management import get_torch_device
import folder_paths

import torch


from .util import (
    to_lama,
    to_comfy,
    pad_img_to_modulo,
    make_tensor_ready_for_training,
)
from .refine import refine_predict


class LoadInpaintLaMaModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("inpaint"),),
            }
        }

    RETURN_TYPES = ("INPAINT_LAMA_MODEL",)
    CATEGORY = "inpaint"
    FUNCTION = "load"

    def load(self, model_name: str):
        from spandrel import ModelLoader

        with torch.inference_mode(False):
            model_file = folder_paths.get_full_path("inpaint", model_name)
            if model_file is None:
                raise RuntimeError(f"Model file not found: {model_name}")
            sd = torch.jit.load(model_file, map_location="cpu").state_dict()

            model = ModelLoader().load_from_state_dict(sd)
            device = get_torch_device()
            model.to(device)
            # We deliberately do two seemingly conflicting things here:
            #  - `make_tensor_ready_for_training(...)` turns on gradient tracking so we
            #     can run autograd during the refinement loop that follows.
            #  - `model.eval()` forces the network into inference mode so that
            #     * BatchNorm layers use their running (population) statistics instead of
            #       recalculating per-image means/vars, and
            #     * Dropout (if any) is disabled.
            #
            # This combo—"gradients ON, inference behaviour ON"— is necessarry for inference with refinement.
            make_tensor_ready_for_training(model.model.get_submodule("model.model"))
            model.eval()
            return (model,)


class InpaintWithLamaRefinerModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inpaint_model": ("INPAINT_LAMA_MODEL",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "inpaint"
    FUNCTION = "inpaint"

    def inpaint(
        self,
        inpaint_model: Any,
        image: Tensor,
        mask: Tensor,
        seed: int,
    ):
        image, mask = to_lama(image, mask)

        batch_size = image.shape[0]
        if mask.shape[0] != batch_size:
            mask = mask[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)

        device = get_torch_device()

        batch_image = []
        pbar = ProgressBar(batch_size)

        for i in trange(batch_size):
            torch.manual_seed(seed)

            batch = dict(image=image[i], mask=mask[i])
            batch["unpad_to_size"] = [
                torch.tensor([batch["image"].shape[1]]),
                torch.tensor([batch["image"].shape[2]]),
            ]
            batch["image"] = torch.tensor(pad_img_to_modulo(batch["image"], 8))[None].to(device)
            batch["mask"] = (
                torch.tensor(pad_img_to_modulo(batch["mask"], 8))[None].float().to(device)
            )

            model = inpaint_model.model.get_submodule("model.model")
            with torch.inference_mode(False):
                res = refine_predict(batch, model)

            batch_image.append(res)
            pbar.update(1)

        result = torch.cat(batch_image, dim=0)
        return (to_comfy(result),)
