# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
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

import logging
import math
import os
import types

import comfy
import cv2
import folder_paths
import numpy as np
import torch
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms.functional import normalize

from .BEN2 import BEN_Base
from .dreamo_hook import dreamo_forward, dreamo_forward_orig


def tensor_to_image(tensor, rgb2bgr=False):
    # inp: pytorch tensor, shape [b, h, w, c], range [0, 1], rgb
    # out: numpy image, shape [h, w, c], range [0, 255]
    image = tensor.squeeze(0).mul(255).clamp(0, 255).byte().cpu().numpy()
    if rgb2bgr:
        image = image[..., [2, 1, 0]]
    return image

def image_to_tensor(image, bgr2rgb=False):
    # inp: numpy image, shape [h, w, c], range [0, 255]
    # out: pytorch tensor, shape [h, w, c], range [0, 1]
    tensor = torch.clamp(torch.from_numpy(image).float() / 255., 0, 1)
    if bgr2rgb:
        tensor = tensor[..., [2, 1, 0]]
    return tensor

def resize_numpy_image_area(image, area=512 * 512):
    h, w = image.shape[:2]
    k = math.sqrt(area / (h * w))
    h = int(h * k) - (int(h * k) % 16)
    w = int(w * k) - (int(w * k) % 16)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return image

def resize_numpy_image_long(image, long_edge=768):
    h, w = image.shape[:2]
    if max(h, w) <= long_edge:
        return image
    k = long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return image

def set_extra_config_model_path(extra_config_models_dir_key, models_dir_name:str):
    models_dir_default = os.path.join(folder_paths.models_dir, models_dir_name)
    if extra_config_models_dir_key not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[extra_config_models_dir_key] = (
            [os.path.join(folder_paths.models_dir, models_dir_name)], folder_paths.supported_pt_extensions)
    else:
        if not os.path.exists(models_dir_default):
            os.makedirs(models_dir_default, exist_ok=True)
        folder_paths.add_model_folder_path(extra_config_models_dir_key, models_dir_default, is_default=True)

set_extra_config_model_path("dreamo", "dreamo")

class DreamOProcessorLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("DREAMO_PROCESSOR",)
    FUNCTION = "load_model"
    CATEGORY = "dreamo"

    def load_model(self):
        device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        hf_hub_download(repo_id='PramaLLC/BEN2', filename='model.safetensors',
                        local_dir=os.path.join(folder_paths.models_dir, 'dreamo'))
        ben2_path = folder_paths.get_full_path("dreamo", "model.safetensors")

        ben2 = BEN_Base().eval()
        ben2.load_state_dict(load_file(ben2_path))

        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=device,
        )
        face_helper.face_det.to(offload_device)
        face_helper.face_parse.to(offload_device)

        dreamo_processor = {
            "ben2": ben2,
            "face_helper": face_helper,
        }

        return (dreamo_processor,)

class DreamORefEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pixels": ("IMAGE", ),
            "vae": ("VAE", ),
            "dreamo_processor": ("DREAMO_PROCESSOR", ),
            "resolution": ("INT", {"default": 512, "min": 512, "max": 1024, "step": 16,
                              "tooltip": "The resolution of the reference image."}),
            "ref_task": (
                ["ip","id","style"],
                {
                    "default": "ip",
                    "tooltip": "ip: will remove the backgound of the reference image.\n"
                               "id: will align&crop the face from the reference image, similar to PuLID\n"
                               "style: will keep the backgound of the reference image. you still need trigger meta prompt to activate the style transfer task\n"
                }
            )
        }}
    RETURN_TYPES = ("LATENT", "IMAGE",)
    FUNCTION = "encode"

    CATEGORY = "dreamo"

    def encode(self, pixels, vae, dreamo_processor, resolution, ref_task):
        device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        image = tensor_to_image(pixels, rgb2bgr=False)

        def get_align_face(img):
            dreamo_processor["face_helper"].clean_all()
            image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            dreamo_processor["face_helper"].read_image(image_bgr)
            dreamo_processor["face_helper"].get_face_landmarks_5(only_center_face=True)
            dreamo_processor["face_helper"].align_warp_face()
            if len(dreamo_processor["face_helper"].cropped_faces) == 0:
                return None
            align_face = dreamo_processor["face_helper"].cropped_faces[0]

            # input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
            input = image_to_tensor(align_face, bgr2rgb=True).permute(2, 0, 1).unsqueeze(0) # b c h w
            input = input.to(device)
            parsing_out = dreamo_processor["face_helper"].face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(input)
            # only keep the face features
            face_features_image = torch.where(bg, white_image, input)  # b c h w
            face_features_image = tensor_to_image(face_features_image.permute(0, 2, 3, 1), rgb2bgr=False)

            return face_features_image

        # possible bg remove and face alignment
        if ref_task == 'id':
            dreamo_processor["face_helper"].face_det.to(device)
            dreamo_processor["face_helper"].face_parse.to(device)
            image = resize_numpy_image_long(image, long_edge=1024)
            id_image = get_align_face(image)
            if id_image is None:
                # regard as ip if no face detected
                dreamo_processor["ben2"].to(device)
                image = dreamo_processor["ben2"].inference(Image.fromarray(image))
                dreamo_processor["ben2"].to(offload_device)
            else:
                image = id_image
            dreamo_processor["face_helper"].face_det.to(offload_device)
            dreamo_processor["face_helper"].face_parse.to(offload_device)
        elif ref_task != 'style':
            dreamo_processor["ben2"].to(device)
            image = dreamo_processor["ben2"].inference(Image.fromarray(image))
            dreamo_processor["ben2"].to(offload_device)
        if ref_task != 'id':
            image = resize_numpy_image_area(np.array(image), area=resolution*resolution)

        image = image_to_tensor(image, bgr2rgb=False).unsqueeze(0) # b h w c
        latent = vae.encode(image[:,:,:,:3])
        latent = (latent - 0.1159) * 0.3611
        logging.info(f'latent dtype {latent.dtype}')

        return (latent, image,)

class ApplyDreamO:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ref1": ("LATENT",),
            },
            "optional": {
                "ref2": ("LATENT",),
                "ref3": ("LATENT",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "dreamo"

    def apply(self, model, ref1, ref2=None, ref3=None):
        model = model.clone()
        to = model.model_options["transformer_options"]

        hf_hub_download(repo_id='ByteDance/DreamO', filename='embedding.safetensors',
                        local_dir=os.path.join(folder_paths.models_dir, 'dreamo'))
        embedding_path = folder_paths.get_full_path("dreamo", "embedding.safetensors")
        embedding = load_file(embedding_path)

        ref_conds = [ref1]
        if ref2 is not None:
            ref_conds.append(ref2)
        if ref3 is not None:
            ref_conds.append(ref3)
        to['dreamo_ref_conds'] = ref_conds
        to["dreamo_task_embedding"] = embedding["dreamo_task_embedding.weight"]  # 10x3072
        to["dreamo_idx_embedding"] = embedding["dreamo_idx_embedding.weight"]  # 2x3072

        if len(model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "dreamo")) == 0:
            # Just add it once when connecting in series
            model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "dreamo",
                                       dreamo_outer_sample_wrappers_with_override)

        return (model,)


def set_hook(diffusion_model, target_forward_orig, target_forward):
    diffusion_model.old_forward_orig_for_dreamo = diffusion_model.forward_orig
    diffusion_model.forward_orig = types.MethodType(target_forward_orig, diffusion_model)

    diffusion_model.old_forward_for_dreamo = diffusion_model.forward
    diffusion_model.forward = types.MethodType(target_forward, diffusion_model)

def clean_hook(diffusion_model):
    if hasattr(diffusion_model, "old_forward_orig_for_dreamo"):
        diffusion_model.forward_orig = diffusion_model.old_forward_orig_for_dreamo
        diffusion_model.forward = diffusion_model.old_forward_for_dreamo
        del diffusion_model.old_forward_orig_for_dreamo
        del diffusion_model.old_forward_for_dreamo

def dreamo_outer_sample_wrappers_with_override(wrapper_executor, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    cfg_guider = wrapper_executor.class_obj
    diffusion_model = cfg_guider.model_patcher.model.diffusion_model
    set_hook(diffusion_model, dreamo_forward_orig, dreamo_forward)
    try :
        out = wrapper_executor(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
    finally:
        clean_hook(diffusion_model)

    return out

NODE_CLASS_MAPPINGS = {
    "DreamOProcessorLoader": DreamOProcessorLoader,
    "DreamORefEncode": DreamORefEncode,
    "ApplyDreamO": ApplyDreamO,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamOProcessorLoader": "DreamO Processor Loader",
    "DreamORefEncode": "DreamO Ref Image Encode",
    "ApplyDreamO": "Apply DreamO",
}