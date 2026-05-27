# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 Alibaba Ovis-Image Team and The HuggingFace Team. All rights reserved.
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

import inspect
import json
import os
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import Qwen2TokenizerFast, Qwen3Model
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.ovis_image.ovis_image_transformer import OvisImageTransformer2DModel
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = init_logger(__name__)

# ----my_debug---- Module loaded
logger.info("----my_debug---- [pipeline_ovis_image.py] Module loaded successfully")


def get_ovis_image_post_process_func(
    od_config: OmniDiffusionConfig,
):
    logger.info("----my_debug---- [get_ovis_image_post_process_func] Starting post process func setup")
    model_name = od_config.model
    logger.info(f"----my_debug---- [get_ovis_image_post_process_func] model_name={model_name}")
    if os.path.exists(model_name):
        model_path = model_name
        logger.info(f"----my_debug---- [get_ovis_image_post_process_func] Using local model path: {model_path}")
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
        logger.info(f"----my_debug---- [get_ovis_image_post_process_func] Downloaded model to: {model_path}")

    vae_config_path = os.path.join(model_path, "vae/config.json")
    logger.info(f"----my_debug---- [get_ovis_image_post_process_func] Loading VAE config from: {vae_config_path}")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) if "block_out_channels" in vae_config else 8
        logger.info(f"----my_debug---- [get_ovis_image_post_process_func] VAE config: block_out_channels={vae_config.get('block_out_channels')}, vae_scale_factor={vae_scale_factor}")

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)
    logger.info(f"----my_debug---- [get_ovis_image_post_process_func] Image processor created with vae_scale_factor={vae_scale_factor * 2}")

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, int]:
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timetemps
    from the scheduler after the call. Handles custom timeteps. Any kwargs will be supplied to `scheduler.set_timeteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`, *optional*):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """

    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"the current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_timesteps = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"the current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigma schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class OvisImagePipeline(nn.Module, CFGParallelMixin, DiffusionPipelineProfilerMixin):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        logger.info("----my_debug---- [OvisImagePipeline.__init__] Starting pipeline initialization")
        super().__init__()
        self.od_config = od_config
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] od_config.model={od_config.model}")
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] od_config.dtype={od_config.dtype}")
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] od_config.tf_model_config={od_config.tf_model_config}")

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] weights_sources configured: subfolder=transformer, prefix=transformer.")

        self._execution_device = get_local_device()
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] _execution_device={self._execution_device}")

        model = od_config.model
        local_files_only = os.path.exists(model)
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] local_files_only={local_files_only}, model_path_exists={os.path.exists(model)}")

        # Load scheduler
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] Loading scheduler from {model}/scheduler")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] Scheduler loaded: {type(self.scheduler).__name__}")
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] Scheduler config: {self.scheduler.config}")

        # Load text_encoder
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] Loading text_encoder from {model}/text_encoder")
        self.text_encoder = Qwen3Model.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only, dtype=od_config.dtype
        )
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] Text encoder loaded: {type(self.text_encoder).__name__}")
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] Text encoder dtype={self.text_encoder.dtype}, device={next(self.text_encoder.parameters()).device if hasattr(self.text_encoder, 'parameters') else 'N/A'}")
        # Log text encoder config
        if hasattr(self.text_encoder, 'config'):
            te_config = self.text_encoder.config
            logger.info(f"----my_debug---- [OvisImagePipeline.__init__] Text encoder config: hidden_size={getattr(te_config, 'hidden_size', 'N/A')}, num_attention_heads={getattr(te_config, 'num_attention_heads', 'N/A')}, num_hidden_layers={getattr(te_config, 'num_hidden_layers', 'N/A')}, vocab_size={getattr(te_config, 'vocab_size', 'N/A')}")

        # Load VAE
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] Loading VAE from {model}/vae")
        self.vae = AutoencoderKL.from_pretrained(model, subfolder="vae", local_files_only=local_files_only).to(
            self._execution_device
        )
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] VAE loaded: {type(self.vae).__name__}")
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] VAE dtype={self.vae.dtype}, device={next(self.vae.parameters()).device}")
        # Log VAE config
        if hasattr(self.vae, 'config'):
            vae_config = self.vae.config
            logger.info(f"----my_debug---- [OvisImagePipeline.__init__] VAE config: block_out_channels={vae_config.block_out_channels if hasattr(vae_config, 'block_out_channels') else 'N/A'}, latent_channels={getattr(vae_config, 'latent_channels', 'N/A')}, scaling_factor={getattr(vae_config, 'scaling_factor', 'N/A')}, shift_factor={getattr(vae_config, 'shift_factor', 'N/A')}")

        # Load tokenizer
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] Loading tokenizer from {model}/tokenizer")
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only
        )
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] Tokenizer loaded: {type(self.tokenizer).__name__}")
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] Tokenizer vocab_size={self.tokenizer.vocab_size}, model_max_length={self.tokenizer.model_max_length}")

        # Initialize transformer
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] Initializing OvisImageTransformer2DModel")
        self.transformer = OvisImageTransformer2DModel(od_config=od_config)
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] Transformer initialized: {type(self.transformer).__name__}")
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] Transformer in_channels={self.transformer.in_channels}, out_channels={self.transformer.out_channels}, inner_dim={self.transformer.inner_dim}")

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] vae_scale_factor={self.vae_scale_factor}")

        self.tokenizer_max_length = 1024
        self.system_prompt = """Describe the image by detailing the color, quantity, text, shape, size, texture, spatial
        relationships of the objects and background: """
        self.user_prompt_begin_id = 28
        self.tokenizer_max_length = 256 + self.user_prompt_begin_id
        self.default_sample_size = 128
        logger.info(f"----my_debug---- [OvisImagePipeline.__init__] tokenizer_max_length={self.tokenizer_max_length}, user_prompt_begin_id={self.user_prompt_begin_id}, default_sample_size={self.default_sample_size}")

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )
        logger.info("----my_debug---- [OvisImagePipeline.__init__] Pipeline initialization complete")

    def _get_messages(
        self,
        prompt: str | list[str] = None,
    ):
        logger.info(f"----my_debug---- [_get_messages] Input prompt type: {type(prompt)}, value: {prompt[:100] if isinstance(prompt, str) else prompt}")
        prompt = [prompt] if isinstance(prompt, str) else prompt

        messages = []

        for each_prompt in prompt:
            message = [
                {
                    "role": "user",
                    "content": self.system_prompt + each_prompt,
                }
            ]
            message = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            messages.append(message)

        logger.info(f"----my_debug---- [_get_messages] Generated {len(messages)} messages")
        for i, msg in enumerate(messages):
            logger.info(f"----my_debug---- [_get_messages] Message {i} length: {len(msg)} chars, preview: {msg[:200]}...")
        return messages

    def _get_ovis_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        num_images_per_prompt: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        logger.info("----my_debug---- [_get_ovis_prompt_embeds] Starting prompt embedding generation")
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        logger.info(f"----my_debug---- [_get_ovis_prompt_embeds] device={device}, dtype={dtype}")

        messages = self._get_messages(prompt)

        batch_size = len(messages)
        logger.info(f"----my_debug---- [_get_ovis_prompt_embeds] batch_size={batch_size}, num_images_per_prompt={num_images_per_prompt}")

        # Tokenization
        logger.info(f"----my_debug---- [_get_ovis_prompt_embeds] Tokenizing with max_length={self.tokenizer_max_length}")
        tokens = self.tokenizer(
            messages,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = tokens.input_ids.to(device=device)
        attention_mask = tokens.attention_mask.to(device=device)
        logger.info(f"----my_debug---- [_get_ovis_prompt_embeds] input_ids shape={input_ids.shape}, dtype={input_ids.dtype}")
        logger.info(f"----my_debug---- [_get_ovis_prompt_embeds] attention_mask shape={attention_mask.shape}, dtype={attention_mask.dtype}")
        logger.info(f"----my_debug---- [_get_ovis_prompt_embeds] input_ids sample (first 50 tokens): {input_ids[0, :50].tolist()}")

        # Text encoder forward pass
        logger.info(f"----my_debug---- [_get_ovis_prompt_embeds] Running text encoder forward pass")
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        prompt_embeds = outputs.last_hidden_state
        logger.info(f"----my_debug---- [_get_ovis_prompt_embeds] last_hidden_state shape={prompt_embeds.shape}, dtype={prompt_embeds.dtype}")
        logger.info(f"----my_debug---- [_get_ovis_prompt_embeds] last_hidden_state stats: min={prompt_embeds.min().item():.4f}, max={prompt_embeds.max().item():.4f}, mean={prompt_embeds.mean().item():.4f}")

        prompt_embeds = prompt_embeds * attention_mask[..., None]
        logger.info(f"----my_debug---- [_get_ovis_prompt_embeds] After attention mask multiplication: shape={prompt_embeds.shape}")

        prompt_embeds = prompt_embeds[:, self.user_prompt_begin_id :, :]
        logger.info(f"----my_debug---- [_get_ovis_prompt_embeds] After slicing from user_prompt_begin_id={self.user_prompt_begin_id}: shape={prompt_embeds.shape}")

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        logger.info(f"----my_debug---- [_get_ovis_prompt_embeds] Final prompt_embeds shape={prompt_embeds.shape}")

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.FloatTensor | None = None,
    ):
        r"""

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`, *optional*):
                torch.device
            num_images_per_prompt: (`int`):
                number of images that should be generated per prompt
            prompt_embeds: (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, text embeddings will be generated from `prompt` input argument.
        """
        logger.info("----my_debug---- [encode_prompt] Starting prompt encoding")
        logger.info(f"----my_debug---- [encode_prompt] prompt={prompt[:100] if isinstance(prompt, str) else prompt}, num_images_per_prompt={num_images_per_prompt}")

        device = device or self._execution_device

        if prompt_embeds is None:
            prompt_embeds = self._get_ovis_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
        else:
            logger.info(f"----my_debug---- [encode_prompt] Using pre-generated prompt_embeds, shape={prompt_embeds.shape}")

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3)
        text_ids[..., 1] = text_ids[..., 1] + torch.arange(prompt_embeds.shape[1])[None, :]
        text_ids[..., 2] = text_ids[..., 2] + torch.arange(prompt_embeds.shape[1])[None, :]
        text_ids = text_ids.to(device=device, dtype=dtype)
        logger.info(f"----my_debug---- [encode_prompt] text_ids shape={text_ids.shape}, dtype={text_ids.dtype}")
        logger.info(f"----my_debug---- [encode_prompt] text_ids sample (first 5 rows): {text_ids[:5].tolist()}")
        return prompt_embeds, text_ids

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"""`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are
                {height} and {width}. Dimension will be resized accordingly"""
            )

        # if callback_on_step_end_tensor_inputs is not None and not all(
        #     k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        # ):
        #     raise ValueError(
        #         f"""`callback_on_step_end_tensor_inputs` has to contain the following keys:
        #         {self._callback_tensor_inputs.keys()}"""
        #     )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list[str]` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: "
                f"{negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if max_sequence_length is not None and max_sequence_length > 256:
            raise ValueError(f"`max_sequence_length` has to be less than or equal to 256 but is {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channel_latents, height, width):
        latents = latents.view(batch_size, num_channel_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channel_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2
        height = int(2 * (int(height) // (vae_scale_factor * 2)))
        width = int(2 * (int(width) // (vae_scale_factor * 2)))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channel_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        logger.info("----my_debug---- [prepare_latents] Starting latent preparation")
        logger.info(f"----my_debug---- [prepare_latents] Input: batch_size={batch_size}, num_channel_latents={num_channel_latents}, height={height}, width={width}, dtype={dtype}, device={device}")

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = int(2 * (int(height) // (self.vae_scale_factor * 2)))
        width = int(2 * (int(width) // (self.vae_scale_factor * 2)))
        logger.info(f"----my_debug---- [prepare_latents] After VAE scaling: height={height}, width={width}")

        shape = (batch_size, num_channel_latents, height, width)
        logger.info(f"----my_debug---- [prepare_latents] Latent shape before packing: {shape}")

        if latents is not None:
            logger.info(f"----my_debug---- [prepare_latents] Using provided latents, shape={latents.shape}")
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        logger.info(f"----my_debug---- [prepare_latents] Generating random noise with generator={generator}")
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        logger.info(f"----my_debug---- [prepare_latents] Random noise generated: shape={latents.shape}, dtype={latents.dtype}")
        logger.info(f"----my_debug---- [prepare_latents] Random noise stats: min={latents.min().item():.4f}, max={latents.max().item():.4f}, mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")

        latents = self._pack_latents(latents, batch_size, num_channel_latents, height, width)
        logger.info(f"----my_debug---- [prepare_latents] After packing: shape={latents.shape}")

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
        logger.info(f"----my_debug---- [prepare_latents] latent_image_ids shape={latent_image_ids.shape}, dtype={latent_image_ids.dtype}")

        return latents, latent_image_ids

    def prepare_timesteps(self, num_inference_steps, sigmas, image_seq_len):
        logger.info("----my_debug---- [prepare_timesteps] Starting timesteps preparation")
        logger.info(f"----my_debug---- [prepare_timesteps] num_inference_steps={num_inference_steps}, sigmas={sigmas}, image_seq_len={image_seq_len}")

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        logger.info(f"----my_debug---- [prepare_timesteps] sigmas after processing: {sigmas[:5] if sigmas is not None else None}... (first 5)")

        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        logger.info(f"----my_debug---- [prepare_timesteps] calculated mu={mu}")

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            self._execution_device,
            sigmas=sigmas,
            mu=mu,
        )
        logger.info(f"----my_debug---- [prepare_timesteps] timesteps shape={timesteps.shape}, num_inference_steps={num_inference_steps}")
        logger.info(f"----my_debug---- [prepare_timesteps] timesteps values (first 10): {timesteps[:10].tolist()}")
        logger.info(f"----my_debug---- [prepare_timesteps] timesteps values (last 10): {timesteps[-10:].tolist()}")
        return timesteps, num_inference_steps

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        negative_text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        do_true_cfg: bool,
        guidance_scale: float,
        cfg_normalize: bool = False,
    ) -> torch.Tensor:
        """
        Diffusion loop with optional classifier-free guidance.

        Args:
            latents: Noise latents to denoise
            timesteps: Diffusion timesteps
            prompt_embeds: Positive prompt embeddings
            negative_prompt_embeds: Negative prompt embeddings
            text_ids: Position IDs for positive text
            negative_text_ids: Position IDs for negative text
            latent_image_ids: Position IDs for image latents
            do_true_cfg: Whether to apply CFG
            guidance_scale: CFG scale factor
            cfg_normalize: Whether to normalize CFG output (default: False)

        Returns:
            Denoised latents
        """
        logger.info("----my_debug---- [diffuse] Starting diffusion loop")
        logger.info(f"----my_debug---- [diffuse] latents shape={latents.shape}, dtype={latents.dtype}")
        logger.info(f"----my_debug---- [diffuse] timesteps count={len(timesteps)}, range=[{timesteps[0].item():.4f}, {timesteps[-1].item():.4f}]")
        logger.info(f"----my_debug---- [diffuse] prompt_embeds shape={prompt_embeds.shape}")
        logger.info(f"----my_debug---- [diffuse] text_ids shape={text_ids.shape}")
        logger.info(f"----my_debug---- [diffuse] latent_image_ids shape={latent_image_ids.shape}")
        logger.info(f"----my_debug---- [diffuse] do_true_cfg={do_true_cfg}, guidance_scale={guidance_scale}, cfg_normalize={cfg_normalize}")

        self.scheduler.set_begin_index(0)

        for i, t in enumerate(timesteps):
            if self.interrupt:
                logger.info(f"----my_debug---- [diffuse] Interrupted at step {i}")
                break

            self._current_timestep = t
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            if i == 0 or i == len(timesteps) - 1:
                logger.info(f"----my_debug---- [diffuse] Step {i}/{len(timesteps)}: timestep={t.item():.4f}, latents stats: min={latents.min().item():.4f}, max={latents.max().item():.4f}, mean={latents.mean().item():.4f}")

            positive_kwargs = {
                "hidden_states": latents,
                "timestep": timestep / 1000,
                "encoder_hidden_states": prompt_embeds,
                "txt_ids": text_ids,
                "img_ids": latent_image_ids,
                "return_dict": False,
            }
            if do_true_cfg:
                negative_kwargs = {
                    "hidden_states": latents,
                    "timestep": timestep / 1000,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "txt_ids": negative_text_ids,
                    "img_ids": latent_image_ids,
                    "return_dict": False,
                }
            else:
                negative_kwargs = None

            # Predict noise with automatic CFG parallel handling
            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg,
                guidance_scale,
                positive_kwargs,
                negative_kwargs,
                cfg_normalize,
            )

            if i == 0 or i == len(timesteps) - 1:
                logger.info(f"----my_debug---- [diffuse] Step {i}: noise_pred shape={noise_pred.shape}, stats: min={noise_pred.min().item():.4f}, max={noise_pred.max().item():.4f}, mean={noise_pred.mean().item():.4f}")

            # Compute the previous noisy sample x_t -> x_t-1 with automatic CFG sync
            latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)

        logger.info(f"----my_debug---- [diffuse] Diffusion loop complete, final latents shape={latents.shape}")
        logger.info(f"----my_debug---- [diffuse] Final latents stats: min={latents.min().item():.4f}, max={latents.max().item():.4f}, mean={latents.mean().item():.4f}")
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        guidance_scale: float = 5.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 256,
    ) -> DiffusionOutput:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                not greater than `1`).
            guidance_scale (`float`, *optional*, defaults to 1.0):
                True classifier-free guidance (guidance scale) is enabled when `guidance_scale` > 1 and
                `negative_prompt` is provided.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`list[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`list`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.ovis_image.OvisImagePipelineOutput`] or `tuple`:
            [`~pipelines.ovis_image.OvisImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """
        logger.info("----my_debug---- [forward] ============ OvisImagePipeline.forward START ============")
        logger.info(f"----my_debug---- [forward] Request prompts: {req.prompts}")
        logger.info(f"----my_debug---- [forward] Request sampling_params: height={req.sampling_params.height}, width={req.sampling_params.width}, num_inference_steps={req.sampling_params.num_inference_steps}, guidance_scale={req.sampling_params.guidance_scale}")
        logger.info(f"----my_debug---- [forward] Request seed={req.sampling_params.seed}, generator={req.sampling_params.generator}")

        # TODO: In online mode, sometimes it receives [{"negative_prompt": None}, {...}], so cannot use .get("...", "")
        # TODO: May be some data formatting operations on the API side. Hack for now.
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        elif req.prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]

        logger.info(f"----my_debug---- [forward] Processed prompt: {prompt}")
        logger.info(f"----my_debug---- [forward] Processed negative_prompt: {negative_prompt}")

        height = req.sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = req.sampling_params.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        sigmas = req.sampling_params.sigmas or sigmas
        guidance_scale = (
            req.sampling_params.guidance_scale if req.sampling_params.guidance_scale is not None else guidance_scale
        )
        generator = req.sampling_params.generator or generator
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )

        logger.info(f"----my_debug---- [forward] Final parameters: height={height}, width={width}, num_inference_steps={num_inference_steps}, guidance_scale={guidance_scale}, num_images_per_prompt={num_images_per_prompt}")

        # Steps:
        # 1. Check Inputs
        # 2. encode prompts
        # 4. Prepare latents
        # 5. Prepare timesteps
        # 6. diffusion latents
        # 7. decode latents
        # 8. post process outputs

        logger.info("----my_debug---- [forward] Step 1: Checking inputs")
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        device = self._execution_device
        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        do_classifier_free_guidance = guidance_scale > 1.0
        logger.info(f"----my_debug---- [forward] batch_size={batch_size}, do_classifier_free_guidance={do_classifier_free_guidance}")

        logger.info("----my_debug---- [forward] Step 2: Encoding prompts")
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )

        negative_text_ids = None
        if do_classifier_free_guidance:
            logger.info("----my_debug---- [forward] Encoding negative prompts for CFG")
            negative_prompt = negative_prompt if negative_prompt is not None else ""
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            logger.info(f"----my_debug---- [forward] negative_prompt_embeds shape={negative_prompt_embeds.shape}")

        # 4. Prepare latent variables
        logger.info("----my_debug---- [forward] Step 4: Preparing latent variables")
        num_channel_latents = self.transformer.in_channels // 4
        logger.info(f"----my_debug---- [forward] num_channel_latents={num_channel_latents} (transformer.in_channels={self.transformer.in_channels})")
        latents, latent_image_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_channel_latents=num_channel_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 5. Prepare timesteps
        logger.info("----my_debug---- [forward] Step 5: Preparing timesteps")

        image_seq_len = latents.shape[1]
        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, image_seq_len)

        # num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        # 6. Denoising loop using diffuse method
        logger.info("----my_debug---- [forward] Step 6: Starting denoising loop")
        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds if do_classifier_free_guidance else None,
            text_ids=text_ids,
            negative_text_ids=negative_text_ids if do_classifier_free_guidance else None,
            latent_image_ids=latent_image_ids,
            do_true_cfg=do_classifier_free_guidance,
            guidance_scale=guidance_scale,
            cfg_normalize=False,
        )

        self._current_timestep = None
        logger.info("----my_debug---- [forward] Step 7: Decoding latents")
        if output_type == "latent":
            image = latents
            logger.info(f"----my_debug---- [forward] Output type is 'latent', skipping VAE decode")
        else:
            logger.info(f"----my_debug---- [forward] Unpacking latents from shape {latents.shape}")
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            logger.info(f"----my_debug---- [forward] Unpacked latents shape={latents.shape}")
            logger.info(f"----my_debug---- [forward] VAE config: scaling_factor={self.vae.config.scaling_factor}, shift_factor={self.vae.config.shift_factor}")
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            logger.info(f"----my_debug---- [forward] Latents after scaling: shape={latents.shape}, stats: min={latents.min().item():.4f}, max={latents.max().item():.4f}")
            logger.info(f"----my_debug---- [forward] Running VAE decode")
            image = self.vae.decode(latents, return_dict=False)[0]
            logger.info(f"----my_debug---- [forward] VAE decode output shape={image.shape}, dtype={image.dtype}")
            logger.info(f"----my_debug---- [forward] VAE decode output stats: min={image.min().item():.4f}, max={image.max().item():.4f}, mean={image.mean().item():.4f}")

        logger.info("----my_debug---- [forward] ============ OvisImagePipeline.forward END ============")
        return DiffusionOutput(
            output=image, stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        logger.info("----my_debug---- [load_weights] Starting weights loading for OvisImagePipeline")
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(weights)
        logger.info(f"----my_debug---- [load_weights] Loaded {len(loaded_params)} parameters")
        logger.info(f"----my_debug---- [load_weights] Sample loaded params (first 10): {list(loaded_params)[:10]}")
        return loaded_params
