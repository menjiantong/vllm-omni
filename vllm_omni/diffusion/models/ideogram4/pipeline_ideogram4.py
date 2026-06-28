# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from diffusers.pipelines.ideogram4.pipeline_ideogram4

import json
import logging
import math
import os
from collections.abc import Iterable
from typing import ClassVar

import torch
import torch.nn as nn
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import (
    from_pretrained_with_prefetch,
    prefetch_subfolders,
)
from vllm_omni.diffusion.models.ideogram4.transformer_ideogram4 import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
    Ideogram4Transformer2DModel,
)
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = logging.getLogger(__name__)


# Latent normalization constants for Ideogram4
# These are learned statistics for denormalizing latents before VAE decode
LATENT_SHIFT = (
    0.01984364,
    0.10149707,
    0.29689495,
    0.27188619,
    -0.21445648,
    -0.15979549,
    0.05021099,
    -0.15083604,
    -0.15360136,
    -0.20131799,
    0.01922352,
    0.0622626,
    0.10140969,
    -0.06739428,
    0.3758261,
    -0.233712,
    0.35164491,
    -0.02590912,
    -0.0271935,
    -0.10833897,
    -0.1476848,
    -0.01130957,
    -0.2298372,
    0.23526423,
    -0.10893522,
    0.11957631,
    0.04047799,
    0.3134589,
    -0.17225064,
    -0.18646109,
    -0.34691978,
    -0.03571246,
    0.02583857,
    0.10190072,
    0.28402294,
    0.26952152,
    -0.21634675,
    -0.17938656,
    0.04358909,
    -0.15007621,
    -0.1548502,
    -0.18971131,
    0.02710861,
    0.05609494,
    0.10697846,
    -0.06854968,
    0.38167698,
    -0.24269937,
    0.35705471,
    -0.03063305,
    -0.02946109,
    -0.11244286,
    -0.14336038,
    -0.01362137,
    -0.21863696,
    0.23228983,
    -0.11739769,
    0.11693044,
    0.02563311,
    0.31356594,
    -0.17420591,
    -0.19006285,
    -0.34905377,
    -0.04025005,
    0.01924137,
    0.07652984,
    0.2995608,
    0.2628057,
    -0.22011674,
    -0.12715361,
    0.04879879,
    -0.14075719,
    -0.15935895,
    -0.2123584,
    0.01974813,
    0.05523547,
    0.10011992,
    -0.06428964,
    0.37781868,
    -0.21491644,
    0.34254215,
    -0.03153528,
    -0.0310082,
    -0.10761415,
    -0.14730405,
    -0.02475182,
    -0.2285588,
    0.2515081,
    -0.10445128,
    0.12446,
    0.07062869,
    0.30880162,
    -0.18016875,
    -0.18869164,
    -0.34533499,
    -0.0129177,
    0.02578168,
    0.07993659,
    0.28642181,
    0.26038408,
    -0.22459419,
    -0.14820155,
    0.04059549,
    -0.14043529,
    -0.16111187,
    -0.2020305,
    0.02602069,
    0.04852717,
    0.10432153,
    -0.06309942,
    0.38402443,
    -0.22397003,
    0.34814481,
    -0.03774432,
    -0.03381438,
    -0.11245691,
    -0.14128767,
    -0.02853208,
    -0.21752016,
    0.24872463,
    -0.11399775,
    0.1222687,
    0.05620835,
    0.309178,
    -0.18065738,
    -0.19401479,
    -0.34495114,
    -0.01760592,
)

LATENT_SCALE = (
    1.63933691,
    1.70204478,
    1.73642566,
    1.90004803,
    1.6675316,
    1.69059584,
    1.56853198,
    1.62314944,
    1.89106626,
    1.58086668,
    1.60822129,
    1.60962993,
    1.63322129,
    1.56074359,
    1.73419528,
    1.7919265,
    1.64040632,
    1.66802808,
    1.60390303,
    1.75480492,
    1.63187587,
    1.64334594,
    1.61722884,
    1.60146046,
    1.63459219,
    1.55291476,
    1.68771497,
    1.68415657,
    1.78966054,
    1.66631641,
    1.65626686,
    1.65976433,
    1.63487607,
    1.69513249,
    1.72933756,
    1.91310663,
    1.67035057,
    1.72286863,
    1.56719251,
    1.61934825,
    1.88628859,
    1.56911539,
    1.59455129,
    1.60829869,
    1.62470611,
    1.56052853,
    1.73677003,
    1.77563606,
    1.63732541,
    1.66370527,
    1.59508952,
    1.75153949,
    1.63029275,
    1.64517667,
    1.61659342,
    1.59722044,
    1.64103121,
    1.5408531,
    1.68610394,
    1.67772755,
    1.78998563,
    1.66621713,
    1.65458955,
    1.66041308,
    1.64710857,
    1.68163503,
    1.74000294,
    1.92784786,
    1.67411194,
    1.67395548,
    1.57406532,
    1.62199356,
    1.87618195,
    1.5584375,
    1.57438785,
    1.61711053,
    1.63094305,
    1.55644029,
    1.73124302,
    1.80666627,
    1.6463621,
    1.65932006,
    1.60816188,
    1.75682671,
    1.64695873,
    1.63121722,
    1.61380832,
    1.60478651,
    1.63396035,
    1.53505068,
    1.65534289,
    1.67132281,
    1.80317197,
    1.6767314,
    1.65700938,
    1.68426259,
    1.65339716,
    1.67540638,
    1.73298504,
    1.94067348,
    1.67893609,
    1.70635117,
    1.5730906,
    1.61928553,
    1.87148809,
    1.56244866,
    1.56697152,
    1.61584394,
    1.62759496,
    1.55480378,
    1.73484107,
    1.79055143,
    1.64688773,
    1.66121492,
    1.60135887,
    1.75254572,
    1.64798332,
    1.62989921,
    1.61381592,
    1.60792883,
    1.63939668,
    1.53075757,
    1.65371318,
    1.66801185,
    1.80029087,
    1.67591476,
    1.65655173,
    1.68533454,
)


# Hidden states of these Qwen3-VL decoder layers are concatenated to form the per-token
# text conditioning consumed by the Ideogram4 transformer.
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

# Subfolders to prefetch for Ideogram4
IDEOGRAM4_SUBFOLDERS = [
    "text_encoder",
    "tokenizer",
    "vae",
    "scheduler",
    "transformer",
    "unconditional_transformer",
]


def _logit_normal_sigmas(
    num_inference_steps: int,
    mu: float,
    std: float = 1.0,
    logsnr_min: float = -15.0,
    logsnr_max: float = 18.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a length-`num_inference_steps` sigma schedule using the Ideogram4 logit-normal flow-matching schedule."""
    intervals = torch.linspace(0.0, 1.0, num_inference_steps + 1, dtype=torch.float64)
    z = torch.special.ndtri(intervals)
    y = mu + std * z
    t = 1.0 - torch.special.expit(y)
    t_min = 1.0 / (1.0 + math.exp(0.5 * logsnr_max))
    t_max = 1.0 / (1.0 + math.exp(0.5 * logsnr_min))
    t = t.clamp(t_min, t_max)
    sigmas = (1.0 - t).flip(0)
    sigmas = sigmas[:-1].to(dtype=torch.float32, device=device)
    return sigmas


def _resolution_aware_mu(
    height: int,
    width: int,
    base_mu: float,
    base_resolution: tuple[int, int] = (512, 512),
) -> float:
    """Shift the schedule mean as a function of image resolution."""
    num_pixels = height * width
    base_pixels = base_resolution[0] * base_resolution[1]
    return base_mu + 0.5 * math.log(num_pixels / base_pixels)


def _expand_tensor_to_effective_batch(
    tensor: torch.Tensor,
    batch_size: int,
    num_per_prompt: int,
    tensor_name: str | None = None,
) -> torch.Tensor:
    """Replicate `tensor` along dim 0 from `batch_size` (or 1) to `batch_size * num_per_prompt`."""
    target_batch_size = batch_size * num_per_prompt

    if tensor.shape[0] == target_batch_size:
        return tensor

    if tensor.shape[0] == 1:
        repeat_by = target_batch_size
    elif tensor.shape[0] == batch_size:
        repeat_by = num_per_prompt
    else:
        tensor_name = f"`{tensor_name}`" if tensor_name is not None else "Tensor"
        raise ValueError(
            f"{tensor_name} batch size must be 1, `batch_size` ({batch_size}), or "
            f"`batch_size * num_*_per_prompt` ({target_batch_size}), but got {tensor.shape[0]}."
        )

    return torch.repeat_interleave(tensor, repeats=repeat_by, dim=0, output_size=tensor.shape[0] * repeat_by)


def get_ideogram4_post_process_func(od_config: OmniDiffusionConfig):
    """Get post-processing function for Ideogram4 output."""
    if od_config.output_type == "latent":
        return lambda x: x

    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config.get("block_out_channels", [])) - 1)

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


class Ideogram4Pipeline(
    nn.Module,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    """Text-to-image pipeline for Ideogram4.

    Ideogram4 is a flow-matching model trained with asymmetric classifier-free guidance:
    a `transformer` consumes text-conditioned features alongside the image latents,
    while a separate `unconditional_transformer` denoises with zeroed text features.
    """

    _dit_modules: ClassVar[list[str]] = ["transformer", "unconditional_transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.od_config = od_config

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="unconditional_transformer",
                revision=None,
                prefix="unconditional_transformer.",
                fall_back_to_pt=True,
            ),
        ]

        self._execution_device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        # Prefetch all subfolders to avoid race conditions with gated repos
        prefetch_subfolders(model, IDEOGRAM4_SUBFOLDERS, local_files_only=local_files_only)

        # Scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        # VAE
        self.vae = from_pretrained_with_prefetch(
            AutoencoderKLFlux2.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=IDEOGRAM4_SUBFOLDERS,
            local_files_only=local_files_only,
        ).to(self._execution_device)

        # Text encoder (Qwen3-VL)
        # Check if text_encoder uses Ideogram's weight-only FP8 format
        from huggingface_hub import hf_hub_download

        try:
            text_encoder_config_path = hf_hub_download(
                model, "text_encoder/config.json", local_files_only=local_files_only
            )
            with open(text_encoder_config_path) as f:
                text_encoder_config_dict = json.load(f)
            is_ideogram_fp8 = text_encoder_config_dict.get("ideogram_fp8_weight_only", False)
        except Exception:
            is_ideogram_fp8 = False

        if is_ideogram_fp8:
            # Use custom FP8 loading for Ideogram-4's weight-only FP8 format
            self.text_encoder = self._load_text_encoder_ideogram_fp8(
                model, self._execution_device, od_config.dtype, local_files_only
            )
            logger.info("Loaded text_encoder with Ideogram-4 weight-only FP8 format")
        else:
            # Standard loading
            from transformers import Qwen3VLForConditionalGeneration

            self.text_encoder = from_pretrained_with_prefetch(
                Qwen3VLForConditionalGeneration.from_pretrained,
                model,
                subfolder="text_encoder",
                prefetch_list=IDEOGRAM4_SUBFOLDERS,
                local_files_only=local_files_only,
            ).to(self._execution_device)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)

        # Ideogram4 uses head_dim=256 which is not supported by cuDNN attention kernels.
        # Force TORCH_SDPA backend to avoid "No available kernel" errors.
        from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

        if od_config.diffusion_attention_config is None:
            od_config.diffusion_attention_config = AttentionConfig()
        if od_config.diffusion_attention_config.default is None:
            od_config.diffusion_attention_config.default = AttentionSpec(backend="TORCH_SDPA")
            logger.info("Ideogram4: forcing TORCH_SDPA attention backend (cuDNN does not support head_dim=256)")

        # Transformer (conditional)
        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, Ideogram4Transformer2DModel)
        self.transformer = Ideogram4Transformer2DModel(
            quant_config=od_config.quantization_config,
            od_config=od_config,
            **transformer_kwargs,
        )

        # Unconditional transformer
        self.unconditional_transformer = Ideogram4Transformer2DModel(
            quant_config=od_config.quantization_config,
            od_config=od_config,
            **transformer_kwargs,
        )

        # VAE scale factor
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        # Ideogram4 patchifies the VAE output by a factor of 2
        self.patch_size = 2
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * self.patch_size)

        # Latent normalization constants (learned statistics for VAE decode)
        self.register_buffer(
            "latent_shift",
            torch.tensor(LATENT_SHIFT, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "latent_scale",
            torch.tensor(LATENT_SCALE, dtype=torch.float32),
            persistent=False,
        )

        self._guidance_scale = None
        _num_timesteps = None
        self._interrupt = False

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _load_text_encoder_ideogram_fp8(
        self,
        model: str,
        device: torch.device,
        dtype: torch.dtype,
        local_files_only: bool,
    ):
        """Load Qwen3-VL text encoder with Ideogram-4's weight-only FP8 format.

        transformers' from_pretrained can't read Ideogram's float8 layout, so we:
        1. Instantiate the architecture with from_config
        2. Swap the quantized Linears for Ideogram4Fp8Linear
        3. Load the FP8 state dict with assign=True
        """
        from transformers import AutoConfig, AutoModel

        from vllm_omni.diffusion.models.ideogram4.ideogram_fp8 import (
            is_ideogram_fp8_state_dict,
            load_ideogram_fp8_state_dict,
            swap_linears_to_fp8,
        )

        # 1. Load config and create model architecture
        config = AutoConfig.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only, trust_remote_code=True
        )
        text_encoder = AutoModel.from_config(config, trust_remote_code=True)

        # 2. Load state dict
        state_dict = self._load_text_encoder_state_dict(model, local_files_only)

        # 3. Swap Linears to FP8 if needed
        if is_ideogram_fp8_state_dict(state_dict):
            swap_linears_to_fp8(text_encoder, state_dict, compute_dtype=dtype)

        # 4. Load FP8 weights
        load_ideogram_fp8_state_dict(text_encoder, state_dict, device=device, dtype=dtype, assign=True)

        return text_encoder.eval()

    def _load_text_encoder_state_dict(self, model: str, local_files_only: bool) -> dict[str, torch.Tensor]:
        """Load text encoder state dict from safetensors (handles sharded and single file)."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        # Check for sharded checkpoint
        try:
            index_path = hf_hub_download(
                model, "text_encoder/model.safetensors.index.json", local_files_only=local_files_only
            )
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index["weight_map"]

            # Load all shards
            state_dict = {}
            shards = sorted(set(weight_map.values()))
            for shard in shards:
                shard_path = hf_hub_download(model, f"text_encoder/{shard}", local_files_only=local_files_only)
                state_dict.update(load_file(shard_path))
            return state_dict
        except Exception:
            # Single file
            model_path = hf_hub_download(model, "text_encoder/model.safetensors", local_files_only=local_files_only)
            return load_file(model_path)

    @staticmethod
    def _prepare_ids(
        text_lengths: list[int],
        grid_h: int,
        grid_w: int,
        max_text_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the packed `[left-pad][text][image]` layout."""
        batch_size = len(text_lengths)
        num_image_tokens = grid_h * grid_w
        total_seq_len = max_text_tokens + num_image_tokens

        # Image position ids (t=0, h, w); offset keeps them disjoint from text positions.
        h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
        t_idx = torch.zeros_like(h_idx)
        image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

        position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
        segment_ids = torch.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
        indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

        for b, num_text in enumerate(text_lengths):
            offset = max_text_tokens - num_text

            text_pos = torch.arange(num_text)
            text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
            position_ids[b, offset : offset + num_text] = text_pos_3d
            position_ids[b, offset + num_text :] = image_pos

            indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
            indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR

            segment_ids[b, offset : offset + num_text + num_image_tokens] = 1

        return position_ids.to(device), segment_ids.to(device), indicator.to(device)

    @staticmethod
    def _get_text_encoder_hidden_states(
        text_encoder,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_2d: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run the text encoder's decoder layers, returning hidden states from activation layers."""
        language_model = text_encoder.language_model

        inputs_embeds = language_model.embed_tokens(token_ids)

        position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
        text_position_ids = position_ids_4d[0]
        mrope_position_ids = position_ids_4d[1:]

        from transformers.masking_utils import create_causal_mask

        causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

        tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
        captured: dict[int, torch.Tensor] = {}
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if layer_idx in tap_set:
                captured[layer_idx] = hidden_states

        return [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]

    def encode_prompt(
        self,
        prompt: str | list[str],
        grid_h: int,
        grid_w: int,
        max_sequence_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare the conditioning for the packed text+image sequence."""
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        batch_size = len(prompts)
        num_image_tokens = grid_h * grid_w

        # Tokenize each chat-formatted prompt and left-pad to `max_sequence_length`.
        token_ids = torch.zeros(batch_size, max_sequence_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_sequence_length, dtype=torch.long)
        text_position_ids = torch.zeros(batch_size, max_sequence_length, dtype=torch.long)
        text_lengths = []
        for b, text_prompt in enumerate(prompts):
            messages = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            toks = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            n = int(toks.shape[0])
            if n > max_sequence_length:
                raise ValueError(f"prompt has {n} tokens, exceeds max_sequence_length={max_sequence_length}")
            text_lengths.append(n)
            offset = max_sequence_length - n
            token_ids[b, offset:] = toks
            attention_mask[b, offset:] = 1
            text_position_ids[b, offset:] = torch.arange(n)

        te_device = self.text_encoder.device
        token_ids = token_ids.to(te_device)
        attention_mask = attention_mask.to(te_device)
        text_position_ids = text_position_ids.to(te_device)

        # Concatenate the tapped activation-layer hidden states into per-token text features.
        selected = self._get_text_encoder_hidden_states(self.text_encoder, token_ids, attention_mask, text_position_ids)
        text_features = torch.stack(selected, dim=0).permute(1, 2, 3, 0).reshape(batch_size, max_sequence_length, -1)
        text_features = (text_features * attention_mask.to(text_features.dtype).unsqueeze(-1)).to(torch.float32)
        text_features = text_features.to(device)

        position_ids, segment_ids, indicator = self._prepare_ids(
            text_lengths, grid_h, grid_w, max_sequence_length, device
        )

        # Pack the text features into the full sequence; image positions carry no text features.
        image_feature_padding = torch.zeros(
            batch_size, num_image_tokens, text_features.shape[-1], dtype=text_features.dtype, device=device
        )
        prompt_embeds = torch.cat([text_features, image_feature_padding], dim=1)

        return prompt_embeds, position_ids, segment_ids, indicator

    def prepare_latents(
        self,
        batch_size: int,
        num_image_tokens: int,
        latent_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = (batch_size, num_image_tokens, latent_dim)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self) -> float | None:
        return self._guidance_scale

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def interrupt(self) -> bool:
        return self._interrupt

    def forward(
        self,
        req: OmniDiffusionRequest,
        height: int = 2048,
        width: int = 2048,
        num_inference_steps: int = 48,
        guidance_scale: float | None = None,
        guidance_schedule: list[float] | torch.Tensor | None = None,
        mu: float = 0.0,
        std: float = 1.5,
        max_sequence_length: int = 2048,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
    ) -> DiffusionOutput:
        # Extract prompt from request
        if len(req.prompts) > 1:
            logger.warning("Ideogram4 only supports a single prompt. Using the first one.")
        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")

        # Get parameters from request or use defaults
        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        guidance_scale = (
            req.sampling_params.guidance_scale if req.sampling_params.guidance_scale is not None else guidance_scale
        )
        generator = req.sampling_params.generator or generator
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )
        max_sequence_length = req.sampling_params.max_sequence_length or max_sequence_length

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        self._interrupt = False

        # Default guidance schedule (recommended by Ideogram4)
        if guidance_scale is not None and guidance_schedule is None:
            guidance_schedule = [guidance_scale] * num_inference_steps
        elif guidance_scale is None and guidance_schedule is None:
            guidance_schedule = (7.0,) * 45 + (3.0,) * 3

        # 1. Image grid
        grid_h, grid_w = (
            height // (self.vae_scale_factor * self.patch_size),
            width // (self.vae_scale_factor * self.patch_size),
        )
        num_image_tokens = grid_h * grid_w

        # 2. Encode prompts
        llm_features, position_ids, segment_ids, indicator = self.encode_prompt(
            prompt=prompt,
            grid_h=grid_h,
            grid_w=grid_w,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # 3. Replicate for num_images_per_prompt
        llm_features = _expand_tensor_to_effective_batch(llm_features, batch_size, num_images_per_prompt)
        position_ids = _expand_tensor_to_effective_batch(position_ids, batch_size, num_images_per_prompt)
        segment_ids = _expand_tensor_to_effective_batch(segment_ids, batch_size, num_images_per_prompt)
        indicator = _expand_tensor_to_effective_batch(indicator, batch_size, num_images_per_prompt)

        # 4. Unconditional branch
        neg_llm_features = torch.zeros(
            batch_size * num_images_per_prompt,
            num_image_tokens,
            llm_features.shape[-1],
            dtype=llm_features.dtype,
            device=device,
        )
        neg_position_ids = position_ids[:, max_sequence_length:]
        neg_segment_ids = segment_ids[:, max_sequence_length:]
        neg_indicator = indicator[:, max_sequence_length:]

        # 5. Set up sigma schedule
        schedule_mu = _resolution_aware_mu(height=height, width=width, base_mu=mu)
        sigmas = _logit_normal_sigmas(num_inference_steps, schedule_mu, std=std, device=device)
        self.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 6. Guidance weights
        gw = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=device)

        # 7. Prepare latents
        latent_dim = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_image_tokens=num_image_tokens,
            latent_dim=latent_dim,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 8. Padding for text region
        max_text_tokens = max_sequence_length
        text_z_padding = torch.zeros(
            batch_size * num_images_per_prompt,
            max_text_tokens,
            latent_dim,
            dtype=torch.float32,
            device=device,
        )

        # Cast text features to transformer dtype
        llm_features = llm_features.to(self.transformer.dtype)
        neg_llm_features = neg_llm_features.to(self.unconditional_transformer.dtype)

        # 9. Denoising loop
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Map sigma-domain timestep to model time `t` in [0, 1]
                t_model = 1.0 - (t.float() / num_train_timesteps)
                t_model = t_model.expand(batch_size * num_images_per_prompt).to(self.transformer.dtype)

                # Conditional pass
                pos_z = torch.cat([text_z_padding, latents], dim=1).to(self.transformer.dtype)
                pos_out = self.transformer(
                    hidden_states=pos_z,
                    timestep=t_model,
                    encoder_hidden_states=llm_features,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    indicator=indicator,
                    return_dict=False,
                )[0]
                pos_v = pos_out[:, max_text_tokens:].to(torch.float32)

                # Unconditional pass
                neg_v = self.unconditional_transformer(
                    hidden_states=latents.to(self.unconditional_transformer.dtype),
                    timestep=t_model,
                    encoder_hidden_states=neg_llm_features,
                    position_ids=neg_position_ids,
                    segment_ids=neg_segment_ids,
                    indicator=neg_indicator,
                    return_dict=False,
                )[0].to(torch.float32)

                # CFG blending
                self._guidance_scale = guidance_schedule[i]
                gw_i = gw[i]
                v = gw_i * pos_v + (1.0 - gw_i) * neg_v

                latents = self.scheduler.step(-v, t, latents, return_dict=False)[0]

                progress_bar.update()

        # 10. Decode
        if output_type == "latent":
            image = latents
        else:
            z = latents
            # Apply latent normalization (learned statistics, NOT VAE batch-norm)
            z = z * self.latent_scale + self.latent_shift

            # Unpatchify
            patch = self.patch_size
            ae_channels = z.shape[-1] // (patch * patch)
            z = z.view(batch_size * num_images_per_prompt, grid_h, grid_w, patch, patch, ae_channels)
            z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
            z = z.view(batch_size * num_images_per_prompt, ae_channels, grid_h * patch, grid_w * patch)

            decoded = self.vae.decode(z.to(self.vae.dtype), return_dict=False)[0]
            # Return tensor for post_process_func, not PIL images
            image = decoded.float()

        return DiffusionOutput(output=image)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
