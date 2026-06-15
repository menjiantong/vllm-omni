"""Stage input processor for MammothModa2 (AR -> DiT)."""

import logging
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt

logger = logging.getLogger(__name__)


def ar2dit(
    source_outputs: list[Any],
    prompts: OmniTokensPrompt | TextPrompt | list[OmniTokensPrompt | TextPrompt] | None = None,
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Convert AR stage outputs to DiT stage inputs."""
    logger.info("--my--debug-- ar2dit processor called: source_outputs count=%s, prompts type=%s",
                len(source_outputs) if source_outputs else 0, type(prompts).__name__)
    ar_outputs = source_outputs

    # Normalize prompts to list
    if not isinstance(prompts, list):
        prompts = [prompts] if prompts is not None else [{}]

    dit_inputs: list[OmniTokensPrompt] = []
    for idx, (ar_output, prompt) in enumerate(zip(ar_outputs, prompts)):
        logger.info("--my--debug-- ar2dit processing request %d: ar_output type=%s", idx, type(ar_output).__name__)
        addi_info = prompt["additional_information"]
        image_height = addi_info["image_height"][0]
        image_width = addi_info["image_width"][0]
        text_guidance_scale = addi_info["text_guidance_scale"][0]
        cfg_range = addi_info["cfg_range"]
        num_inference_steps = addi_info["num_inference_steps"][0]
        gen_vocab_start_index = addi_info["visual_token_start_id"][0]
        # ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
        visual_ids = addi_info["visual_ids"]

        prompt_token_ids = ar_output.prompt_token_ids
        # exclude the last token because it has no corresponding hidden state
        completion_output = ar_output.outputs[0]
        gen_token_ids = completion_output.cumulative_token_ids[:-1]
        full_token_ids = prompt_token_ids + gen_token_ids

        logger.info("--my--debug-- ar2dit request %d: prompt_tokens=%d, gen_tokens=%d, total_tokens=%d, image_size=%dx%d",
                    idx, len(prompt_token_ids), len(gen_token_ids), len(full_token_ids), image_height, image_width)

        mm_output = getattr(completion_output, "multimodal_output", None)
        if not isinstance(mm_output, dict) or "latent" not in mm_output:
            logger.error("--my--debug-- ar2dit: AR output missing latent! mm_output type=%s, keys=%s",
                        type(mm_output).__name__, list(mm_output.keys()) if isinstance(mm_output, dict) else "N/A")
            raise ValueError(
                "AR stage output missing latent multimodal output. "
                f"request_id={getattr(ar_output, 'request_id', None)}, "
                f"completion_has_mm={hasattr(completion_output, 'multimodal_output')}"
            )
        full_hidden_states = mm_output["latent"]
        hidden_total = int(full_hidden_states.shape[0])
        logger.info("--my--debug-- ar2dit request %d: full_hidden_states.shape=%s, device=%s, dtype=%s",
                    idx, tuple(full_hidden_states.shape), full_hidden_states.device, full_hidden_states.dtype)
        assert hidden_total == len(prompt_token_ids) + len(gen_token_ids), (
            f"Hidden states length mismatch: expected {len(prompt_token_ids) + len(gen_token_ids)}, got {hidden_total}"
        )

        mask_device = full_hidden_states.device
        full_token_ids_t = torch.tensor(full_token_ids, dtype=torch.long, device=mask_device)
        attention_mask = torch.ones_like(full_token_ids_t, dtype=torch.bool)

        pos = torch.arange(full_token_ids_t.shape[0], device=mask_device)
        answer_start_index = len(prompt_token_ids)
        questions_mask = pos < answer_start_index
        answers_mask = ~questions_mask

        gen_token_mask = full_token_ids_t >= gen_vocab_start_index

        visual_token_mask = torch.isin(
            full_token_ids_t,
            torch.tensor(visual_ids, dtype=torch.long, device=mask_device),
        )

        text_condition_token_mask = questions_mask & ~(visual_token_mask | gen_token_mask) & attention_mask
        image_condition_token_mask = answers_mask & gen_token_mask & attention_mask

        text_condition = full_hidden_states[text_condition_token_mask]
        image_condition = full_hidden_states[image_condition_token_mask]

        text_prompt_embeds = text_condition.to(dtype=torch.float32).contiguous()
        image_prompt_embeds = image_condition.to(dtype=torch.float32).contiguous()

        logger.info("--my--debug-- ar2dit request %d: text_prompt_embeds.shape=%s, image_prompt_embeds.shape=%s",
                    idx, tuple(text_prompt_embeds.shape), tuple(image_prompt_embeds.shape))

        additional_information = {
            "text_prompt_embeds": text_prompt_embeds,
            "text_prompt_embeds_shape": list(text_prompt_embeds.shape),
            "image_prompt_embeds": image_prompt_embeds,
            "image_prompt_embeds_shape": list(image_prompt_embeds.shape),
            "image_height": [int(image_height)],
            "image_width": [int(image_width)],
            "text_guidance_scale": [float(text_guidance_scale)],
            "cfg_range": [float(cfg_range[0]), float(cfg_range[1])],
            "num_inference_steps": [int(num_inference_steps)],
        }

        dit_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    logger.info("--my--debug-- ar2dit done: returning %d dit_inputs", len(dit_inputs))
    return dit_inputs
