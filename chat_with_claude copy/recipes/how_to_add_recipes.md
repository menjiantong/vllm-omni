
## Community Recipes To-Do List 

The initial `recipes/` template has been merged #2646. We now welcome community contributions for practical “run model X on hardware Y for task Z” recipes in `vllm-omni`.

Please use the merged template as the starting point and organize recipes by model vendor, for example:

```text
recipes/
  Qwen/
    Qwen3-Omni.md
    Qwen3-TTS.md
    Qwen2.5-Omni.md
  Tencent-Hunyuan/
    HunyuanVideo.md
  GLM/
    GLM-Image.md
  MiMo/
    MiMo-Audio.md
```

Each recipe should include:

- Model name and task
- Tested hardware configuration
- Required environment
- Launch command
- Verification command or expected output
- Important flags or stage configs
- Known limitations
- Links back to canonical `docs/` and runnable `examples/`

## High Priority Starter Recipes

- [ ] `recipes/Qwen/Qwen3-TTS.md`
  - Suggested scope: text-to-speech serving with Qwen3-TTS
  - Suggested hardware sections: CUDA GPU, NPU if available
  - Link to existing Qwen3-TTS docs/examples where possible

- [ ] `recipes/Qwen/Qwen2.5-Omni.md`
  - Suggested scope: omni-modal chat or speech interaction
  - Suggested hardware sections: CUDA GPU, NPU if available
  - Link to existing Qwen2.5-Omni docs/examples where possible

- [ ] `recipes/Tencent-Hunyuan/HunyuanVideo.md`
  - Suggested scope: text-to-video or image-to-video generation
  - Include tested VRAM requirements and performance notes where possible

- [ ] `recipes/GLM/GLM-Image.md`
  - Suggested scope: image generation with GLM-Image
  - Include recommended generation settings and validation output

- [ ] `recipes/MiMo/MiMo-Audio.md`
  - Suggested scope: audio generation or speech-related serving
  - Include model-specific setup and verification notes

## Hardware Coverage We Want

- [ ] NVIDIA CUDA recipes
  - Example targets: A100 80GB, H100, L40S, RTX 4090 where applicable

- [ ] AMD ROCm recipes
  - Include ROCm version, GPU model, and any known caveats

- [ ] Intel XPU recipes
  - Suggested devices from community feedback:
  - Intel Arc Pro B50, 16GB
  - Intel Arc Pro B60, 24GB
  - Intel Arc Pro B70, 32GB

- [ ] Huawei NPU recipes
  - Include CANN/runtime versions and model-specific limitations

## Good First Contributions

- [ ] Add a recipe for a model you have successfully run locally
- [ ] Add another hardware section to an existing recipe
- [ ] Add verification output to an existing recipe
- [ ] Link an existing recipe to the relevant `docs/` and `examples/`
- [ ] Improve clarity around memory usage, flags, or known limitations

## Contribution Guidelines

When opening a recipe PR, please include:

- The exact command you used
- Hardware details
- Software/runtime versions
- Whether the recipe was personally tested
- Any limitations or assumptions
- Links to relevant examples or docs

Recipes should not replace canonical documentation. They should act as practical, community-maintained runbooks that point users to the right docs and runnable examples.

----------------------------------------------------------------------------------

### Motivation.

We’d like to propose a new community-maintained `recipes/` area in `vllm-project/vllm-omni` to answer a recurring user question:

> How do I run model X on hardware Y for task Z?

Today, users often struggle to find the best path for a concrete deployment scenario in vLLM-Omni.

There are a few reasons:

1. We currently do not have a dedicated place for operational runbooks that map a specific model, hardware target, and task to a known-good setup.
2. The current `examples/` layout mixes model-specific and task-specific directories, which can be confusing for discovery.
3. We want to align the user experience with [`vllm-project/recipes`](https://github.com/vllm-project/recipes), which already provides this style of practical guide for upstream vLLM.


### Proposed Change.


Introduce a top-level `recipes/` directory in `vllm-omni` for community-maintained runbooks.

To align with upstream `vllm-project/recipes`, recipes should be grouped by model vendor at the top level, with one Markdown file per model family by default.

Example direction:

```text
recipes/
  Qwen/
    Qwen3-Omni.md
    Qwen3-TTS.md
    Qwen2.5-Omni.md
  Tencent-Hunyuan/
    HunyuanVideo.md
  GLM/
    GLM-Image.md
  MiMo/
    MiMo-Audio.md
```

Within each model doc, we should include multiple hardware-specific sections in the same Markdown file, following the structure used by the upstream DeepSeek guide:
https://github.com/vllm-project/recipes/blob/main/DeepSeek/DeepSeek-V3.md

For example, a single recipe doc could contain sections such as:

- 1x A100 80GB
- 2x L40S
- 4x H100
- ROCm / NPU variants where applicable

Each section would describe:

- supported task(s)
- tested hardware
- required environment
- launch commands
- important flags / stage configs
- verification steps
- known limitations

## Design Principles

- Align the top-level organization with `vllm-project/recipes` where practical, so discovery feels familiar across vLLM and vLLM-Omni.
- Keep one Markdown file per model family by default.
- Keep multiple hardware configurations inside the same recipe document unless the document becomes too large or hard to maintain.
- Use recipes for practical “known-good setup” guidance, not as the canonical source of product documentation.

## Relationship to `examples/`

The current `examples/` folder is still valuable, but it serves a different purpose:

- `examples/`: runnable code and scripts
- `docs/`: canonical documentation
- `recipes/`: practical “known-good setup” guides for concrete user scenarios

One benefit of adding `recipes/` is that it may reduce pressure to make `examples/` itself carry all discovery and onboarding needs.

This also gives us a cleaner answer to users who ask for a concrete deployment path without requiring them to infer it from a mix of task-oriented and model-oriented example folders.

## Scope

This proposal is for community recipes only.

It is not intended to replace:

- canonical product documentation under `docs/`
- source-of-truth runnable examples under `examples/`

Instead, recipes would act as a user-oriented entry point that links back to those canonical materials.

## Open Questions

1. Should recipe ownership be fully community-maintained, or should each recipe have one or two named maintainers?
2. Should recipes live only in the repo at first, or also be surfaced in the documentation site under a Community section?
3. Should we add a lightweight recipe template from the beginning to keep structure consistent?
4. Should any future cleanup of `examples/` be handled in a separate RFC after `recipes/` is established?

## Initial Success Criteria

- A new user can quickly find a concrete recipe for a target model, hardware, and task.
- Recipes follow a consistent structure.
- Recipes are organized by vendor in a way that feels familiar to users of upstream `vllm-project/recipes`.
- Recipes link to `examples/` and `docs/` instead of duplicating canonical content.
- The approach improves user experience without adding confusion about where canonical documentation lives.

## Suggested First Recipes

- `recipes/Qwen/Qwen3-Omni.md`
- `recipes/Qwen/Qwen3-TTS.md`
- `recipes/Qwen/Qwen2.5-Omni.md`
- `recipes/Tencent-Hunyuan/HunyuanVideo.md`
- `recipes/GLM/GLM-Image.md`
- `recipes/MiMo/MiMo-Audio.md`

Feedback welcome on structure, ownership, and how tightly we should align with upstream `vllm-project/recipes`.

### Feedback Period.

_No response_

### CC List.

vllm-omni maintainer team @ywang96 @Gaohan123 @ZJY0516 @princepride .....

### Any Other Things.

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://vllm-omni.readthedocs.io), which can answer lots of frequently asked questions.