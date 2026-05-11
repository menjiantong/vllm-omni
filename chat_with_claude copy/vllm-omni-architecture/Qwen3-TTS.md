# Qwen3-TTS for text-to-speech on 1x A100 80GB

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` / `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` / `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- Task: Text-to-speech synthesis with multiple voice modes
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API
- Maintainer: Community


## When to use this recipe

Use this recipe when you want a known-good starting point for serving Qwen3-TTS models with vLLM-Omni for text-to-speech synthesis. Choose the appropriate model variant based on your use case:

- **CustomVoice**: Use predefined speakers (fastest, easiest)
- **VoiceDesign**: Describe voice style in natural language
- **Base**: Clone voice from reference audio

## References

- Upstream or canonical docs:
  [`docs/user_guide/examples/online_serving/qwen3_tts.md`](../../docs/user_guide/examples/online_serving/qwen3_tts.md)
- Related example under `examples/`:
  [`examples/online_serving/qwen3_tts/README.md`](../../examples/online_serving/qwen3_tts/README.md)
- Related issue or discussion:
  [RFC: add recipes folder](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware Support

## GPU

### 1x A100 80GB

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA environment with an A100 80 GB GPU
- vLLM version: 0.19.0
- vLLM-Omni version or commit: 0.19.0rc1

#### Command

**CustomVoice (predefined speakers):**

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --omni \
    --port 8091
```

**VoiceDesign (natural language voice description):**

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --omni \
    --port 8091
```

**Base (voice cloning):**

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --omni \
    --port 8091
```

For synchronous end-to-end mode (lower latency streaming), disable async chunk:

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --omni \
    --port 8091 \
    --no-async-chunk
```

#### Verification

**Verification for CustomVoice model**

Generate voice using a predefined speaker

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "vivian",
        "language": "English"
    }' --output output.wav
```

List available voices:

```bash
curl http://localhost:8091/v1/audio/voices
```

**Verification for VoiceDesign model**
```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "task_type":"VoiceDesign",
        "input": "哥哥，你回来啦",
        "instructions": "体现撒娇稚嫩的萝莉女声，音调偏高"
    }' --output output.wav
```


**Verification for Base (voice cloning) model**
```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "task_type": "Base",
        "input": "Hello, this is a cloned voice",
        "ref_audio": "'$(base64 -w 0 /temp/reference.wav)'",
        "ref_text": "Original transcript of the reference audio",
        "file_name": "reference.wav"
    }' --output output.wav
```

#### Notes

- Memory usage: The 1.7B model fits comfortably on 80GB; smaller 0.6B variant available for constrained environments.
- Key flags: `--omni` is required. Async chunk is enabled by default for lower TTFA.
- Streaming: Use `stream=true` with `response_format="pcm"` for real-time audio streaming.
- Voice cloning (Base model): Requires `ref_audio` and `ref_text` parameters in the request.
- Known limitations: Batch processing is not yet optimized for online serving.

