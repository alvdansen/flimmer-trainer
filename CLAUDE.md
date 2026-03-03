# Flimmer — Video LoRA Training Toolkit

## What Is Flimmer?

Flimmer is a video LoRA training toolkit for diffusion transformer models (Wan 2.1/2.2 T2V and I2V). It covers the full pipeline from raw footage to trained LoRA checkpoints: scene detection, clip splitting, caption generation, dataset validation, latent pre-encoding, and training with differential MoE expert support.

The data preparation tools are **standalone** — they produce standard formats compatible with any trainer (kohya/sd-scripts, ai-toolkit, etc.).

## Design Philosophy

1. **Video is the target. Everything else is a control signal.** Clear separation between what the model learns to produce (video) and what it learns to obey (text, reference images, depth maps, etc.).
2. **Standalone tools.** Each pipeline stage works independently and outputs standard formats.
3. **Validate before you compute.** Catch config and dataset errors before burning GPU time.
4. **Model-agnostic infrastructure.** Adding a new model means implementing a backend, not rewriting the trainer.

## Package Structure

```
flimmer/
├── config/        # YAML config schemas (Pydantic v2), data + training configs
├── video/         # Video ingestion: probe, scene detect, split, normalize, extract frames
├── caption/       # VLM captioning: Gemini, Replicate, OpenAI-compatible backends
├── dataset/       # Dataset validation, organization, bucketing, manifest generation
├── triage/        # Smart clip selection and filtering
├── encoding/      # Latent pre-encoding: VAE (video+images) and T5 (text) caching
└── training/      # Training loop, LoRA injection, noise scheduling, checkpoints
    └── wan/       # Wan-specific backend (2.1 T2V, 2.2 T2V/I2V, MoE routing)
```

## Code Standards

- Python 3.10+, type hints everywhere
- YAML for all config files (PyYAML + Pydantic validation)
- Every function gets a docstring explaining what it does and WHY
- File paths use `pathlib.Path`, Windows-compatible
- Tests in `tests/` — run with `pytest tests/`

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

Most tests are unit tests with mocked dependencies. No GPU required.

## Key Concepts

- **Data config** (`flimmer_data.yaml`): Defines dataset structure — video specs, caption handling, control signals, quality thresholds.
- **Training config** (`train.yaml`): Model selection, hyperparameters, LoRA settings, MoE expert configuration, checkpoint output.
- **Control signals**: Every conditioning input (text, reference images, depth maps) is a first-class signal with its own prepare → validate → encode pipeline.
- **Differential MoE training**: Wan 2.2's dual-expert architecture gets per-expert hyperparameters (different learning rates, ranks, epoch counts for high-noise vs low-noise experts).

## CLI Entry Points

```bash
python -m flimmer.video       # Video ingestion and processing
python -m flimmer.caption     # Caption generation (if run separately)
python -m flimmer.dataset     # Dataset validation and organization
python -m flimmer.encoding    # Latent pre-encoding (VAE + T5)
python -m flimmer.training    # Training loop
```

## Dependencies

Core: `pydantic`, `pyyaml`
Video: `scenedetect[opencv]`
Caption: `google-genai`, `requests`
Training/Encoding: `torch`, `diffusers`, `transformers`, `peft`, `accelerate`, `safetensors`, `bitsandbytes`

Install what you need: `pip install -e ".[video,caption,training,wan]"` or `pip install -e ".[all]"` for everything.
