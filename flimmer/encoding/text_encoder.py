"""T5 text encoder for caption encoding.

Encodes text captions through the T5 text encoder (typically UMT5-XXL
for Wan models) to produce text embeddings for training. Text encoding
is run separately from VAE encoding so they never compete for VRAM.

The text encoder produces:
    - text_emb: [seq_len, hidden_dim] tensor of token embeddings
    - text_mask: [seq_len] attention mask (1 = real token, 0 = padding)

These are cached per-stem (not per-expansion) because the caption
is the same regardless of how many frames we take from the video.

This module requires torch and transformers. It is NOT imported by
default — only loaded when cache-text is actually run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flimmer.encoding.errors import EncoderError


class T5TextEncoder:
    """T5 text encoder implementing the ControlEncoder protocol.

    Loads the T5 model and tokenizer from a HuggingFace model ID or
    local path. Manages GPU memory — loaded lazily on first encode(),
    released on cleanup().

    Usage::

        encoder = T5TextEncoder(
            model_id="google/umt5-xxl",
            dtype="bf16",
            max_length=512,
        )
        result = encoder.encode("path/to/caption.txt")
        text_emb = result["text_emb"]    # torch.Tensor [seq_len, dim]
        text_mask = result["text_mask"]  # torch.Tensor [seq_len]
        encoder.cleanup()
    """

    def __init__(
        self,
        model_id: str = "google/umt5-xxl",
        dtype: str = "bf16",
        device: str = "cuda",
        max_length: int = 512,
        t5_path: str | None = None,
    ) -> None:
        """Initialize the text encoder (model loaded lazily).

        Args:
            model_id: HuggingFace model ID or local path for the T5 encoder.
                For Wan models, the text encoder subfolder is used.
            dtype: Tensor dtype ('bf16', 'fp16', 'fp32').
            device: Target device ('cuda', 'cpu').
            max_length: Maximum token sequence length.
            t5_path: Optional path to a single T5 weights file (.pth or
                .safetensors). When set, the tokenizer is loaded from
                HuggingFace 'google/umt5-xxl' (always small), and the
                model weights from this file.
        """
        self._model_id = model_id
        self._dtype_str = dtype
        self._device = device
        self._max_length = max_length
        self._t5_path = t5_path
        self._model = None
        self._tokenizer = None

    @property
    def encoder_id(self) -> str:
        """Unique identifier for this encoder configuration."""
        return f"{self._model_id}/{self._dtype_str}/max{self._max_length}"

    @property
    def signal_type(self) -> str:
        """Signal type: 'text' for text encodings."""
        return "text"

    def _ensure_model(self) -> None:
        """Load the T5 model and tokenizer if not already loaded.

        Tries to load from a local Wan model directory first (looking
        for text_encoder/ and tokenizer/ subfolders), then falls back
        to a standalone HuggingFace model ID.

        Raises:
            EncoderError: If loading fails.
        """
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, UMT5EncoderModel
        except ImportError as e:
            raise EncoderError(
                "t5_text",
                f"Required packages not installed: {e}\n"
                f"Install with: pip install 'flimmer[wan]'"
            )

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        dtype = dtype_map.get(self._dtype_str, torch.bfloat16)

        try:
            # Priority 1: Direct path to a single weights file
            if self._t5_path is not None:
                t5_file = Path(self._t5_path)
                if not t5_file.is_file():
                    raise EncoderError(
                        "t5_text",
                        f"T5 weights file not found: '{self._t5_path}'"
                    )

                # Tokenizer always from HuggingFace — it's tiny
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "google/umt5-xxl",
                )

                # Load weights from the single file
                if t5_file.suffix == ".pth":
                    state_dict = torch.load(
                        str(t5_file),
                        map_location="cpu",
                        weights_only=True,
                    )
                elif t5_file.suffix == ".safetensors":
                    from safetensors.torch import load_file
                    state_dict = load_file(str(t5_file))
                else:
                    raise EncoderError(
                        "t5_text",
                        f"Unsupported T5 file format: '{t5_file.suffix}'. "
                        f"Expected .pth or .safetensors."
                    )

                # Create model from config, then load weights manually.
                # Newer transformers don't allow state_dict + model name together,
                # and AutoConfig can't resolve umt5-xxl on some versions.
                from transformers import UMT5Config
                # Try local cache first (pre-cached during setup), fall back to download
                try:
                    t5_config = UMT5Config.from_pretrained(
                        "google/umt5-xxl", local_files_only=True
                    )
                except OSError:
                    t5_config = UMT5Config.from_pretrained("google/umt5-xxl")
                self._model = UMT5EncoderModel(t5_config).to(dtype=dtype)
                self._model.load_state_dict(state_dict, strict=False)
                # Fix embed_tokens weight tying (see _fix_t5_embed_tokens)
                self._fix_t5_embed_tokens(self._model)

            # Priority 2: Wan model directory with text_encoder subfolder
            elif (model_path := Path(self._model_id)).is_dir() and (
                model_path / "text_encoder"
            ).is_dir():
                self._model = UMT5EncoderModel.from_pretrained(
                    str(model_path),
                    subfolder="text_encoder",
                    torch_dtype=dtype,
                )
                # Fix embed_tokens weight tying (see _fix_t5_embed_tokens)
                self._fix_t5_embed_tokens(self._model)
                tokenizer_path = model_path / "tokenizer"
                if tokenizer_path.is_dir():
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        str(tokenizer_path),
                    )
                else:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        str(model_path),
                        subfolder="tokenizer",
                    )

            # Priority 3: HuggingFace model ID — try with text_encoder
            # subfolder first (Wan pipeline repos), then standalone T5 model
            else:
                try:
                    # Wan pipeline repos have a text_encoder subfolder
                    self._model = UMT5EncoderModel.from_pretrained(
                        self._model_id,
                        subfolder="text_encoder",
                        torch_dtype=dtype,
                    )
                    # Tokenizer from tokenizer subfolder
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self._model_id,
                        subfolder="tokenizer",
                    )
                except (OSError, ValueError):
                    # Fall back to standalone T5 model ID (e.g. "google/umt5-xxl")
                    self._model = UMT5EncoderModel.from_pretrained(
                        self._model_id,
                        torch_dtype=dtype,
                    )
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self._model_id,
                    )
                # Fix embed_tokens weight tying (see _fix_t5_embed_tokens)
                self._fix_t5_embed_tokens(self._model)

            self._model.to(self._device)
            self._model.eval()

        except EncoderError:
            raise
        except Exception as e:
            source = self._t5_path if self._t5_path else self._model_id
            raise EncoderError(
                "t5_text",
                f"Failed to load T5 encoder from '{source}': {e}\n"
                f"Check that the path or model ID is correct."
            ) from e

    @staticmethod
    def _fix_t5_embed_tokens(model: Any) -> None:
        """Fix T5 embed_tokens weight tying bug.

        The Wan-AI T5 checkpoint stores the token embedding as
        "shared.weight", but UMT5EncoderModel keeps a separate
        "encoder.embed_tokens.weight" that should be tied to it.
        Neither load_state_dict(strict=False) nor from_pretrained()
        performs this tying automatically.

        When loaded via load_state_dict (from .pth), embed_tokens is
        left as all zeros. When loaded via from_pretrained, embed_tokens
        is randomly initialized. In both cases, the weight is WRONG --
        it should be identical to shared.weight.

        WHY: Without this fix, T5 produces embeddings but ignores
        most of the prompt content -- the model generates coherent
        but hallucinated output that doesn't follow the text.

        Fix: always copy shared.weight to encoder.embed_tokens if they
        differ. The shared.weight always has the correct trained values.
        Safe no-op if weights are already identical.
        """
        if (
            hasattr(model, "shared")
            and hasattr(model, "encoder")
            and hasattr(model.encoder, "embed_tokens")
        ):
            import torch
            shared = model.shared.weight
            embed = model.encoder.embed_tokens.weight
            # Fix if embed_tokens differs from shared (wrong initialization)
            # shared.weight always has the correct trained values
            if not torch.equal(shared, embed):
                model.encoder.embed_tokens.weight = shared

    def encode(self, input_path: str, **kwargs: Any) -> dict[str, Any]:
        """Encode a caption text file through T5.

        Args:
            input_path: Path to a .txt file containing the caption.

        Returns:
            Dict with 'text_emb' and 'text_mask' tensor keys.
            text_emb shape: [seq_len, hidden_dim] (squeezed from batch)
            text_mask shape: [seq_len]

        Raises:
            EncoderError: If encoding fails.
        """
        self._ensure_model()

        try:
            import torch
        except ImportError as e:
            raise EncoderError("t5_text", f"torch required: {e}")

        # Step 1: Read caption text
        caption_path = Path(input_path)
        if not caption_path.is_file():
            raise EncoderError(
                "t5_text",
                f"Caption file not found: '{input_path}'"
            )

        try:
            caption = caption_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            raise EncoderError(
                "t5_text",
                f"Failed to read caption from '{input_path}': {e}"
            )

        if not caption:
            # Empty caption — return zero embeddings
            # The training loop handles this via caption dropout
            caption = ""

        try:
            # Step 2: Tokenize
            tokens = self._tokenizer(
                caption,
                max_length=self._max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].to(self._device)
            attention_mask = tokens["attention_mask"].to(self._device)

            # Step 3: Encode
            with torch.no_grad():
                output = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # output.last_hidden_state: [1, seq_len, hidden_dim]
                text_emb = output.last_hidden_state.squeeze(0).cpu()
                text_mask = attention_mask.squeeze(0).cpu()

            return {
                "text_emb": text_emb,    # [seq_len, hidden_dim]
                "text_mask": text_mask,   # [seq_len]
            }

        except EncoderError:
            raise
        except Exception as e:
            raise EncoderError(
                "t5_text",
                f"T5 encoding failed for '{input_path}': {e}"
            ) from e

    def cleanup(self) -> None:
        """Release GPU memory.

        Uses gc.collect() before torch.cuda.empty_cache() to ensure
        Python objects holding CUDA tensor references are freed first.
        Without gc.collect(), empty_cache() cannot reclaim memory held
        by unreachable but not-yet-collected Python objects.
        """
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
