"""LTX-2.3 variant defaults for the Flimmer training config system.

Provides default configuration values for LTX model variants. These are
merged into the global VARIANT_DEFAULTS dict by training_loader.py.
"""

from __future__ import annotations

LTX_VARIANT_DEFAULTS: dict[str, dict] = {
    "2.3_t2v": {
        "model": {
            "family": "ltx",
            "variant": "2.3_t2v",
            "in_channels": 128,
            "num_layers": 48,
            "num_train_timesteps": 1000,
            "is_moe": False,
        },
        "lora": {
            "rank": 32,
            "alpha": 32,
        },
        "training": {
            "mixed_precision": "bf16",
        },
    },
}
