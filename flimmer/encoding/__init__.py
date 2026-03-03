"""Flimmer encoding — latent pre-encoding and caching pipeline.

Bridges organized datasets (Phase 4 output) to training-ready cached tensors.
Covers the full pipeline: discovery → expansion → bucketing → encoding → caching.

The GPU-free parts (discovery, expansion, bucketing, cache manifest) work
independently and can be tested without torch or GPU access. The encoder
implementations (VAE, T5) are separate modules that require torch.

Public API::

    from flimmer.encoding import (
        # Discovery
        discover_samples,
        DiscoveredSample,
        SampleRole,
        # Expansion
        expand_samples,
        ExpandedSample,
        # Bucketing
        assign_buckets,
        generate_buckets,
        # Cache
        CacheEntry,
        CacheManifest,
        build_cache_manifest,
        load_cache_manifest,
        save_cache_manifest,
        # Encoder protocol
        ControlEncoder,
        EncoderRegistry,
        # Dataset
        CachedLatentDataset,
        BucketBatchSampler,
    )
"""

from flimmer.encoding.models import (
    CacheEntry,
    CacheManifest,
    DiscoveredSample,
    ExpandedSample,
    SampleRole,
)
from flimmer.encoding.errors import (
    CacheError,
    FlimmerEncodingError,
    EncoderError,
    ExpansionError,
)
from flimmer.encoding.discover import discover_samples
from flimmer.encoding.expand import expand_samples
from flimmer.encoding.bucket import assign_buckets, generate_buckets
from flimmer.encoding.cache import (
    build_cache_manifest,
    load_cache_manifest,
    save_cache_manifest,
)
from flimmer.encoding.encoder import ControlEncoder, EncoderRegistry

__all__ = [
    # Models
    "CacheEntry",
    "CacheManifest",
    "DiscoveredSample",
    "ExpandedSample",
    "SampleRole",
    # Errors
    "CacheError",
    "FlimmerEncodingError",
    "EncoderError",
    "ExpansionError",
    # Discovery
    "discover_samples",
    # Expansion
    "expand_samples",
    # Bucketing
    "assign_buckets",
    "generate_buckets",
    # Cache
    "build_cache_manifest",
    "load_cache_manifest",
    "save_cache_manifest",
    # Encoder protocol
    "ControlEncoder",
    "EncoderRegistry",
]
