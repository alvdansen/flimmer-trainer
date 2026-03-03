"""Flimmer dataset validation and organization.

Standalone tools for validating complete training datasets: pairing targets
with signals, checking quality, previewing bucketing, generating manifests,
and organizing into trainer-ready directories.
Works with any trainer — musubi-tuner, ai-toolkit, or Flimmer itself.

Quick start:
    python -m flimmer.dataset validate ./my_dataset
    python -m flimmer.dataset organize ./my_dataset -o ./clean -t musubi
"""

from flimmer.dataset.bucketing import (
    BucketAssignment,
    BucketGroup,
    BucketingResult,
    compute_bucket_key,
    preview_bucketing,
)
from flimmer.dataset.discover import (
    detect_structure,
    discover_all_datasets,
    discover_dataset,
    discover_files,
    pair_samples,
)
from flimmer.dataset.errors import (
    DatasetValidationError,
    FlimmerDatasetError,
    OrganizeError,
)
from flimmer.dataset.manifest import (
    build_manifest,
    read_manifest,
    write_manifest,
)
from flimmer.dataset.models import (
    DatasetReport,
    DatasetValidation,
    OrganizeLayout,
    OrganizeResult,
    OrganizedSample,
    SamplePair,
    StructureType,
)
from flimmer.dataset.organize import (
    organize_dataset,
)
from flimmer.dataset.validate import (
    validate_all,
    validate_dataset,
    validate_sample,
)

__all__ = [
    # Errors
    "DatasetValidationError",
    "FlimmerDatasetError",
    "OrganizeError",
    # Models
    "DatasetReport",
    "DatasetValidation",
    "OrganizeLayout",
    "OrganizeResult",
    "OrganizedSample",
    "SamplePair",
    "StructureType",
    # Bucketing
    "BucketAssignment",
    "BucketGroup",
    "BucketingResult",
    "compute_bucket_key",
    "preview_bucketing",
    # Discovery
    "detect_structure",
    "discover_all_datasets",
    "discover_dataset",
    "discover_files",
    "pair_samples",
    # Manifest
    "build_manifest",
    "read_manifest",
    "write_manifest",
    # Organize
    "organize_dataset",
    # Validation
    "validate_all",
    "validate_dataset",
    "validate_sample",
]
