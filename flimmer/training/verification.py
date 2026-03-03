"""Frozen-expert weight verification for MoE training integrity.

During differential MoE training, one expert trains while the other stays
frozen. This module verifies that the frozen expert's weights are truly
unchanged after each phase -- catching silent bugs that would corrupt
the differential training thesis.

Strategy: Checksum the frozen expert's checkpoint FILE on disk (fast, no
GPU memory needed). The frozen expert stays on disk during the other
expert's training phase -- the file checksum is sufficient.

If the frozen expert is not on disk (e.g., preloaded in memory), fall
back to checksumming a small number of sentinel parameters from the
state dict (first, last, and one middle tensor). This avoids the 30+
second full state dict checksum on 14B models.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    """Result of a frozen-expert weight verification check.

    Attributes:
        expert_name: Which expert was verified ('high_noise' or 'low_noise').
        passed: True if checksums match (weights unchanged).
        details: Human-readable description of the result.
        checksum_before: Checksum taken before the phase.
        checksum_after: Checksum taken after the phase.
    """
    expert_name: str
    passed: bool
    details: str
    checksum_before: str
    checksum_after: str


class WeightVerifier:
    """Verifies frozen expert weights are unchanged after training phases.

    During MoE training, one expert trains while the other is frozen.
    This class takes a checksum snapshot of the frozen expert before
    training, then verifies the checksum matches after training.

    Two checksum strategies:
    1. File-based (primary): SHA-256 of the checkpoint file on disk.
       Fast, no GPU memory needed.
    2. Sentinel-based (fallback): SHA-256 of the first, middle, and last
       tensors in a state dict. Used when no file is available.

    Usage:
        verifier = WeightVerifier()
        verifier.snapshot("high_noise", checkpoint_path=Path("..."))
        # ... train low_noise expert ...
        result = verifier.verify("high_noise", checkpoint_path=Path("..."))
        # result.passed = True/False, result.details = "..."
    """

    def __init__(self) -> None:
        self._snapshots: dict[str, str] = {}  # expert_name -> checksum

    def snapshot(
        self,
        expert_name: str,
        checkpoint_path: Path | None = None,
        state_dict: dict[str, Any] | None = None,
    ) -> str:
        """Take a checksum snapshot of an expert's weights.

        Prefers checkpoint_path (fast file hash). Falls back to
        state_dict sentinel hashing if no file available.

        Args:
            expert_name: Name of the expert ('high_noise' or 'low_noise').
            checkpoint_path: Path to the expert's checkpoint file on disk.
            state_dict: Expert's state dict (fallback if no file).

        Returns:
            The checksum string.

        Raises:
            ValueError: If neither checkpoint_path nor state_dict provided.
        """
        if checkpoint_path is not None and checkpoint_path.is_file():
            checksum = self._file_checksum(checkpoint_path)
        elif state_dict is not None:
            checksum = self._sentinel_checksum(state_dict)
        else:
            raise ValueError(
                f"Cannot snapshot expert '{expert_name}': no checkpoint file "
                f"or state dict provided. Ensure the expert has been saved "
                f"to disk before snapshotting."
            )

        self._snapshots[expert_name] = checksum
        return checksum

    def verify(
        self,
        expert_name: str,
        checkpoint_path: Path | None = None,
        state_dict: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Verify expert weights match the snapshot.

        Computes a new checksum using the same strategy (file or sentinel)
        and compares against the stored snapshot.

        Args:
            expert_name: Name of the expert to verify.
            checkpoint_path: Path to the expert's checkpoint file on disk.
            state_dict: Expert's state dict (fallback if no file).

        Returns:
            VerificationResult with passed=True if checksums match.

        Raises:
            ValueError: If no prior snapshot exists for this expert.
            ValueError: If neither checkpoint_path nor state_dict provided.
        """
        if expert_name not in self._snapshots:
            raise ValueError(
                f"No snapshot found for expert '{expert_name}'. "
                f"Call snapshot() before verify()."
            )

        checksum_before = self._snapshots[expert_name]

        if checkpoint_path is not None and checkpoint_path.is_file():
            checksum_after = self._file_checksum(checkpoint_path)
        elif state_dict is not None:
            checksum_after = self._sentinel_checksum(state_dict)
        else:
            raise ValueError(
                f"Cannot verify expert '{expert_name}': no checkpoint file "
                f"or state dict provided."
            )

        passed = checksum_before == checksum_after

        if passed:
            details = (
                f"Frozen expert '{expert_name}' weights verified unchanged. "
                f"Checksum: {checksum_before[:16]}..."
            )
        else:
            details = (
                f"FROZEN EXPERT CHANGED: '{expert_name}' weights differ! "
                f"Before: {checksum_before[:16]}... "
                f"After: {checksum_after[:16]}..."
            )

        return VerificationResult(
            expert_name=expert_name,
            passed=passed,
            details=details,
            checksum_before=checksum_before,
            checksum_after=checksum_after,
        )

    @staticmethod
    def _file_checksum(path: Path) -> str:
        """SHA-256 of a file's contents. Read in 64KB chunks for efficiency.

        This is the primary checksum strategy -- fast and requires no
        GPU memory since we're reading the file from disk.

        Args:
            path: Path to the file to checksum.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)  # 64KB chunks
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _sentinel_checksum(state_dict: dict[str, Any]) -> str:
        """SHA-256 of sentinel tensors (first, middle, last) from state dict.

        This is MUCH faster than checksumming the entire state dict.
        Only used when no checkpoint file is available (e.g., expert
        preloaded in memory).

        Picks three tensors: first key, middle key, and last key from
        the sorted keys. Hashes their raw bytes together.

        Args:
            state_dict: Model state dict (keys -> tensors).

        Returns:
            Hex-encoded SHA-256 digest of the sentinel tensors.
        """
        if not state_dict:
            return hashlib.sha256(b"empty").hexdigest()

        keys = sorted(state_dict.keys())
        # Pick sentinel indices: first, middle, last
        indices = [0, len(keys) // 2, len(keys) - 1]
        # Deduplicate (important when len(keys) <= 2)
        indices = sorted(set(indices))

        sha256 = hashlib.sha256()
        for idx in indices:
            key = keys[idx]
            tensor = state_dict[key]

            # Get raw bytes from the tensor. Support both torch.Tensor
            # and numpy arrays (for testing without GPU).
            if hasattr(tensor, "cpu"):
                # torch.Tensor -> move to CPU, convert to numpy bytes
                raw = tensor.cpu().detach().numpy().tobytes()
            elif hasattr(tensor, "tobytes"):
                # numpy array
                raw = tensor.tobytes()
            elif isinstance(tensor, bytes):
                raw = tensor
            else:
                # Fallback: use repr as bytes
                raw = repr(tensor).encode()

            sha256.update(key.encode())
            sha256.update(raw)

        return sha256.hexdigest()
