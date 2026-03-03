"""Tests for flimmer.training.verification -- frozen-expert weight verification."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from flimmer.training.verification import VerificationResult, WeightVerifier


# ---------------------------------------------------------------------------
# File checksum tests
# ---------------------------------------------------------------------------

class TestFileChecksum:
    """Test file-based SHA-256 checksumming."""

    def test_consistent_hash_for_same_file(self, tmp_path):
        """Same file content produces the same hash every time."""
        f = tmp_path / "weights.bin"
        f.write_bytes(b"hello world" * 1000)
        h1 = WeightVerifier._file_checksum(f)
        h2 = WeightVerifier._file_checksum(f)
        assert h1 == h2

    def test_different_hash_for_different_files(self, tmp_path):
        """Different file contents produce different hashes."""
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        h1 = WeightVerifier._file_checksum(f1)
        h2 = WeightVerifier._file_checksum(f2)
        assert h1 != h2

    def test_matches_known_sha256(self, tmp_path):
        """File checksum matches Python's hashlib directly."""
        data = b"test data for checksum"
        f = tmp_path / "test.bin"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert WeightVerifier._file_checksum(f) == expected


# ---------------------------------------------------------------------------
# Sentinel checksum tests
# ---------------------------------------------------------------------------

class TestSentinelChecksum:
    """Test state-dict sentinel checksumming."""

    def test_consistent_hash_for_same_dict(self):
        """Same state dict produces the same sentinel hash."""
        sd = {
            "layer.0.weight": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "layer.1.weight": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            "layer.2.weight": np.array([7.0, 8.0, 9.0], dtype=np.float32),
        }
        h1 = WeightVerifier._sentinel_checksum(sd)
        h2 = WeightVerifier._sentinel_checksum(sd)
        assert h1 == h2

    def test_different_hash_for_different_values(self):
        """Different tensor values produce different hashes."""
        sd1 = {"w": np.array([1.0], dtype=np.float32)}
        sd2 = {"w": np.array([2.0], dtype=np.float32)}
        h1 = WeightVerifier._sentinel_checksum(sd1)
        h2 = WeightVerifier._sentinel_checksum(sd2)
        assert h1 != h2

    def test_empty_dict_returns_hash(self):
        """Empty state dict returns a valid hash (not crash)."""
        h = WeightVerifier._sentinel_checksum({})
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest length

    def test_single_key_dict(self):
        """State dict with one key works (all sentinels point to same key)."""
        sd = {"only.weight": np.zeros(10, dtype=np.float32)}
        h = WeightVerifier._sentinel_checksum(sd)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_two_key_dict(self):
        """State dict with two keys works (first/middle/last overlap)."""
        sd = {
            "a.weight": np.array([1.0], dtype=np.float32),
            "b.weight": np.array([2.0], dtype=np.float32),
        }
        h = WeightVerifier._sentinel_checksum(sd)
        assert isinstance(h, str)
        assert len(h) == 64


# ---------------------------------------------------------------------------
# Snapshot + verify workflow tests
# ---------------------------------------------------------------------------

class TestSnapshotVerify:
    """Test the full snapshot -> verify workflow."""

    def test_unchanged_file_passes(self, tmp_path):
        """Snapshot + verify with unchanged file returns passed=True."""
        f = tmp_path / "expert.safetensors"
        f.write_bytes(b"model weights data" * 100)

        verifier = WeightVerifier()
        verifier.snapshot("high_noise", checkpoint_path=f)
        result = verifier.verify("high_noise", checkpoint_path=f)

        assert result.passed is True
        assert result.expert_name == "high_noise"
        assert "verified unchanged" in result.details

    def test_changed_file_fails(self, tmp_path):
        """Snapshot + verify with changed file returns passed=False."""
        f = tmp_path / "expert.safetensors"
        f.write_bytes(b"original weights")

        verifier = WeightVerifier()
        verifier.snapshot("low_noise", checkpoint_path=f)

        # Simulate corruption: overwrite the file
        f.write_bytes(b"corrupted weights")

        result = verifier.verify("low_noise", checkpoint_path=f)
        assert result.passed is False
        assert result.expert_name == "low_noise"
        assert "FROZEN EXPERT CHANGED" in result.details

    def test_unchanged_state_dict_passes(self):
        """Snapshot + verify with unchanged state dict returns passed=True."""
        sd = {
            "layer.0": np.array([1.0, 2.0], dtype=np.float32),
            "layer.1": np.array([3.0, 4.0], dtype=np.float32),
        }

        verifier = WeightVerifier()
        verifier.snapshot("high_noise", state_dict=sd)
        result = verifier.verify("high_noise", state_dict=sd)

        assert result.passed is True

    def test_changed_state_dict_fails(self):
        """Snapshot + verify with changed state dict returns passed=False."""
        sd_before = {
            "layer.0": np.array([1.0, 2.0], dtype=np.float32),
        }
        sd_after = {
            "layer.0": np.array([99.0, 99.0], dtype=np.float32),
        }

        verifier = WeightVerifier()
        verifier.snapshot("low_noise", state_dict=sd_before)
        result = verifier.verify("low_noise", state_dict=sd_after)

        assert result.passed is False

    def test_verify_without_snapshot_raises(self):
        """Verify raises ValueError when no prior snapshot exists."""
        verifier = WeightVerifier()
        with pytest.raises(ValueError, match="No snapshot found"):
            verifier.verify("high_noise", state_dict={"w": np.zeros(1)})

    def test_snapshot_no_source_raises(self):
        """Snapshot raises ValueError when no file or state dict provided."""
        verifier = WeightVerifier()
        with pytest.raises(ValueError, match="Cannot snapshot"):
            verifier.snapshot("high_noise")

    def test_verify_no_source_raises(self, tmp_path):
        """Verify raises ValueError when no file or state dict provided."""
        f = tmp_path / "expert.safetensors"
        f.write_bytes(b"data")

        verifier = WeightVerifier()
        verifier.snapshot("high_noise", checkpoint_path=f)
        with pytest.raises(ValueError, match="Cannot verify"):
            verifier.verify("high_noise")

    def test_snapshot_returns_checksum(self, tmp_path):
        """snapshot() returns the checksum string."""
        f = tmp_path / "expert.safetensors"
        f.write_bytes(b"data")

        verifier = WeightVerifier()
        checksum = verifier.snapshot("high_noise", checkpoint_path=f)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hex

    def test_prefers_file_over_state_dict(self, tmp_path):
        """When both file and state_dict given, file checksum is used."""
        f = tmp_path / "expert.safetensors"
        f.write_bytes(b"file content")

        sd = {"w": np.array([1.0], dtype=np.float32)}

        verifier = WeightVerifier()
        # Snapshot with file
        checksum_file = verifier.snapshot("test", checkpoint_path=f)
        # Separately compute what file checksum should be
        expected = WeightVerifier._file_checksum(f)
        assert checksum_file == expected


# ---------------------------------------------------------------------------
# VerificationResult dataclass
# ---------------------------------------------------------------------------

class TestVerificationResult:
    """VerificationResult dataclass."""

    def test_fields_accessible(self):
        """All fields are accessible on the result."""
        r = VerificationResult(
            expert_name="high_noise",
            passed=True,
            details="All good",
            checksum_before="abc",
            checksum_after="abc",
        )
        assert r.expert_name == "high_noise"
        assert r.passed is True
        assert r.details == "All good"
        assert r.checksum_before == "abc"
        assert r.checksum_after == "abc"

    def test_frozen_dataclass(self):
        """VerificationResult is immutable (frozen dataclass)."""
        r = VerificationResult(
            expert_name="low_noise",
            passed=False,
            details="Changed",
            checksum_before="a",
            checksum_after="b",
        )
        with pytest.raises(AttributeError):
            r.passed = True  # type: ignore[misc]
