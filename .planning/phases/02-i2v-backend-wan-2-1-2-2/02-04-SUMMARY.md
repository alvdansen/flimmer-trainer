---
phase: 02-i2v-backend-wan-2-1-2-2
plan: 04
subsystem: encoding
tags: [vae, i2v, first-frame, pil, safetensors, latent-encoding]

# Dependency graph
requires:
  - phase: 02-01
    provides: I2V model definitions with first_frame modality
  - phase: 02-02
    provides: first_frame_dropout_rate and training loop I2V support
  - phase: 02-03
    provides: test coverage for I2V registry and backend variants
provides:
  - encode_reference_image() method on WanVaeEncoder for single-image VAE encoding
  - reference_source_path field on CacheEntry for I2V first-frame tracking
  - Reference encoding pass in cache-latents CLI command
  - Full test coverage for reference encoding pipeline
affects: [training, dataset-loading, cache-pipeline]

# Tech tracking
tech-stack:
  added: [PIL/Pillow]
  patterns: [PIL-based single image encoding, reference dedup by source path]

key-files:
  created:
    - tests/test_vae_encoder.py
  modified:
    - flimmer/encoding/models.py
    - flimmer/encoding/cache.py
    - flimmer/encoding/vae_encoder.py
    - flimmer/encoding/__main__.py
    - tests/test_encoding_cache.py

key-decisions:
  - "PIL over ffmpeg for reference images: single image encoding avoids process overhead"
  - "Reference dedup by source_path in cache-latents: prevents re-encoding shared references"
  - "Bucket key parsed for target resolution: reference encoded at video bucket resolution"

patterns-established:
  - "Single image VAE encoding via PIL + numpy + torch pipeline (load -> resize -> normalize -> encode)"
  - "Reference source path tracking parallels caption_source_path pattern"

requirements-completed: [I2V-04]

# Metrics
duration: 3min
completed: 2026-03-04
---

# Phase 02 Plan 04: First-Frame Encoder Summary

**PIL-based VAE encoding for I2V reference images with cache pipeline integration and deduplication**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-04T02:14:40Z
- **Completed:** 2026-03-04T02:18:01Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Added `encode_reference_image()` to WanVaeEncoder that loads a PNG via PIL, resizes to bucket resolution, normalizes to [-1,1], and encodes through the Wan VAE as a single frame
- Populated `reference_source_path` on CacheEntry from `sample.reference` in `build_cache_manifest()` so the encoding loop knows which image to encode
- Added reference encoding pass to `cache-latents` CLI command with deduplication by source path, skip-if-cached logic, and error handling
- Full test coverage: CacheEntry field tests, manifest population tests, encode_reference_image output shape/resize/error tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Add reference encoding to VAE encoder and cache pipeline** - `4f4ef87` (feat)
2. **Task 2: Add tests for reference encoding** - `24714c1` (test)

## Files Created/Modified
- `flimmer/encoding/models.py` - Added `reference_source_path` field to CacheEntry
- `flimmer/encoding/cache.py` - Populated `reference_source_path` in `build_cache_manifest()`
- `flimmer/encoding/vae_encoder.py` - Added `encode_reference_image()` method to WanVaeEncoder
- `flimmer/encoding/__main__.py` - Added reference encoding pass after video latents in `cmd_cache_latents()`
- `tests/test_encoding_cache.py` - Added tests for reference_source_path field and manifest population
- `tests/test_vae_encoder.py` - New test file for encode_reference_image with mocked VAE

## Decisions Made
- Used PIL (Pillow) for image loading instead of ffmpeg -- single image encoding doesn't need process overhead
- Reference deduplication in cache-latents uses source_path set to avoid re-encoding the same image for different frame count expansions
- Target resolution for reference encoding parsed from entry's bucket_key, matching the video resolution

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- I2V first-frame conditioning is complete through the full pipeline: model definition, training loop dropout, test coverage, and now reference encoding
- Phase 02 (I2V Backend Wan 2.1/2.2) is fully complete
- Ready for Phase 3 (local scripts) or Phase 4 (docs)

## Self-Check: PASSED

All 6 files verified present. Both task commits (4f4ef87, 24714c1) verified in git log.

---
*Phase: 02-i2v-backend-wan-2-1-2-2*
*Completed: 2026-03-04*
