---
phase: 06-image-training-support
plan: 01
subsystem: encoding
tags: [vae, pil, image-encoding, i2v, self-reference, expand]

# Dependency graph
requires:
  - phase: 02-i2v-backend
    provides: encode_reference_image PIL-based VAE path, I2V hidden state builder
provides:
  - "_encode_image_as_latent() method for PIL-based single image VAE encoding"
  - "auto_self_reference parameter on expand_samples and _expand_image_sample"
  - "I2V context threading in cache CLI (auto_self_reference=True for I2V mode)"
affects: [06-image-training-support, training-loop, cache-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "PIL-based image encoding via _encode_image_as_latent (reuses encode_reference_image pattern)"
    - "Auto self-referencing: image samples use themselves as first-frame conditioning in I2V mode"
    - "I2V detection: reference.source != 'none' triggers auto_self_reference=True"

key-files:
  created:
    - tests/test_vae_encoder_image.py
    - tests/test_expand_auto_self_ref.py
  modified:
    - flimmer/encoding/vae_encoder.py
    - flimmer/encoding/expand.py
    - flimmer/encoding/__main__.py
    - tests/test_encoding_cli.py

key-decisions:
  - "PIL-based image encoding reuses encode_reference_image pattern but keeps batch dim [1,C,1,H//8,W//8]"
  - "Image routing in encode() by extension or target_frames==1, not a separate method call"
  - "auto_self_reference defaults to False for backwards compatibility"

patterns-established:
  - "Image detection in encoder: check IMAGE_EXTENSIONS or target_frames==1 before ffmpeg"
  - "I2V context threading: data config reference.source drives auto_self_reference flag"

requirements-completed: [IMG-01, IMG-02, IMG-04]

# Metrics
duration: 7min
completed: 2026-03-05
---

# Phase 6 Plan 1: Image Encoding + I2V Auto Self-Referencing Summary

**PIL-based VAE encoding for single images with auto self-referencing so I2V mode works on stills without manual reference setup**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-05T04:51:04Z
- **Completed:** 2026-03-05T04:57:55Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- VAE encoder can now encode single images via PIL without invoking ffmpeg
- Image samples with no explicit first_frame reference auto-use themselves as I2V conditioning
- Cache CLI threads I2V context so expand_samples knows when to enable self-referencing
- All 2318 existing tests pass with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: PIL-based image encoding + auto self-referencing** (TDD)
   - `45f3255` (test) - Failing tests for image encoding and auto self-referencing
   - `38d4ffe` (feat) - PIL-based image encoding and auto self-referencing for I2V
2. **Task 2: Thread I2V context through cache CLI** - `2b6d7c4` (feat)

## Files Created/Modified
- `flimmer/encoding/vae_encoder.py` - Added _encode_image_as_latent() and image routing in encode()
- `flimmer/encoding/expand.py` - Added auto_self_reference parameter to _expand_image_sample and expand_samples
- `flimmer/encoding/__main__.py` - I2V detection and auto_self_reference threading in cmd_cache_latents and cmd_info
- `tests/test_vae_encoder_image.py` - 8 tests for image encoding and routing
- `tests/test_expand_auto_self_ref.py` - 7 tests for auto self-referencing
- `tests/test_encoding_cli.py` - 4 tests for I2V context threading in CLI

## Decisions Made
- PIL-based image encoding follows the same pattern as encode_reference_image() but keeps batch dim (returns [1,C,1,H//8,W//8] vs [C,1,H//8,W//8])
- Image detection uses both extension check and target_frames==1 for robustness
- auto_self_reference defaults to False so existing video-only workflows are unaffected
- I2V detection: any reference.source != "none" enables auto self-referencing (covers both "first_frame" and "folder" modes)

## Deviations from Plan

None - plan executed exactly as written. The __main__.py auto_self_reference changes had already been applied in a prior plan execution (06-02), so only the test additions were committed for Task 2.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Image encoding pipeline is complete: discover -> expand (with auto self-ref) -> encode (PIL path) -> cache
- Ready for Plan 02 (image_repeat config and training loop integration) or already covered by 06-02
- Cache pipeline already handles reference entries for self-referenced images via existing manifest/encoding loops

## Self-Check: PASSED

All files exist, all commits verified, all 2318 tests pass.

---
*Phase: 06-image-training-support*
*Completed: 2026-03-05*
