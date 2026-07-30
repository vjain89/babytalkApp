# DJW smarter merge-back — scope

**Complexity: M** (standalone policy + offline eval is small; production wiring + knob UX + regressions is medium).

## Problem

DJW + word-split defaults are already best on the IoU sweep (`iou_sweep.json`: manual IoU≥0.5 ≈ 57%). Dominant failure is **over-split / too-short children** (e.g. *tea|cher*), not too-long blobs. Tightening split knobs does not help; we need a **post-cut reunite** step (or external word boundaries such as Whisper).

## What merge-back does

After stage 3b produces children (same `parentSpanId`, same `speakerCluster`, abutting or near cuts after trim):

1. Walk sibling chains in time order.
2. For each adjacent pair, decide **keep-cut** vs **merge** into one candidate.
3. Emit a shorter candidate list for Review / scoring. No change to VAD, diarization, or DJW nuclei finding.

## Smart signals (prototype)

Invert / soften `WORD_SPLIT` rather than re-tune it:

| Signal | Merge when… | Guard |
|--------|-------------|--------|
| Weak valley | Intensity prominence across the cut &lt; `weak_dip_db` (default 5.0; WORD_SPLIT cuts at ≥4.0) | — |
| Short nucleus gap | Piece gap / local sep &lt; `weak_sep_ms` (~280 ms) | — |
| Short piece | Either sibling duration &lt; `short_piece_ms` (~550) | Still need weak/continuous energy |
| Continuous energy | Gap median not ≫ quieter than flanks (`energy_drop_db`) | Reject deep silence |
| Same cluster | `speakerCluster` equal (or both absent) | Never cross speakers |
| Near abut | `0 ≤ gap ≤ max_gap_ms` (~80–120) | Long pause → keep cut |
| Cap | Merged span ≤ `max_merged_ms` (~1800) | Avoid re-gluing multi-word turns |

Default policy: merge only if **near-abut + same cluster + under max duration + continuous energy**, and (**short piece** or **weak valley**).

## Pipeline placement

```
VAD → speechlike → ECAPA → pause-split → DJW children → **merge-back** → speech gate → annotations
```

Prototype runs merge-back **after** the offline SoTA candidate list (post speech-gate annotations) so we can A/B score without touching production. Production would ideally sit **immediately after** `resegment_pieces` (piece dicts) and **before** the speech gate so scores are recomputed on merged spans — optional follow-up.

## Risks

- **Re-gluing multi-word phrases** when valleys between words are shallow (fast speech, IDS coarticulation).
- **Masking real word boundaries** that DJW correctly found with borderline dip/sep.
- **Interaction with force-split** leftovers (`RESEG_TARGET_MS`): merge-back may undo intentional long-span cuts if gap energy looks continuous.
- Metric tradeoff: fewer `candTooShort` / higher IoU can raise `candTooLong2x` if policy is too aggressive.

## Complexity estimate

| Slice | Size | Notes |
|-------|------|--------|
| Offline prototype + metrics (this work) | **S** | One analysis script; reuse `run_vad_on_audio` + `ml_delta.compare` |
| Production hook in `vad_segments` + flags/CLI | **S–M** | Small code; needs Review sanity checks |
| Knob sweep / calibrated thresholds | **M** | Same cost class as `iou_sweep.py` |
| Full alternative: Whisper word timestamps | **M–L** | Model dep, latency, language; orthogonal lever |

## Decision frame vs Whisper

Merge-back is a **cheap acoustic prior** on existing DJW cuts. Whisper is an **ASR word-boundary prior**. From merge-back alone: ship a conservative policy if IoU≥0.5 moves up without ballooning too-long rate; otherwise treat as complementary and wait for Whisper numbers on the same kits before picking a single production path.

## Short-gate follow-up

Offline sweep (`djw_merge_back_eval.py --short-gate-sweep`) tested requiring a
**clearly short** piece (weak valley never sufficient alone). See
`merge_back_short_gate.md` / `.json`.

**Result:** hypothesis **not** jointly satisfied. Best English-protecting gate is
`require_clearly_short` + `short_piece_ms=400` (EN verbal −1.5 pp vs legacy −8.4;
St_75 excl-nv +2.8). For full pooled excl-nv lift use 450 ms (EN verbal still −5.4).
Relative durRatio&lt;0.65 ≈ legacy on these kits. No production wiring yet.
