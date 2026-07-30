# DJW merge-back — short-gate analysis

## Rule under test

Guards unchanged: same `parentSpanId`, same `speakerCluster`, gap in `[-20, max_gap_ms]`,
mergedDur ≤ `max_merged_ms`, continuous energy (flank−valley ≤ `energy_drop_db`).

**Legacy OR:** merge if `short_piece` (min dur < `short_piece_ms`, default 550) **OR**
`weak_valley` (prom < `weak_dip_db` and gap ≤ `weak_sep_ms`).

**Short-gated:** merge only if **clearly short**:
`min(dur_a, dur_b) < short_piece_ms`
**OR** (when enabled) `min(dur_a, dur_b) / session_median_manual_excl_nv_tag_ms < dur_ratio_max`
(probe 0.65). Weak valley is logged as a secondary reason but is **never sufficient alone**.

Session medians on these kits: St_87 = 810 ms, St_75 = 960 ms
→ 0.65× relative gate ≈ 526 / 624 ms (near legacy 550).

## Hypothesis

Trigger merge-back only when a piece is clearly short (not merely a weak valley) to
**remove the English-kit verbal-all regression** while **keeping St_75 over-split gains**.

## Sweep results

| Policy | Kit / pool | excl-nv IoU≥0.5 | verbal IoU≥0.5 | med durR | tooShort% | nCands | nMerges |
|--------|------------|----------------:|---------------:|---------:|----------:|-------:|--------:|
| `baseline` | 26_07_27__19:53:00 (EN) | 47.7 | 72.9 | 0.65 | 64.6 | 1015 | 0 |
| `baseline` | 26_07_05__00:00:00 (St_75) | 51.4 | 47.6 | 0.61 | 76.1 | 369 | 0 |
| `baseline` | **POOLED** | **49.2** | **65.6** | 0.65 | 69.4 | 1384 | 0 |
| `legacy_or_short550` | EN | 51.4 | 64.5 | 0.93 | 40.4 | 847 | 168 |
| `legacy_or_short550` | St_75 | 56.9 | 52.4 | 0.66 | 67.6 | 319 | 50 |
| `legacy_or_short550` | **POOLED** | **53.6** | **61.1** | 0.93 | 51.8 | 1166 | 218 |
| `short_req_abs350` | EN | 47.7 | 72.9 | 0.66 | 62.6 | 1003 | 12 |
| `short_req_abs350` | St_75 | 51.4 | 47.6 | 0.61 | 74.6 | 364 | 5 |
| `short_req_abs350` | **POOLED** | **49.2** | **65.6** | 0.66 | 67.6 | 1367 | 17 |
| `short_req_abs400` | EN | 46.8 | 71.4 | 0.68 | 58.6 | 944 | 71 |
| `short_req_abs400` | St_75 | 54.2 | 50.0 | 0.64 | 74.6 | 344 | 25 |
| `short_req_abs400` | **POOLED** | **49.7** | **65.3** | 0.68 | 65.3 | 1288 | 96 |
| `short_req_abs450` | EN | 50.5 | 67.5 | 0.81 | 47.5 | 883 | 132 |
| `short_req_abs450` | St_75 | 58.3 | 53.7 | 0.66 | 67.6 | 325 | 44 |
| `short_req_abs450` | **POOLED** | **53.6** | **63.5** | 0.81 | 55.9 | 1208 | 176 |
| `short_req_abs500` | EN | 48.6 | 65.0 | 0.86 | 44.4 | 871 | 144 |
| `short_req_abs500` | St_75 | 58.3 | 53.7 | 0.66 | 67.6 | 322 | 47 |
| `short_req_abs500` | **POOLED** | **52.5** | **61.8** | 0.86 | 54.1 | 1193 | 191 |
| `short_req_abs350+rel0.65` | EN | 49.5 | 64.5 | 0.87 | 42.4 | 864 | 151 |
| `short_req_abs350+rel0.65` | St_75 | 56.9 | 52.4 | 0.66 | 67.6 | 319 | 50 |
| `short_req_abs350+rel0.65` | **POOLED** | **52.5** | **61.1** | 0.87 | 52.9 | 1183 | 201 |

Deltas vs baseline (verbal / excl-nv):

| Policy | EN verbal Δ | EN excl-nv Δ | St_75 excl-nv Δ | Pooled excl-nv Δ | Pooled verbal Δ |
|--------|------------:|-------------:|----------------:|-----------------:|----------------:|
| legacy OR 550 | −8.4 | +3.7 | +5.5 | +4.4 | −4.5 |
| short≤350 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| short≤400 | −1.5 | −0.9 | +2.8 | +0.5 | −0.3 |
| short≤450 | −5.4 | +2.8 | +6.9 | +4.4 | −2.1 |
| short≤500 | −7.9 | +0.9 | +6.9 | +3.3 | −3.8 |
| abs350+rel0.65 | −8.4 | +1.8 | +5.5 | +3.3 | −4.5 |

## Verdict

**Hypothesis: no.** Short-gating alone does not fix English verbal regression while
preserving full St_75-style gains — it is a Pareto knob. Blocking weak-valley-only
merges is not enough; most legacy merges already had `short_piece` (threshold dominates).

**Recommended threshold:** `require_clearly_short=True`, **`short_piece_ms=400`**
when the goal is English verbal protection (verbal −1.5 vs legacy −8.4; St_75 still +2.8 excl-nv).

**Alternate:** **`short_piece_ms=450`** if north-star pooled excl-nv (+4.4, same as legacy)
and stronger St_75 (+6.9) matter more than fully fixing English verbal (−5.4 remaining).

Analysis only — do not wire into production yet. Artifacts:
`tools/analysis/out/merge_back_short_gate.json`.
