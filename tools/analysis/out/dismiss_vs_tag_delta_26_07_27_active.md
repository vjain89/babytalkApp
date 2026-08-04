# Dismissed ML vs tags — raw vs active-audio — `26_07_27__19:53:00`

**Kit:** `2026-07-28T04-26-04-490Z_1785211756660_353_Mississippi_St_87`  
**Question:** When a reviewer dismisses a Find-speech box and draws a tag nearby, how much of the mismatch is **real speech** vs **silence padding** inside the boxes?

## Honesty first

The earlier report [`dismiss_vs_tag_delta_26_07_27.md`](dismiss_vs_tag_delta_26_07_27.md) measured only **raw timestamps** (box start/end as stored). A tag that includes 1.0 s of trailing silence looked “1.0 s longer” than a tight ML box — same as if the ML had missed 1.0 s of speech. This follow-up trims each box to **active audio** (energy above a noise floor) and re-scores.

## How active-audio windows are built

- Load `audio.wav` once; frame RMS every 10 ms (30 ms frames).
- Global noise floor ≈ -57.52 dB (25th percentile).
- A frame is **active** if it is ≥ floor + 6.0 dB, **or** within 25.0 dB of the loudest frame in that box (and still above floor+3 dB).
- Active start/end = first→last active frame ± 40.0 ms, **clipped to the raw box** (we only shrink).
- Then recompute Δstart / Δend / durRatio / IoU on active windows (same classifiers as raw).

## Counts (same pairing as raw)

| | n |
|---|---:|
| Dismissed candidates | 311 |
| Matched to hand-drawn tag | 190 |
| Unique manual tags matched | 120 |
| Junk dismissals (no tag) | 118 |
| Tags trimmed (had silence lead/trail) | 42.6% of pairs |
| Cands trimmed | 22.1% of pairs |

## Side-by-side: raw vs active (hand-drawn rescues)

### Error modes

| Mode | Raw % | Active % |
|---|---:|---:|
| `too_short` | 81.1 | 81.1 |
| `mixed_boundary` | 6.3 | 4.7 |
| `too_long` | 4.2 | 3.7 |
| `shifted` | 3.2 | 3.2 |
| `late_start` | 2.6 | 4.2 |
| `ok` | 1.6 | 2.1 |
| `early_end` | 1.1 | 1.1 |

### Boundary medians

| Metric | Raw median | Active median |
|---|---:|---:|
| start delta (tag−cand) ms | **-230.0** | **-230.0** |
| end delta (tag−cand) ms | **150.0** | **140.0** |
| dur ratio cand/tag | **0.506** | **0.512** |
| IoU | **0.44** | **0.454** |
| IoU ≥ 0.5 (%) | 35.3 | 38.9 |

### Per-tag (best IoU) too_short

- Raw deduped too_short: **75.0%**  
- Active deduped too_short: **75.0%**

## Silence inside the boxes

How much of each raw box was trimmed away as non-active?

**Bottom line on silence:** on this kit, median silence lead/trail is **0 ms** and median active fraction is **~1.0**. Some tags trim tens of ms at the edges (p90 lead ~60 ms), but that does **not** explain the hundreds-of-ms under-cover. Raw “too short” was already mostly speech.

| | p25 | median | p75 | p90 |
|---|---:|---:|---:|---:|
| tag silence lead (ms) | 0.0 | **0.0** | 27.5 | 60.0 |
| tag silence trail (ms) | 0.0 | **0.0** | 17.5 | 30.0 |
| cand silence lead (ms) | 0.0 | **0.0** | 0.0 | 0.0 |
| cand silence trail (ms) | 0.0 | **0.0** | 0.0 | 20.0 |
| tag active fraction | 0.968 | **1.0** | 1.0 | 1.0 |
| cand active fraction | 1.0 | **1.0** | 1.0 | 1.0 |
| Δ|start| shrink after trim (ms) | 0.0 | **0.0** | 20.0 | 51.0 |
| Δ|end| shrink after trim (ms) | 0.0 | **0.0** | 20.0 | 40.0 |

### Active-mode detail

| Mode | n | % | Meaning |
|---|---:|---:|---|
| `too_short` | 154 | 81.1 | ML box shorter than human tag (under-cover / syllable cut) |
| `mixed_boundary` | 9 | 4.7 | both ends wrong in mixed ways |
| `late_start` | 8 | 4.2 | ML started late — missed the onset the human included |
| `too_long` | 7 | 3.7 | ML box longer than human tag (over-cover / merged neighbors) |
| `shifted` | 6 | 3.2 | whole box slid (start and end moved together) |
| `ok` | 4 | 2.1 | close enough (IoU high / small edge diffs) |
| `early_end` | 2 | 1.1 | ML ended early — cut off the tail the human kept |

### Active boundary distributions

| Metric | p25 | median | p75 | p90 |
|---|---:|---:|---:|---:|
| start delta (tag−cand) | -460.0 | **-230.0** | -10.0 | 0.0 |
| end delta (tag−cand) | -7.5 | **140.0** | 517.5 | 823.0 |
| dur ratio cand/tag | 0.372 | **0.512** | 0.689 | 0.925 |
| IoU | 0.337 | **0.454** | 0.552 | 0.693 |
| cand duration | 410.0 | **455.0** | 617.5 | 830.0 |
| tag duration | 810.0 | **970.0** | 1245.0 | 1930.0 |

Deduped active modes: `{'too_short': 75.0, 'late_start': 6.7, 'mixed_boundary': 5.0, 'too_long': 4.2, 'shifted': 4.2, 'ok': 3.3, 'early_end': 1.7}`  
Deduped median Δstart **-170.0** ms · Δend **90.0** ms · durR **0.603** · IoU **0.52**.

## Examples (active mode)

### `too_short` — ML box shorter than human tag (under-cover / syllable cut)

- **children** (Baby) raw ML 3:02.240–3:02.940 vs tag 3:02.010–3:02.980; active ML 3:02.240–3:02.940 vs tag 3:02.040–3:02.980; raw Δ=-230/+40 → active Δ=-200/+40, durR 0.722→0.745, IoU 0.722→0.745
- **spile** (Parent) raw ML 7:49.970–7:50.350 vs tag 7:49.850–7:50.370; active ML 7:49.970–7:50.350 vs tag 7:49.850–7:50.370; raw Δ=-120/+20 → active Δ=-120/+20, durR 0.731→0.731, IoU 0.731→0.731
- **pyjama** (Baby) raw ML 13:10.650–13:11.330 vs tag 13:10.370–13:11.330; active ML 13:10.650–13:11.330 vs tag 13:10.370–13:11.330; raw Δ=-280/+0 → active Δ=-280/+0, durR 0.708→0.708, IoU 0.708→0.708

### `mixed_boundary` — both ends wrong in mixed ways

- **children** (Baby) raw ML 3:06.920–3:07.760 vs tag 3:06.550–3:07.540; active ML 3:06.920–3:07.740 vs tag 3:06.550–3:07.540; raw Δ=-370/-220 → active Δ=-370/-200, durR 0.848→0.828, IoU 0.512→0.521
- **pyjama** (Parent) raw ML 11:27.020–11:27.910 vs tag 11:26.790–11:27.600; active ML 11:27.020–11:27.910 vs tag 11:26.790–11:27.600; raw Δ=-230/-310 → active Δ=-230/-310, durR 1.099→1.099, IoU 0.518→0.518
- **pyjama** (Parent) raw ML 11:27.020–11:27.910 vs tag 11:26.790–11:27.600; active ML 11:27.020–11:27.910 vs tag 11:26.790–11:27.600; raw Δ=-230/-310 → active Δ=-230/-310, durR 1.099→1.099, IoU 0.518→0.518

### `late_start` — ML started late — missed the onset the human included

- **what** (Baby) raw ML 3:03.095–3:03.510 vs tag 3:03.010–3:03.520; active ML 3:03.095–3:03.510 vs tag 3:03.010–3:03.520; raw Δ=-85/+10 → active Δ=-85/+10, durR 0.814→0.814, IoU 0.814→0.814
- **children** (Baby) raw ML 3:10.260–3:11.010 vs tag 3:10.110–3:10.970; active ML 3:10.260–3:11.010 vs tag 3:10.120–3:10.950; raw Δ=-150/-40 → active Δ=-140/-60, durR 0.872→0.904, IoU 0.789→0.775
- **vijay** (Baby) raw ML 7:02.700–7:03.260 vs tag 7:02.540–7:03.250; active ML 7:02.700–7:03.260 vs tag 7:02.540–7:03.250; raw Δ=-160/-10 → active Δ=-160/-10, durR 0.789→0.789, IoU 0.764→0.764

### `early_end` — ML ended early — cut off the tail the human kept

- **children** (Baby) raw ML 3:05.450–3:06.230 vs tag 3:05.450–3:06.380; active ML 3:05.450–3:06.230 vs tag 3:05.450–3:06.380; raw Δ=+0/+150 → active Δ=+0/+150, durR 0.839→0.839, IoU 0.839→0.839
- **cama** (Baby) raw ML 4:31.670–4:32.300 vs tag 4:31.670–4:32.450; active ML 4:31.670–4:32.300 vs tag 4:31.670–4:32.450; raw Δ=+0/+150 → active Δ=+0/+150, durR 0.808→0.808, IoU 0.808→0.808

### `too_long` — ML box longer than human tag (over-cover / merged neighbors)

- **do** (Baby) raw ML 3:16.950–3:17.360 vs tag 3:16.940–3:17.240; active ML 3:16.950–3:17.360 vs tag 3:16.940–3:17.240; raw Δ=-10/-120 → active Δ=-10/-120, durR 1.367→1.367, IoU 0.69→0.69
- **es** (Parent) raw ML 9:57.170–9:57.510 vs tag 9:57.170–9:57.380; active ML 9:57.170–9:57.510 vs tag 9:57.170–9:57.380; raw Δ=+0/-130 → active Δ=+0/-130, durR 1.619→1.619, IoU 0.618→0.618
- **aman** (Baby) raw ML 5:37.980–5:39.000 vs tag 5:38.360–5:38.920; active ML 5:37.980–5:38.990 vs tag 5:38.360–5:38.920; raw Δ=+380/-80 → active Δ=+380/-70, durR 1.821→1.804, IoU 0.549→0.554

### `shifted` — whole box slid (start and end moved together)

- **heidi** (Parent) raw ML 10:59.060–10:59.480 vs tag 10:58.930–10:59.320; active ML 10:59.060–10:59.480 vs tag 10:58.930–10:59.320; raw Δ=-130/-160 → active Δ=-130/-160, durR 1.077→1.077, IoU 0.473→0.473
- **pyjama** (Parent) raw ML 11:21.350–11:21.870 vs tag 11:21.150–11:21.690; active ML 11:21.350–11:21.870 vs tag 11:21.150–11:21.690; raw Δ=-200/-180 → active Δ=-200/-180, durR 0.963→0.963, IoU 0.472→0.472
- **green frog** (Baby) raw ML 6:46.650–6:47.410 vs tag 6:46.290–6:47.090; active ML 6:46.650–6:47.410 vs tag 6:46.330–6:47.090; raw Δ=-360/-320 → active Δ=-320/-320, durR 0.95→1.0, IoU 0.393→0.407

## Merge-back recommendation

**Production (keep):** `short_piece_ms=400`, `max_gap_ms=100`.  
**Explored from raw oracle gaps (not accepted):** `short_piece_ms=450`, `max_gap_ms=300`.

**Verdict:** `pending_loosen`  
**Pending acceptance (do not ship silently):** `short_piece_ms=400`, `max_gap_ms=200`.

Active-audio still shows dominant too_short with large *speech* edge errors (median Δstart -230.0 ms, Δend 140.0 ms, durR 0.512). Recommend pending acceptance of short_piece_ms=400, max_gap_ms=200 — do not ship until explicitly accepted. 450/300 from raw oracle gaps remains not accepted.

## Plain takeaways

- Honesty note: the earlier dismiss-vs-tag report (`dismiss_vs_tag_delta_26_07_27.md`) used **raw timestamps only** — silence inside a human tag counted the same as speech.
- Active-audio trim: frames ≥ noise floor (-57.52 dB) + 6.0 dB, or near the in-box peak (≤25.0 dB down, still above floor+3), then ±40.0 ms pad, clipped to the raw box.
- Raw modes too_short 81.1% → active 81.1% (deduped 75.0% → 75.0%). Median Δstart -230.0 → -230.0 ms; Δend 150.0 → 140.0 ms; active dur ratio 0.512.
- Median silence inside tags: lead 0.0 ms, trail 0.0 ms (cand lead/trail 0.0/0.0 ms). Silence-explained edge shrink (median): start 0.0 ms, end 0.0 ms.
- Active-audio still shows dominant too_short with large *speech* edge errors (median Δstart -230.0 ms, Δend 140.0 ms, durR 0.512). Recommend pending acceptance of short_piece_ms=400, max_gap_ms=200 — do not ship until explicitly accepted. 450/300 from raw oracle gaps remains not accepted.
- Production stays short_piece_ms=400, max_gap_ms=100 until a pending suggestion is explicitly accepted. 450/300 from raw oracle gaps is not accepted.

_Generated by `tools/analysis/dismiss_vs_tag_delta.py --active-audio`. Does not modify library files or production defaults._
