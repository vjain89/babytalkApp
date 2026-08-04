# Dismissed ML vs manual tags — `26_07_27__19:53:00`

**Kit:** `2026-07-28T04-26-04-490Z_1785211756660_353_Mississippi_St_87`  
**Question:** When the reviewer dismisses a Find-speech candidate and draws (or keeps) a tag nearby, how do the boxes differ?

## Counts

| | n |
|---|---:|
| Dismissed candidates | 311 |
| Matched to any overlapping tag | 193 |
| … to hand-drawn (`source≠ml_confirmed`) | 190 |
| … to `ml_confirmed` only | 3 |
| Dismissed with **no** tag overlap (junk) | 118 |
| Manual tags with **no** dismissed overlap | 18 |
| `ml_confirmed` tags with no dismissed overlap | 133 |
| All tags / manual / ml_confirmed | 284 / 148 / 136 |

### Sign convention

- **start delta** = tagStart − candStart (ms). Positive → human started **later** → ML had **lead-in** (started early).
- **end delta** = tagEnd − candEnd. Positive → human ended **later** → ML **cut off early**.
- **duration ratio** = candDur / tagDur. Below 1 → ML shorter than human.

## Hand-drawn rescues (dismiss + manual tag)

n = **190** pairs (dismissed cand ↔ best overlapping non-`ml_confirmed` tag).

### Error modes

| Mode | n | % | Meaning |
|---|---:|---:|---|
| `too_short` | 154 | 81.1 | ML box shorter than human tag (under-cover / syllable cut) |
| `mixed_boundary` | 12 | 6.3 | both ends wrong in mixed ways |
| `too_long` | 8 | 4.2 | ML box longer than human tag (over-cover / merged neighbors) |
| `shifted` | 6 | 3.2 | whole box slid (start and end moved together) |
| `late_start` | 5 | 2.6 | ML started late — missed the onset the human included |
| `ok` | 3 | 1.6 | close enough (IoU high / small edge diffs) |
| `early_end` | 2 | 1.1 | ML ended early — cut off the tail the human kept |

### Boundary distributions (ms / ratio)

| Metric | p25 | median | p75 | p90 |
|---|---:|---:|---:|---:|
| start delta (tag−cand) | -480.0 | **-230.0** | -20.0 | 0.0 |
| end delta (tag−cand) | -20.0 | **150.0** | 550.0 | 851.0 |
| dur ratio cand/tag | 0.369 | **0.506** | 0.682 | 0.929 |
| IoU | 0.327 | **0.44** | 0.549 | 0.684 |
| cand duration | 420.0 | **460.0** | 620.0 | 830.0 |
| tag duration | 810.0 | **990.0** | 1265.0 | 2060.0 |

IoU ≥ 0.5: **35.3%** · IoU ≥ 0.7: **8.9%**

### Per-tag view (best-IoU pair only)

Unique manual tags matched: **120** (55 tags hit by ≥2 dismissed boxes; mean 1.58 dismissals/tag).

Modes (deduped): {'too_short': 75.0, 'mixed_boundary': 7.5, 'too_long': 5.0, 'late_start': 4.2, 'shifted': 4.2, 'ok': 2.5, 'early_end': 1.7}

Median Δstart **-165.0** ms · Δend **115.0** ms · dur ratio **0.606** · IoU **0.51** (IoU≥0.5: 55.0%).

## Overlap with `ml_confirmed` tags only

n = 3 (dismissed cand overlapping a confirmed-ML tag — usually a neighbor or a prior confirm, not the Add-tag workflow).

Modes: {'ok': 100.0}
Median dStart 0.0 ms, dEnd 0.0 ms, durRatio 1.0.

## Examples (by mode)

### `too_short` — ML box shorter than human tag (under-cover / syllable cut)

- **vijay** (Baby, `user`) @ 5:34.715–5:35.060 (ML) vs 5:34.590–5:35.060 (tag); Δstart=-125 ms, Δend=+0 ms, durR=0.734, IoU=0.734
- **spile** (Parent, `user`) @ 7:49.970–7:50.350 (ML) vs 7:49.850–7:50.370 (tag); Δstart=-120 ms, Δend=+20 ms, durR=0.731, IoU=0.731
- **children** (Baby, `user`) @ 3:02.240–3:02.940 (ML) vs 3:02.010–3:02.980 (tag); Δstart=-230 ms, Δend=+40 ms, durR=0.722, IoU=0.722

### `mixed_boundary` — both ends wrong in mixed ways

- **pyjama** (Parent, `user`) @ 11:27.020–11:27.910 (ML) vs 11:26.790–11:27.600 (tag); Δstart=-230 ms, Δend=-310 ms, durR=1.099, IoU=0.518
- **pyjama** (Parent, `user`) @ 11:27.020–11:27.910 (ML) vs 11:26.790–11:27.600 (tag); Δstart=-230 ms, Δend=-310 ms, durR=1.099, IoU=0.518
- **children** (Baby, `user`) @ 3:06.920–3:07.760 (ML) vs 3:06.550–3:07.540 (tag); Δstart=-370 ms, Δend=-220 ms, durR=0.848, IoU=0.512

### `late_start` — ML started late — missed the onset the human included

- **what** (Baby, `user`) @ 3:03.095–3:03.510 (ML) vs 3:03.010–3:03.520 (tag); Δstart=-85 ms, Δend=+10 ms, durR=0.814, IoU=0.814
- **children** (Baby, `user`) @ 3:10.260–3:11.010 (ML) vs 3:10.110–3:10.970 (tag); Δstart=-150 ms, Δend=-40 ms, durR=0.872, IoU=0.789
- **vijay** (Baby, `user`) @ 7:02.700–7:03.260 (ML) vs 7:02.540–7:03.250 (tag); Δstart=-160 ms, Δend=-10 ms, durR=0.789, IoU=0.764

### `early_end` — ML ended early — cut off the tail the human kept

- **children** (Baby, `user`) @ 3:05.450–3:06.230 (ML) vs 3:05.450–3:06.380 (tag); Δstart=+0 ms, Δend=+150 ms, durR=0.839, IoU=0.839
- **cama** (Baby, `user`) @ 4:31.670–4:32.300 (ML) vs 4:31.670–4:32.450 (tag); Δstart=+0 ms, Δend=+150 ms, durR=0.808, IoU=0.808

### `too_long` — ML box longer than human tag (over-cover / merged neighbors)

- **do** (Baby, `user`) @ 3:16.950–3:17.360 (ML) vs 3:16.940–3:17.240 (tag); Δstart=-10 ms, Δend=-120 ms, durR=1.367, IoU=0.69
- **es** (Parent, `user`) @ 9:57.170–9:57.510 (ML) vs 9:57.170–9:57.380 (tag); Δstart=+0 ms, Δend=-130 ms, durR=1.619, IoU=0.618
- **aman** (Baby, `user`) @ 5:37.980–5:39.000 (ML) vs 5:38.360–5:38.920 (tag); Δstart=+380 ms, Δend=-80 ms, durR=1.821, IoU=0.549

### `shifted` — whole box slid (start and end moved together)

- **blybe** (Parent, `user`) @ 9:56.690–9:57.110 (ML) vs 9:56.540–9:56.960 (tag); Δstart=-150 ms, Δend=-150 ms, durR=1.0, IoU=0.474
- **heidi** (Parent, `user`) @ 10:59.060–10:59.480 (ML) vs 10:58.930–10:59.320 (tag); Δstart=-130 ms, Δend=-160 ms, durR=1.077, IoU=0.473
- **pyjama** (Parent, `user`) @ 11:21.350–11:21.870 (ML) vs 11:21.150–11:21.690 (tag); Δstart=-200 ms, Δend=-180 ms, durR=0.963, IoU=0.472

## Junk dismissals (no overlapping tag)

118 dismissed spans with zero tag overlap — reviewer rejected noise / non-target speech. Sample:

- 50.170s–50.640s (470 ms), speechScore=0.768, cluster=SPEAKER_00
- 50.740s–51.160s (420 ms), speechScore=0.829, cluster=SPEAKER_00
- 50.740s–51.500s (760 ms), speechScore=0.613, cluster=SPEAKER_00
- 2:15.270–2:16.100 (830 ms), speechScore=0.831, cluster=SPEAKER_01
- 2:22.840–2:23.260 (420 ms), speechScore=0.706, cluster=SPEAKER_00
- 2:23.490–2:23.940 (450 ms), speechScore=0.908, cluster=SPEAKER_00
- 2:24.240–2:25.370 (1130 ms), speechScore=0.797, cluster=SPEAKER_00
- 2:38.080–2:38.500 (420 ms), speechScore=0.899, cluster=SPEAKER_00

## Manual tags with no dismissed candidate nearby

18 hand-drawn tags never sat on a dismissed box (finder miss, or candidate was confirmed elsewhere). Sample:

- **do** (Baby) @ 2:49.780–2:50.010
- **you** (Baby) @ 2:50.000–2:50.290
- **—** (Baby) @ 2:56.390–2:56.960
- **I** (Baby) @ 3:54.420–3:54.760
- **schaffe** (Baby) @ 4:10.450–4:11.070
- **melanie** (Baby) @ 4:09.770–4:10.460
- **das** (Baby) @ 5:05.890–5:06.210
- **das** (Baby) @ 5:47.390–5:47.710

## Plain takeaways (merge-back / padding / finder)

- Of 311 dismissed VAD proposals, 193 overlap a tag (190 to a hand-drawn tag, 3 to ml_confirmed); 118 are pure junk rejects with no tag nearby.
- On hand-drawn rescues (n=190 pairs → 120 unique tags), main modes: too_short 81.1%, mixed_boundary 6.3%, too_long 4.2%. Median start delta -230.0 ms, end delta 150.0 ms, duration ratio (cand/tag) 0.506.
- Per-tag (best-IoU) view still says under-cover: too_short 75.0%, median Δstart -165.0 ms (ML late / missed onset), Δend 115.0 ms (ML early cut-off), dur ratio 0.606.
- 55 rescued words overlap ≥2 dismissed boxes (mean 1.58 dismissals/tag) — classic syllable/fragment split that merge-back targets; flat edge-pad cannot glue siblings.
- Dominant failure is too_short (cand ~½–⅔ of the human word) on **raw** timestamps. Prefer merge-back / longer word-like pieces over more padding — but retune gap/short only after active-audio confirms the miss is speech, not silence.
- Junk dismissals are 37.9% of dismissals (118/311) — true false positives, not boundary edits; speechScore / role gating matters as much as boundary polish.
- 18 manual tags have no overlapping dismissed candidate (finder miss / never proposed) — coverage gap separate from dismiss-and-redraw boundary fixes.

_Generated by `tools/analysis/dismiss_vs_tag_delta.py`. Does not modify library files._
