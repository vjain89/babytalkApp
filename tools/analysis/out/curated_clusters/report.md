# Curated clusters: can our fingerprints tell same-word groups apart?

> Machine-readable details (every number, every cluster) are in `summary.json` next to this file. Plots: `hist_mel_vs_hubert.png`, `umap_mel.png`, `umap_hubert.png`.

---

## Why this report exists

In BabyTalk’s **Review Browser**, you listen to clips, tag words, and (in the **Clustering** tab) group similar-sounding snippets into clusters. We already use **mel fingerprints** — compact “sound shapes” of each clip — to help auto-group things.

A natural question:

> When *humans* have already grouped clips into a labeled word cluster (e.g. several “boot” takes), do those clips look close to each other in fingerprint space — and far from other words?

If yes, our fingerprints are doing something useful for clustering. If no, the Clustering UI won’t magically get better without better fingerprints or cleaner data.

This study answers that for **your manually curated clusters**, and compares **mel** (what production uses) vs **HuBERT** (a heavier speech model we tried earlier).

---

## What we looked at

**Not** the auto “syllable mush” piles the Clustering tab sometimes makes on its own.  
**Not** simply “any two tags that happen to share the same typed word string.”

We used **curated / labeled clusters** from `clusters.json`: groups a human named or marked as curated, with at least 2 members that map back to real audio tags.

| | Count |
|--|------:|
| Kits scanned | 5 |
| Kits with usable curated clusters | 5 |
| Curated clusters (≥2 members) | 34 |
| Clips (members) before filters | 177 |
| Clips kept after filters | **169** |

**What we filtered out (same spirit as the Clustering tab):**

1. **SHORT fragments** — very brief clips (kit-adaptive cutoff ~400–500 ms, same idea as the SHORT badge). Dropped **8** clips (~5%).
2. **Non-verbal vocalization** clusters — none dropped here (0).
3. Clips with no loadable audio — none missing (0).

So the dataset is: **5 kits · 34 human word groups · 169 clips**.

---

## How to read the numbers (plain English)

### Fingerprint / embedding

A **fingerprint** (also called an **embedding**) is a list of numbers that summarizes how a clip sounds. Similar-sounding clips should get similar fingerprints.

We compared three flavors:

| Name | What it is |
|------|------------|
| **mel** | The production fingerprint BabyTalk clustering already uses |
| **mel_norm** | A slightly cleaned analysis version of mel (volume/grid tweaks) |
| **HuBERT** | A large pretrained speech model (slower / heavier) |

### Distance

**Distance** = how different two fingerprints are (here: cosine distance).  
**Lower** = more alike. **Higher** = more different.

### Within vs between

- **Within** — pairs of clips that sit in the *same* curated cluster (e.g. two “boot”s). We want these distances **small**.
- **Between** — pairs from *different* clusters (e.g. “boot” vs “fisch”). We want these **larger**.

### Gap (the headline score)

**Gap = average between − average within.**

Bigger gap → same-word groups clump together and different words sit farther apart → **better for clustering**.  
Tiny gap → fingerprints don’t separate the human groups well.

### Tight vs loose clusters

For one curated word group, **mean within** = average distance among its members.

- **Tight** (low within) — repeats of that word sound alike in fingerprint space. Example: *zeichne* at **0.32**.
- **Loose** (high within) — clips labeled the same word still sound quite different to the fingerprint. Example: *brown* at **0.97**. Those are good candidates to re-listen in Review Browser.

---

## What we found

### Mel beats HuBERT for this job

| Fingerprint | Avg within (same cluster) | Avg between (different clusters) | **Gap** |
|-------------|--------------------------:|---------------------------------:|--------:|
| mel | 0.701 | 0.852 | **0.151** |
| mel_norm | 0.706 | 0.858 | **0.152** |
| HuBERT | 0.167 | 0.265 | **0.097** |

(492 same-cluster pairs, 13,704 different-cluster pairs.)

**In plain language:** Mel and mel_norm separate human word groups about the same. HuBERT’s gap is clearly smaller (~0.10 vs ~0.15), so it is **not** a better default for BabyTalk clustering here — especially given how much heavier it is.

### Matches the earlier word-string experiment

Earlier we measured the same gap using “same typed word on hand tags” (not curated cluster IDs). Those gaps were mel_norm ≈ **0.154** and HuBERT ≈ **0.095**.

| | mel_norm gap | HuBERT gap |
|--|-------------:|-----------:|
| Earlier (same word string) | 0.154 | 0.095 |
| This study (curated cluster id) | 0.152 | 0.097 |
| Difference | −0.002 | +0.002 |

So: grouping by **curated cluster** vs grouping by **typed word** gives almost the same separation story. Curated IDs are still the cleaner “ground truth” when you have them, because a human explicitly put those clips together.

### Tightest curated words (mel) — fingerprints agree

These groups sound consistent to mel:

| Word | Members | Mean within | Median length (ms) |
|------|--------:|------------:|-------------------:|
| zeichne | 2 | 0.32 | 1145 |
| fahre | 2 | 0.44 | 1130 |
| boot | 4 | 0.52 | 880 |
| da | 3 | 0.52 | 740 |
| vijay | 6 | 0.53 | 650 |
| children | 7 | 0.53 | 970 |
| see | 2 | 0.55 | 805 |
| fisch | 2 | 0.55 | 580 |

### Loosest curated words (mel) — worth a second listen

These share a label but fingerprint distances are large (messy / variable):

| Word | Members | Mean within | Median length (ms) |
|------|--------:|------------:|-------------------:|
| brown | 9 | 0.97 | 650 |
| bus | 3 | 0.93 | 590 |
| melanie | 2 | 0.92 | 1030 |
| fuchs | 5 | 0.91 | 1010 |
| bett | 3 | 0.85 | 670 |
| armadillo | 5 | 0.84 | 1310 |
| ballon | 2 | 0.79 | 705 |
| schuhe | 4 | 0.78 | 1240 |

Other loose-ish groups (also review candidates): cappuccino (0.76), bear (0.76), wasser (0.76), another melanie (0.77).

**Interpretation:** Tight words are good examples of “fingerprint clustering can work.” Loose words may mix different pronunciations, background noise, or borderline tags — the fingerprint isn’t wrong so much as the group isn’t acoustically uniform.

---

## What this does NOT mean

- **Not** proof that the Clustering tab already auto-groups perfectly — we measured *human* curated groups against fingerprints, not auto-cluster quality.
- **Not** “HuBERT is useless forever” — only that on *this* curated set, its separation gap is worse than mel and not worth the cost as a default.
- **Not** that every loose word is a bad tag — it may be a hard word (variable speech) or a small sample (many groups have only 2–3 clips).
- **Not** a huge dataset — 5 kits, 34 clusters, 169 clips. Directionally useful; not a final product benchmark.
- Gap numbers for mel (~0.7 within) vs HuBERT (~0.2 within) sit on different scales; compare **gaps** (and relative stories), not raw within numbers across models.

---

## What to dig into next (prioritized)

1. **Keep mel fingerprints as the default** for clustering / Review Browser — don’t switch to HuBERT based on this.
2. **Keep excluding SHORT clips** when building or seeding clusters — curated truth still looks like word-length audio, not tiny fragments.
3. **Treat curated clusters as ground truth** for future experiments (cleaner than “same typed word string” alone).
4. **Don’t expect Clustering UI magic yet** — mel *does* separate human groups somewhat, but the gap is modest; auto-clustering still needs careful thresholds and human review.
5. **Spot-check the loose words** (*brown*, *bus*, *melanie*, *fuchs*, …) in Review Browser — fix tags or split groups if they aren’t really the same sound.
6. **Optional:** grow the curated set (more kits / more multi-member word clusters) before deciding on bigger model or algorithm changes.

---

*Generated from `tools/analysis/curated_cluster_learn.py`. Same numbers as `summary.json`.*
