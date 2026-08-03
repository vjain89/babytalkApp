# Fingerprint bake-off: which sound fingerprint best matches your curated words?

> Machine numbers: `summary.json` next to this file. Script: `tools/analysis/curated_fingerprint_bakeoff.py`.

---

## Why this report exists

BabyTalk’s Clustering tab groups clips using a **fingerprint** — a short list of numbers that summarize how a clip sounds. Production uses a **mel** fingerprint today. Other models (YAMNet, VGGish, PANNs, BYOL-A, HuBERT, …) make different fingerprints.

We already know mel separates your **human curated word groups** with a pooled gap of about **0.15**. This bake-off asks: *do any frozen off-the-shelf fingerprints do better — including when we leave one recording kit out?*

---

## How to read the numbers (read this before the table)

### Fingerprint / embedding

A **fingerprint** (also called an **embedding**) is a vector of numbers for one clip. Similar-sounding clips should get similar fingerprints. Here every model is **frozen** — we do **not** train or fine-tune on your labels. We only measure how well the ready-made fingerprint already separates human word groups.

### Distance, within, between

**Distance** = how different two fingerprints are (cosine distance). Lower = more alike.

- **Within** — pairs in the *same* curated cluster (two “boot”s). We want this **small**.
- **Between** — pairs from *different* clusters (“boot” vs “fisch”). We want this **larger**.

### Gap (the headline score)

**Gap = average between − average within.**

Bigger gap → same-word clips clump together and different words sit farther apart → **better for clustering**. Tiny gap → the fingerprint doesn’t separate your human groups well.

Compare **gaps across models**, not raw within numbers — different models live on different scales.

### Pooled vs leave-one-kit-out (LOKO)

- **Pooled** — put every clip from every kit in one big pile, then measure gap. This is the familiar ~0.15 mel number from the curated-clusters report. It can look a bit optimistic if kits differ a lot.
- **Leave-one-kit-out (LOKO)** — for each kit in turn, measure the gap **using only that kit’s clips**. Then average those kit gaps. Nothing is “trained” on the other kits (encoders are frozen); LOKO just answers: *does this fingerprint still separate words on a kit you didn’t mix into the pooled score?*

When people say “compare to mel’s 0.15,” check **both** columns: pooled ≈ 0.15 for mel, and mel’s **LOKO mean** for a fair head-to-head with other fingerprints.

---

## What we looked at

Same curated filters as the earlier curated-clusters study:

- Kits kept: **5** (2026-07-20T20-56-36-042Z_353_Mississippi_St_75, 2026-07-21T15-35-25-419Z_1784648090362_353_Mississippi_St_73, 2026-07-24T05-49-13-915Z_1784871954392_353_Mississippi_St_78, 2026-07-24T05-49-15-530Z_1784871954381_UCSF_Benioff_Children_s_Ho, 2026-07-28T04-26-04-490Z_1785211756660_353_Mississippi_St_87)
- Curated clusters (≥2 members after filters): **34**
- Clips kept: **169** (dropped SHORT=8, nonverbal=0)

---

## Results: pooled + LOKO gaps

| Fingerprint | Pooled within | Pooled between | **Pooled gap** | **LOKO mean gap** | LOKO std | Scorable kits |
|-------------|--------------:|---------------:|---------------:|------------------:|---------:|--------------:|
| mel | 0.701 | 0.852 | **0.151** | **0.124** | 0.060 | 4/5 |
| hubert | 0.167 | 0.264 | **0.097** | **0.077** | 0.007 | 4/5 |
| yamnet | 0.212 | 0.306 | **0.094** | **0.050** | 0.053 | 4/5 |
| vggish | 0.056 | 0.085 | **0.029** | **0.012** | 0.010 | 4/5 |
| panns | 0.077 | 0.114 | **0.037** | **0.025** | 0.020 | 4/5 |
| byola | 0.114 | 0.183 | **0.069** | **0.036** | 0.030 | 4/5 |

**Mel baseline:** pooled gap **0.151**, LOKO mean gap **0.124** (± 0.060 across 4 kits).

### Plain-language takeaway

On this curated set, **mel still leads** (or ties) on leave-one-kit-out. Do **not** switch production clustering to another frozen fingerprint based on this bake-off alone.

### Mel LOKO by kit (so “0.15” isn’t confused)

| Holdout kit | Members | Within | Between | Gap |
|-------------|--------:|-------:|--------:|----:|
| 2026-07-20T20-56-36-042Z_353_Mississi… | 49 | 0.681 | 0.886 | 0.205 |
| 2026-07-21T15-35-25-419Z_178464809036… | 46 | 0.742 | 0.855 | 0.114 |
| 2026-07-24T05-49-13-915Z_178487195439… | 6 | 0.743 | 0.781 | 0.038 |
| 2026-07-24T05-49-15-530Z_178487195438… | 3 | — | — | unscorable |
| 2026-07-28T04-26-04-490Z_178521175666… | 65 | 0.683 | 0.821 | 0.138 |

### Skipped fingerprints

- **Audio-JEPA / EAT** — No clearly installable pip package / weights path in tools/.venv without cloning research codebases; skipped (not heroic effort).

---

## MeWEHV (paper check)

**MeWEHV** (Mel and Wave Embeddings for Human Voice Tasks) is a research recipe that glues together two views of the same clip: (1) a big pretrained **wave** encoder (things like HuBERT / WavLM / Wav2Vec-style models) and (2) a small network that reads **MFCCs** (a classic speech feature related to mel). The combo is aimed at **who is speaking**, **which language**, and **which accent** — not at grouping toddler word takes like “boot” vs “fisch.”

**Recommendation:** **Don’t use MeWEHV now for BabyTalk word clustering.** It solves a different problem (speaker/language/accent), adds training complexity, and we already have a simpler frozen-fingerprint bake-off path. Revisit only if you pivot to speaker/language ID.

---

## FAQ — VAD parents/children, SSL, and what to do in Review

### 5. VAD status — parents vs children (why Clustering looked like syllable mush)

BabyTalk’s current speech-finding pipeline is roughly:

1. **Energy VAD + speechlike** — find louder-than-the-room regions that also sound like a voice (not taps/doors/water).
2. **ECAPA (optional)** — cut those regions by speaker so one speaker’s talk isn’t mixed with another’s.
3. **Pause-split** — if a same-speaker span is still very long, cut on silences.
4. **DJW resegment** — de Jong & Wempe syllable-nucleus cuts turn a longer span into **syllable-ish children** that become ML candidates in Review.

**Parents** = the longer pre-resegment spans (VAD / speaker / pause pieces). **Children** = the shorter DJW pieces Review usually shows as “speech segments.”

So when you open Review’s speech segments, you are mostly looking at **children** — often **syllables or short scraps**, not clean dictionary words. Clustering then fingerprints and groups those scraps. That is why auto-clusters can look like **syllable mush**: the input units are syllable-sized by design. Your **curated** clusters are different — you grouped word-like takes by hand — which is why this bake-off uses curated groups as ground truth, not the raw auto mush.

### 7. Two SSL ideas in beginner terms

**SSL** = self-supervised learning: a model learns useful sound patterns from lots of audio **without** word labels, then you reuse it.

1. **Frozen HuBERT (or YAMNet/…) fingerprint bake-off** — what this report did: download a pretrained model, freeze it, turn each clip into a fingerprint, measure gap. Cheap to try; no training on your library.
2. **Training / adapting on all library audio** — run SSL (or fine-tuning) on *your* recordings so the fingerprint learns BabyTalk’s rooms, mics, kids, and languages. More work; only worth it once you have enough clean curated word groups to prove it helped.

**“SSL across datasets”** would mean mixing BabyTalk kits with other public child/adult speech corpora during that adaptation step so the model generalizes better. **Wait** until you have more curated multi-member clusters: without a solid gap/LOKO test set, you can’t tell if the extra training helped word clustering or just shuffled numbers.

### Review workflow — segments, clusters, or both?

- **Review ML speech segments (children)** when you want more *candidates* — accept / edit / dismiss syllable-ish proposals so tags exist. Expect fragments; merge or retag into real words as you listen.
- **Curate clusters** when you already have a few good tags of the same word — group them, name the cluster. That curated set is the ground truth we use for fingerprint experiments.
- **Both, in that order:** segments → clean tags → curated clusters. Don’t spend hours on Clustering auto-mush until you’ve curated a few solid word groups per kit.

### When to look at new fingerprints?

After this bake-off: only chase a new fingerprint if its **LOKO gap clearly beats mel** on more kits, or if a product need appears (e.g. emotion / vocalization type — different job). Otherwise grow curated clusters and keep mel.

### BYOL-A / YAMNet now vs after bake-off?

**After** — this report *is* that bake-off. Read the table above; don’t install them into production Clustering unless LOKO says they’re better. Trying them “just because” before numbers wastes Review time.

---

## What you should do this week

1. **Keep mel** as the Clustering default unless a row above clearly wins LOKO by a meaningful margin and still looks good when you re-listen.
2. **In Review:** confirm/dismiss speech-segment children to grow tags; then **curate** multi-member word clusters (the fuel for the next bake-off). Spot-check loose curated words from the earlier report (*brown*, *bus*, *fuchs*, …).
3. **Skip MeWEHV** and skip SSL-across-datasets until you have more curated clusters; re-run this bake-off script when the curated set grows.

---

*Generated from `curated_fingerprint_bakeoff.py`. Spaces tried: mel, hubert, yamnet, vggish, panns, byola.*
