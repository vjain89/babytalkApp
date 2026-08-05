# BabyTalk: Developmental Timeline — Revised Approach

**Status:** Design handoff for implementation
**Supersedes:** Word-level segmentation as a blocking dependency for parent-facing output
**Context:** Mac-side annotation/ML pipeline (see `ARCHITECTURE_DEEP_DIVE.md` for the iOS capture side, which is unaffected by this doc)

---

## 1. Why this reframe

The original ML pipeline goal was: segment continuous audio into individual word-level units, cluster them, and build a word-frequency-over-time timeline. That requires precise word boundaries, which turned out to be a much harder problem than it looks:

- Waveform shape mostly reflects prosodic envelope (syllable count/stress), not lexical identity — visual similarity between same-shaped words is misleading, not a sign the segmentation problem is easy.
- VAD finds speech/silence, not word boundaries — coarticulation means word boundaries often have no acoustic discontinuity to detect.
- Forced alignment (the standard solution for word-level timing) requires a *known transcript* — it solves "when" given "what." We don't have a reliable "what" for toddler speech (Whisper failed here — domain mismatch, confirmed).
- Open-vocabulary segmentation (discovering word boundaries with no transcript) is an open research problem even for adult speech. It is not a solved technique we're failing to apply correctly; it's not solved, period, at this data volume.
- HomeBank/ACLEW/DARCLE (the standard toddler-speech corpus/annotation approach) doesn't attempt word-level segmentation either — it segments at the vocalization/speaker-turn level and classifies coarse vocalization type. Word-level transcription in that literature is hand-done by trained annotators on small sub-corpora, not automated.

**The actual goal, restated:** show a parent how their child's vocalizations evolve over time — cry → babble → jargon → first words → word combinations — and surface newly-recurring sounds as candidate new words for the parent to label. This does **not** require precise word boundaries. It requires per-vocalization classification, which is a smaller, better-posed, more error-tolerant problem than open segmentation.

Word-level clustering and segmentation work continues, but as a pipeline that *enhances* the timeline and word-discovery features over time — not as a blocker to shipping them.

---

## 2. Developmental stage taxonomy

Label at the **vocalization** level (see §3), not the word level. A vocalization
carries one primary label. Stages **overlap and coexist** in development — score
*this* vocalization, not “how far the child has progressed overall.”

| Label | Description |
|---|---|
| `vegetative` | Biological, non-communicative (cough, sneeze, breathing, burp, snore, hiccup) |
| `cry` | Distress/discomfort — sustained or slowly modulated, not rhythmic bursts |
| `laugh` | Giggle — rhythmic burst of short repeated voiced pulses, positive affect |
| `canonical_babble` | Repeated well-formed CV syllables; no adult-like sentence intonation. Also interim home for vowel-only vocal play (“uh”/“um” with no clear communicative intent) — add a context note (“vowel-only, no CV structure”) when used that way |
| `jargon` | Adult-like sentence-level stress/intonation; no identifiable real word |
| `protoword` | Consistent non-adult-form sound used referentially; stable across occurrences. Not restricted to CV — a stable vowel-only sound (“uh”) with reaching/pointing/directed attention qualifies here (intent present) |
| `single_word` | Identifiable single-word attempt; approximate pronunciation OK |
| `word_combination` | Two or more distinct known words with a short internal gap |
| `noise` | Not a biological vocalization worth tracking (background noise, mic bump, adult speech mis-attributed as child, etc.) |

**Tiebreakers**

- **vegetative vs noise:** came from the child’s body → `vegetative`; didn’t → `noise`
- **cry vs laugh vs jargon:** sustained/strained → `cry`; rhythmic + positive affect → `laugh`; sentence-like melody, no burst repetition → `jargon`
- **protoword vs single_word:** only pronunciation form matters (not frequency/confidence). Matches adult form → `single_word` (even once); never matches adult form despite repetition → `protoword`
- **vowel-only:** communicative intent (pointing/reaching/directed) → `protoword`; no clear intent (vocal play) → `canonical_babble` + context note (“vowel-only, no CV structure”)

Underlying cluster-match evidence (when available) should still be stored — it feeds word discovery and combination detection.

---

## 3. Revised pipeline

```
raw audio (session)
    │
    ▼
Speaker diarization  ─────────────────  existing (ECAPA-TDNN); VTC evaluation still pending, see §7
    │  (who + when: adult/child turns)
    ▼
Vocalization segmentation  ────────────  coarse boundaries, ACLEW/VTC-style granularity
    │  (one child vocalization = one turn, pause-delimited — this is the unit we label)
    ▼
    ├─────────────────────────────┬─────────────────────────────┐
    ▼                             ▼                             ▼
Stage classification      Cluster matching (existing        Combination detection
(§4)                      cluster_sounds.py, extended)      (§6, built on cluster matching)
    │                             │                             │
    ▼                             ▼                             ▼
Timeline / album           New word discovery (§5)        word_combination label
(parent-facing)            (parent-facing prompt)          feeds back into timeline
```

Word-level segmentation (DJW + merge-back) is **not** in this critical path. It remains a parallel, lower-priority effort that improves cluster precision (and therefore stage-classification accuracy and combination detection accuracy) as it matures — see §7.

---

## 4. Feature: Developmental timeline / album

**Goal:** per-vocalization stage label, aggregated into a time-series view a parent can browse.

**Inputs per vocalization** (no word-level boundaries required):
- Duration
- Approximate syllable count — DJW output is usable here even with imperfect merge-back, since syllable *count* is far more robust to over/under-splitting than exact split *placement*
- Pitch contour shape/variability (distinguishes jargon's adult-like intonation from flat babble)
- Whether the vocalization matches one or more existing labeled clusters, and how many distinct clusters it matches (feeds `protoword` / `single_word` / `word_combination` distinction)

**Output:** one stage label per vocalization, timestamped, stored alongside existing diarization/tag data.

**Parent-facing view:** timeline showing stage-label distribution over time (e.g., weekly rollup of vocalization counts per stage), plus an "album" of representative audio snippets per stage per time period. This is the primary deliverable — it should be buildable and demoable without waiting on segmentation quality improvements.

**Build note:** start with a small hand-labeled set of vocalizations across these 6 stages to validate that duration + syllable count + pitch contour features actually separate the classes before committing to a specific classifier. Given six overlapping/non-exclusive classes and probably a modest label set from one child, a simple classifier (e.g., gradient-boosted trees or even hand-tuned rules on the feature set above) is more appropriate than deep learning here — this is a small-data problem, same lesson as the embedding work.

---

## 5. Feature: New word discovery

Builds directly on the existing clustering pipeline (`cluster_sounds.py`) — no new architecture, just a trigger condition and a UI surface.

**Trigger:** a cluster becomes a "candidate new word" when:
- It accumulates enough occurrences within a rolling window (e.g., 3+ occurrences within N days) — threshold TBD empirically once real data volume is available
- Intra-cluster variance is low (occurrences are acoustically consistent with each other)
- It does not match any already-labeled cluster

**Parent-facing prompt:** "Heard this sound N times this week — want to label it?" with playback of 2–3 exemplar occurrences and a free-text or tag field for meaning. This is the same interaction whether the underlying sound is a `protoword` (child's own non-adult form) or a `single_word` (recognizable word) — the parent's label determines which.

**Relationship to review UI:** this is a lighter-weight version of the clustering review UI already being designed — same "show exemplars, ask for a label" pattern, just triggered by occurrence count rather than manual review queue order.

---

## 6. Feature: Word combination detection

**Goal:** detect vocalizations like "wasser trinke" without needing precise internal word boundaries.

**Method:** within one vocalization, check for matches to **two or more distinct known clusters** with a short internal gap between them (reuse the same short-gap-vs-pause heuristic developed for merge-back gating — see §7). This is closed-set template matching against existing clusters, not open segmentation:

- If the vocalization matches ≥2 distinct clusters with an internal gap short enough to indicate within-utterance continuity (not a pause between separate vocalizations) → label `word_combination`
- Store which two (or more) clusters matched, for the timeline/album view to show "child said [word A] + [word B] together"

**Error tolerance:** getting the internal split slightly wrong here is low-cost — worst case a combination gets logged as a single complex word rather than two, which doesn't corrupt any cluster (unlike a bad automatic merge in the old pipeline, which could pollute a cluster's centroid).

---

## 7. What continues in parallel (not blocking)

These remain open workstreams that improve pipeline precision over time, but nothing in §4–§6 depends on them being finished first:

1. **Merge-back duration-ratio + gap-duration gating** — improves DJW output quality, which improves syllable-count accuracy (§4) and combination-detection gap heuristics (§6). Approach:
   - Build empirical duration distribution from already-confirmed correct single-word segments (median + IQR)
   - Gate merges by *both* duration ratio (candidate segment duration relative to that distribution) *and* gap duration (silence gap between adjacent segments — within-word syllable gaps are shorter than between-word pauses)
   - Cap merges from growing a combined segment beyond the upper end of the confirmed-word duration distribution (prevents re-fusing genuinely separate words)

2. **Clustering review UI** — cost-minimized confirm/correct interface:
   - Rank queue by confidence (duration-ratio gate score + distance from nearest cluster centroid), not chronological order
   - Single-keypress confirm with auto-advance for the common case; batch-confirm for runs of high-confidence candidates
   - Boundary-drag correction reserved for flagged low-confidence items only
   - "Reject" requires one of three explicit exits (wrong cluster / junk / new word), not a binary confirm/reject

3. **VTC (Voice Type Classifier) offline evaluation** — run against already-annotated recordings, scored on label accuracy (not segment tightness), before any integration decision. This would improve the diarization stage feeding into vocalization segmentation (§3), independent of everything else in this doc.

---

## 8. Explicitly out of scope for this phase

- Word-level (as opposed to vocalization-level) segmentation as a prerequisite for any parent-facing feature
- Open-vocabulary automatic transcription of toddler speech (not achievable at this data volume; not worth pursuing)
- Forced alignment (not applicable without a reliable transcript — revisit only once per-child cluster labels are reliable enough to act as a mini-lexicon, which is a much later-stage idea, not part of this build)

---

## 9. Suggested build order

1. Vocalization-level feature extraction (duration, syllable count via existing DJW output, pitch contour) — no new segmentation work required
2. Small hand-labeled validation set across the 6 stages; confirm features separate classes before building the classifier
3. Stage classifier (start simple — rules or gradient-boosted trees)
4. Timeline/album view (parent-facing) using stage labels
5. New-word discovery trigger + parent prompt UI (extends existing `cluster_sounds.py` output)
6. Word combination detection (extends cluster matching + reuses gap heuristic)
7. In parallel, continue merge-back gating and review UI work (§7) — feeds back into (1)–(6) as precision improvements, not a gate on shipping them
