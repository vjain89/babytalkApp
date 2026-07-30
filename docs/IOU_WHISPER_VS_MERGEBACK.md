# Whisper word timestamps vs DJW merge-back

**Kits:** `26_07_27__19:53:00` (St_87, English-leaning), `26_07_05__00:00:00` (St_75, Swiss German)  
**Sources:** `tools/analysis/out/whisper_word_ts.json`, `merge_back_eval.json`

## What we’re trying to solve

Human reviewers tag **whole words**. Our ML pipeline (VAD → speaker clustering → pause-split → DJW resegment) often cuts a single word into **syllable-sized boxes** — e.g. *tea|cher*. Those too-short boxes overlap the tag but fail the north-star metric **IoU ≥ 0.5** (how much the machine box and the human box agree). Tuning split knobs does not fix this; we need a way to reunite over-cuts, or an external word-boundary signal.

## What Whisper word timestamps are

Whisper (ASR) can emit not only a transcript but **per-word start/end times**. The idea: use those intervals as word-sized boxes instead of trusting DJW syllable cuts. We ran `faster-whisper/base` on padded windows around manual word tags and scored how well Whisper’s boxes match human spans (by overlap IoU) and text (fuzzy match).

## Why Whisper could help / why it’s hard here

If Whisper’s boundaries landed near human words, we could use them as candidate boxes or as merge targets — even when spelling is wrong. In practice this domain is hostile: **baby speech + Swiss German** → ASR text is unreliable. Whisper also **over-splits** words into multiple tokens, the same failure class as DJW syllables.

## Whisper results

Manual verbal word tags (excl. non-verbal / `ml_confirmed`); **n = 179** pooled.

| Kit | n | Any overlap | Text match | IoU ≥ 0.5 |
|-----|--:|-----------:|-----------:|----------:|
| `26_07_27__19:53:00` | 107 | 95.3% | 11.2% | **30.8%** |
| `26_07_05__00:00:00` | 72 | 93.1% | 5.6% | **18.1%** |
| **Pooled** | **179** | **94.4%** | **8.9%** | **25.7%** |

Coverage looks fine (~94% any overlap), but word-box quality does not: pooled IoU ≥ 0.5 is only **25.7%**, worse than fresh DJW (~50–57%). Dominant error is `word_split` (40.8%). Text match is ~9%, so Whisper cannot gate merges by spelling either.

**Verdict:** Whisper word timestamps alone are **not sufficient**.

## What DJW merge-back is

After DJW cuts siblings under the same parent/speaker, **merge-back** walks adjacent pieces and glues weak cuts back together when energy looks continuous (short piece and/or weak intensity valley, near-abut gap, under a max merged duration). Complexity **M**. Main risk: re-gluing real multi-word phrases when valleys between words are shallow.

## Merge-back results

North star = manual tags excl. non-verbal. Pooled before → after:

| Metric | Before | After |
|--------|-------:|------:|
| IoU ≥ 0.5 | **49.2%** | **53.6%** (+4.4 pp) |
| Median duration ratio (cand/tag) | 0.65 | 0.93 |
| Cand too-short | 69.4% | 51.8% |
| Candidates | 1384 | 1166 |

Any-overlap stayed flat at 93.9%. **Caveat:** broader “verbal” tags dipped pooled IoU ≥ 0.5 from **65.6% → 61.1%**, mostly on the English-leaning kit where many spans were already well-sized — policy may need stricter max-merged / short-piece gates there.

## Recommendation

**Prioritize merge-back in the pipeline** as the primary fix for syllable over-splits. **Do not use Whisper as a replacement** for word-level boxes; at best treat it as an optional weak prior where a single high-confidence token already aligns. Ship a conservative merge policy, watch the verbal-all regression, and only revisit larger ASR / forced-alignment if merge-back plateaus.
