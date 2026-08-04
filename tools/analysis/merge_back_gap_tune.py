#!/usr/bin/env python3
"""Measure DJW sibling gaps vs human-tag oracle; sweep max_gap / short_piece.

Caches baseline DJW cands + precomputed cut acoustics so policy sweeps are fast.

  tools/.venv/bin/python tools/analysis/merge_back_gap_tune.py
  tools/.venv/bin/python tools/analysis/merge_back_gap_tune.py --rebuild
"""

from __future__ import annotations

import json
import pickle
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

from ml_delta import LIBRARY, iou, load, overlap_ms, span
from resegment import (
    MergeBackParams,
    apply_merge_back,
    should_merge,
    _cut_features,
    _mono,
    _piece_intensity,
    _region_median,
)
from vad_segments import run_vad_on_audio
from djw_merge_back_eval import (
    DEFAULT_KITS,
    find_kits,
    filter_tags,
    load_audio,
    score_kit,
    session_median_tag_ms,
)

OUT = ANALYSIS_DIR / "out" / "merge_back_gap_tune.json"
CACHE = ANALYSIS_DIR / "out" / "merge_back_gap_baseline.pkl"

HIT_FRAC = 0.40
HIT_IOU = 0.25


def pctile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[int(round(p * (len(xs) - 1)))], 1)


def summarize(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "p10": pctile(xs, 0.10),
        "p25": pctile(xs, 0.25),
        "median": pctile(xs, 0.50),
        "p75": pctile(xs, 0.75),
        "p90": pctile(xs, 0.90),
        "mean": round(statistics.mean(xs), 1),
        "frac_le100": round(sum(1 for x in xs if x <= 100) / len(xs), 3),
        "frac_le200": round(sum(1 for x in xs if x <= 200) / len(xs), 3),
        "frac_le300": round(sum(1 for x in xs if x <= 300) / len(xs), 3),
        "frac_le400": round(sum(1 for x in xs if x <= 400) / len(xs), 3),
        "frac_le450": round(sum(1 for x in xs if x <= 450) / len(xs), 3),
        "frac_ge500": round(sum(1 for x in xs if x >= 500) / len(xs), 3),
    }


def dur_stats(cands: list[dict]) -> dict:
    durs = [float(c["endMs"]) - float(c["startMs"]) for c in cands]
    if not durs:
        return {"n": 0}
    durs.sort()
    return {
        "n": len(durs),
        "median": round(durs[len(durs) // 2], 1),
        "p25": round(durs[len(durs) // 4], 1),
        "p75": round(durs[(3 * len(durs)) // 4], 1),
        "mean": round(statistics.mean(durs), 1),
    }


def best_tag_hit(piece: dict, tags: list[dict]) -> tuple[int | None, float, float]:
    ps = span(piece)
    if not ps:
        return None, 0.0, 0.0
    pdur = max(1e-6, ps[1] - ps[0])
    best_i, best_ov, best_iou = None, 0.0, 0.0
    for i, t in enumerate(tags):
        ts = span(t)
        if not ts:
            continue
        ov = overlap_ms(ps, ts)
        if ov <= 0:
            continue
        frac = ov / pdur
        ii = iou(ps, ts)
        if frac > best_ov or (frac == best_ov and ii > best_iou):
            best_i, best_ov, best_iou = i, frac, ii
    return best_i, best_ov, best_iou


def oracle_label(a: dict, b: dict, tags: list[dict]) -> str:
    ia, fa, iou_a = best_tag_hit(a, tags)
    ib, fb, iou_b = best_tag_hit(b, tags)
    hit_a = ia is not None and (fa >= HIT_FRAC or iou_a >= HIT_IOU)
    hit_b = ib is not None and (fb >= HIT_FRAC or iou_b >= HIT_IOU)
    if hit_a and hit_b and ia == ib:
        return "should_merge"
    if hit_a and hit_b and ia != ib:
        return "should_keep"
    return "unknown"


def raw_cut_acoustics(audio, sr, a0, a1, b0, b1, cut_window_ms: float = 80.0) -> dict:
    """Policy-independent acoustics for a cut (gap/durs/prom/energy drop)."""
    gap = b0 - a1
    merged_dur = b1 - a0
    times, intensity = _piece_intensity(audio, sr, a0, b1)
    left_med = _region_median(times, intensity, a0, a1)
    right_med = _region_median(times, intensity, b0, b1)
    cut = 0.5 * (a1 + b0)
    win = cut_window_ms
    gap_lo, gap_hi = min(a1, b0), max(a1, b0)
    valley = _region_median(times, intensity, cut - win, cut + win)
    if gap_hi > gap_lo:
        valley = min(valley, _region_median(times, intensity, gap_lo, gap_hi))
    flank = min(left_med, right_med)
    prom = flank - valley
    return {
        "gapMs": round(gap, 1),
        "mergedDurMs": round(merged_dur, 1),
        "leftDurMs": round(a1 - a0, 1),
        "rightDurMs": round(b1 - b0, 1),
        "minDurMs": round(min(a1 - a0, b1 - b0), 1),
        "promDb": round(prom, 2),
        "energyDropDb": round(flank - valley, 2),
        "parentSpanId_l": None,
        "parentSpanId_r": None,
        "speaker_l": None,
        "speaker_r": None,
    }


def feats_for_params(raw: dict, params: MergeBackParams) -> dict:
    min_dur = raw["minDurMs"]
    short_abs = min_dur < params.short_piece_ms
    short_rel = False
    if (
        params.dur_ratio_max is not None
        and params.session_median_tag_ms is not None
        and params.session_median_tag_ms > 0
    ):
        short_rel = (min_dur / params.session_median_tag_ms) < params.dur_ratio_max
    return {
        "gapMs": raw["gapMs"],
        "mergedDurMs": raw["mergedDurMs"],
        "leftDurMs": raw["leftDurMs"],
        "rightDurMs": raw["rightDurMs"],
        "minDurMs": min_dur,
        "promDb": raw["promDb"],
        "energyOk": raw["energyDropDb"] <= params.energy_drop_db,
        "shortPiece": short_abs,
        "shortRel": short_rel,
        "clearlyShort": short_abs or short_rel,
        "weakValley": raw["promDb"] < params.weak_dip_db
        and raw["gapMs"] <= params.weak_sep_ms,
    }


def build_pair_index(baseline, audio, sr, tags) -> list[dict]:
    by_parent: dict[str, list[dict]] = {}
    for c in baseline:
        pid = c.get("parentSpanId")
        if pid:
            by_parent.setdefault(str(pid), []).append(c)
    pairs = []
    for pid, sibs in by_parent.items():
        sibs = sorted(sibs, key=lambda x: (float(x["startMs"]), float(x["endMs"])))
        for a, b in zip(sibs, sibs[1:]):
            raw = raw_cut_acoustics(
                audio,
                sr,
                float(a["startMs"]),
                float(a["endMs"]),
                float(b["startMs"]),
                float(b["endMs"]),
            )
            raw["parentSpanId_l"] = a.get("parentSpanId")
            raw["parentSpanId_r"] = b.get("parentSpanId")
            raw["speaker_l"] = a.get("speakerCluster")
            raw["speaker_r"] = b.get("speakerCluster")
            pairs.append(
                {
                    "raw": raw,
                    "oracle": oracle_label(a, b, tags),
                    "a": a,
                    "b": b,
                }
            )
    return pairs


def eval_pair(pair: dict, params: MergeBackParams) -> tuple[bool, str]:
    feats = feats_for_params(pair["raw"], params)
    left = {
        "parentSpanId": pair["raw"]["parentSpanId_l"],
        "speakerCluster": pair["raw"]["speaker_l"],
    }
    right = {
        "parentSpanId": pair["raw"]["parentSpanId_r"],
        "speakerCluster": pair["raw"]["speaker_r"],
    }
    return should_merge(left, right, feats, params)


def simulate_merge_back(
    baseline: list[dict], pairs: list[dict], params: MergeBackParams
) -> tuple[list[dict], dict]:
    """Greedy LTR merge using cached adjacent-cut acoustics (no audio).

    When A+B already merged and we consider next C, we reuse the B–C cut's
    gap/prom/energy and set leftDur to the running merged duration.
    """
    # Index pairs by (parent, right_startMs) for lookup
    pair_by_right: dict[tuple[str, float], dict] = {}
    for pair in pairs:
        pid = str(pair["raw"]["parentSpanId_l"])
        pair_by_right[(pid, float(pair["b"]["startMs"]))] = pair

    by_parent: dict[str, list[dict]] = {}
    orphans: list[dict] = []
    for c in baseline:
        pid = c.get("parentSpanId")
        if not pid:
            orphans.append(dict(c))
            continue
        by_parent.setdefault(str(pid), []).append(dict(c))

    stats = {
        "nMerges": 0,
        "nConsidered": 0,
        "rejectReasons": {},
        "mergeReasons": {},
    }
    out: list[dict] = list(orphans)
    for pid, sibs in by_parent.items():
        sibs = sorted(sibs, key=lambda x: (float(x["startMs"]), float(x["endMs"])))
        if len(sibs) == 1:
            out.append(sibs[0])
            continue
        acc = sibs[0]
        for nxt in sibs[1:]:
            stats["nConsidered"] += 1
            key = (pid, float(nxt["startMs"]))
            pair = pair_by_right.get(key)
            if pair is None:
                # Fallback: no cached pair — keep cut
                stats["rejectReasons"]["no_cache"] = (
                    stats["rejectReasons"].get("no_cache", 0) + 1
                )
                out.append(acc)
                acc = nxt
                continue
            raw = dict(pair["raw"])
            # Running left duration after prior merges
            raw["leftDurMs"] = round(
                float(acc["endMs"]) - float(acc["startMs"]), 1
            )
            raw["minDurMs"] = round(
                min(raw["leftDurMs"], raw["rightDurMs"]), 1
            )
            raw["mergedDurMs"] = round(
                float(nxt["endMs"]) - float(acc["startMs"]), 1
            )
            # Gap from running end to next start
            raw["gapMs"] = round(float(nxt["startMs"]) - float(acc["endMs"]), 1)
            feats = feats_for_params(raw, params)
            left = {
                "parentSpanId": acc.get("parentSpanId"),
                "speakerCluster": acc.get("speakerCluster"),
            }
            right = {
                "parentSpanId": nxt.get("parentSpanId"),
                "speakerCluster": nxt.get("speakerCluster"),
            }
            ok, reason = should_merge(left, right, feats, params)
            if ok:
                merged = dict(acc)
                merged["endMs"] = nxt["endMs"]
                if "uuid" in merged:
                    merged = {**merged, "uuid": merged["uuid"]}
                acc = merged
                stats["nMerges"] += 1
                stats["mergeReasons"][reason] = (
                    stats["mergeReasons"].get(reason, 0) + 1
                )
            else:
                stats["rejectReasons"][reason] = (
                    stats["rejectReasons"].get(reason, 0) + 1
                )
                out.append(acc)
                acc = nxt
        out.append(acc)
    out.sort(key=lambda x: (float(x["startMs"]), float(x["endMs"])))
    return out, stats


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--kits", nargs="+", default=DEFAULT_KITS)
    ap.add_argument("--diarization", default="none")
    ap.add_argument("--library", default=str(LIBRARY))
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    found = find_kits(Path(args.library).expanduser(), args.kits)
    print("kits:", [(n, p.name) for n, p in found], flush=True)

    if CACHE.exists() and not args.rebuild:
        print(f"loading cache {CACHE}", flush=True)
        kit_data = pickle.loads(CACHE.read_bytes())
        # Drop heavy pair audio refs if present; rebuild pairs below
    else:
        kit_data = []
        for needle, kit in found:
            tags_all = load(kit / "tags.json", "tags")
            tags = filter_tags(tags_all, "manual_excl_nonverbal")
            med = session_median_tag_ms(tags_all)
            print(
                f"\n=== {needle} tags_excl_nv={len(tags)} median_ms={med} ===",
                flush=True,
            )
            audio, sr, audio_path = load_audio(kit)
            t0 = time.time()
            baseline, stats = run_vad_on_audio(
                audio_path,
                diarization=args.diarization,
                resegment=True,
                merge_back=False,
            )
            print(
                f"  baseline DJW={len(baseline)} in {time.time()-t0:.1f}s "
                f"resegSplits={stats.get('resegSplits')}",
                flush=True,
            )
            print("  precomputing pair acoustics…", flush=True)
            t1 = time.time()
            pairs = build_pair_index(baseline, audio, sr, tags)
            print(f"  pairs={len(pairs)} in {time.time()-t1:.1f}s", flush=True)
            kit_data.append(
                {
                    "sessionName": needle,
                    "kit": kit.name,
                    "tags_all": tags_all,
                    "tags": tags,
                    "audio": audio,
                    "sr": sr,
                    "baseline": baseline,
                    "med": med,
                    "pairs": pairs,
                }
            )
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps(kit_data, protocol=pickle.HIGHEST_PROTOCOL))
        print(f"wrote cache {CACHE}", flush=True)

    # Ensure pairs exist (old cache without pairs)
    for kd in kit_data:
        if "pairs" not in kd:
            print(f"rebuilding pairs for {kd['sessionName']}…", flush=True)
            kd["pairs"] = build_pair_index(
                kd["baseline"], kd["audio"], kd["sr"], kd["tags"]
            )

    prod = MergeBackParams()
    report: dict = {
        "oracleRule": f"hit if piece_frac>={HIT_FRAC} or iou>={HIT_IOU}",
        "kits": {},
    }

    print("\n========== GAP DISTRIBUTIONS (oracle) ==========")
    for kd in kit_data:
        pairs = kd["pairs"]
        by_lab = {"should_merge": [], "should_keep": [], "unknown": [], "all": []}
        blocked_reasons: dict[str, int] = {}
        false_m = 0
        blocked_ex = []
        for pair in pairs:
            gap = pair["raw"]["gapMs"]
            by_lab[pair["oracle"]].append(gap)
            by_lab["all"].append(gap)
            ok, reason = eval_pair(pair, prod)
            if pair["oracle"] == "should_merge" and not ok:
                blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
                blocked_ex.append(
                    {
                        "gap": gap,
                        "minDur": pair["raw"]["minDurMs"],
                        "L": pair["raw"]["leftDurMs"],
                        "R": pair["raw"]["rightDurMs"],
                        "prom": pair["raw"]["promDb"],
                        "energyDrop": pair["raw"]["energyDropDb"],
                        "reason": reason,
                    }
                )
            if pair["oracle"] == "should_keep" and ok:
                false_m += 1
        print(f"\n--- {kd['sessionName']} n_pairs={len(pairs)} ---")
        for lab in ("should_merge", "should_keep", "unknown", "all"):
            print(f"  {lab:14s} {summarize(by_lab[lab])}")
        n_sm = len(by_lab["should_merge"])
        print(
            f"  should_merge blocked by prod: {sum(blocked_reasons.values())} / {n_sm} "
            f"reasons={blocked_reasons}"
        )
        print(f"  should_keep wrongly merged by prod: {false_m}")
        report["kits"][kd["sessionName"]] = {
            "nPairs": len(pairs),
            "gap_should_merge": summarize(by_lab["should_merge"]),
            "gap_should_keep": summarize(by_lab["should_keep"]),
            "gap_unknown": summarize(by_lab["unknown"]),
            "gap_all": summarize(by_lab["all"]),
            "prod_blocked_should_merge": blocked_reasons,
            "prod_false_merges_should_keep": false_m,
            "baselineDur": dur_stats(kd["baseline"]),
            "examples_blocked_should_merge": sorted(
                blocked_ex, key=lambda x: x["gap"]
            )[:20],
        }

    policies: list[tuple[str, MergeBackParams]] = []
    for gap in (100.0, 200.0, 300.0, 350.0, 400.0, 450.0):
        for short in (400.0, 450.0):
            policies.append(
                (
                    f"short{short:.0f}_gap{gap:.0f}",
                    MergeBackParams(
                        short_piece_ms=short,
                        max_gap_ms=gap,
                        require_clearly_short=True,
                    ),
                )
            )
    policies.append(
        (
            "legacy_s550_gap300",
            MergeBackParams(
                short_piece_ms=550.0,
                max_gap_ms=300.0,
                require_clearly_short=False,
            ),
        )
    )

    print("\n========== POLICY SWEEP ==========")
    print(
        f"{'policy':22s} {'kit':26s} {'mer':>4} {'nC':>5} {'med':>5} "
        f"{'exclNV':>6} {'dNV':>5} {'verb':>6} {'dV':>5} "
        f"{'smHit':>5} {'skFP':>4} {'tooSh':>5}",
        flush=True,
    )
    sweep_rows = []
    for name, params in policies:
        for kd in kit_data:
            p = MergeBackParams(
                **{**asdict(params), "session_median_tag_ms": kd["med"]}
            )
            # Fast oracle stats from cached acoustics
            sm_hit = sk_fp = 0
            n_sm = n_sk = 0
            for pair in kd["pairs"]:
                ok, _ = eval_pair(pair, p)
                if pair["oracle"] == "should_merge":
                    n_sm += 1
                    if ok:
                        sm_hit += 1
                elif pair["oracle"] == "should_keep":
                    n_sk += 1
                    if ok:
                        sk_fp += 1

            merged, mstats = simulate_merge_back(kd["baseline"], kd["pairs"], p)
            before = score_kit(kd["tags_all"], kd["baseline"])
            after = score_kit(kd["tags_all"], merged)
            bnv = before["manual_excl_nonverbal"]["iou50Pct"]
            anv = after["manual_excl_nonverbal"]["iou50Pct"]
            bv = before["verbal"]["iou50Pct"]
            av = after["verbal"]["iou50Pct"]
            dh = dur_stats(merged)
            row = {
                "policy": name,
                "sessionName": kd["sessionName"],
                "nMerges": mstats["nMerges"],
                "nCands": len(merged),
                "medianDur": dh.get("median"),
                "exclNv": anv,
                "dNv": round(anv - bnv, 1),
                "verbal": av,
                "dV": round(av - bv, 1),
                "oracleShouldMergeHit": sm_hit,
                "oracleShouldMergeN": n_sm,
                "oracleShouldKeepFP": sk_fp,
                "oracleShouldKeepN": n_sk,
                "tooShort": after["manual_excl_nonverbal"]["candTooShortPct"],
                "reject": mstats["rejectReasons"],
                "mergeReasons": mstats["mergeReasons"],
                "params": asdict(params),
            }
            sweep_rows.append(row)
            print(
                f"{name:22s} {kd['sessionName']:26s} {mstats['nMerges']:4d} "
                f"{len(merged):5d} {dh.get('median') or 0:5.0f} "
                f"{anv:6.1f} {anv-bnv:+5.1f} {av:6.1f} {av-bv:+5.1f} "
                f"{sm_hit}/{n_sm} {sk_fp:4d} "
                f"{after['manual_excl_nonverbal']['candTooShortPct']:5.1f}",
                flush=True,
            )

    report["sweep"] = sweep_rows
    baselines = {}
    for kd in kit_data:
        b = score_kit(kd["tags_all"], kd["baseline"])
        baselines[kd["sessionName"]] = {
            "nCands": len(kd["baseline"]),
            "dur": dur_stats(kd["baseline"]),
            "exclNv": b["manual_excl_nonverbal"]["iou50Pct"],
            "verbal": b["verbal"]["iou50Pct"],
            "tooShort": b["manual_excl_nonverbal"]["candTooShortPct"],
        }
    report["baselines"] = baselines

    print("\n========== PICK ==========")
    candidates = [
        r
        for r in sweep_rows
        if r["sessionName"] == "26_07_27__19:53:00"
        and r["params"]["require_clearly_short"]
        and r["params"]["max_gap_ms"] >= 200
    ]
    scored = []
    for r in candidates:
        n_sm = max(1, r["oracleShouldMergeN"])
        hit = r["oracleShouldMergeHit"] / n_sm
        score = (
            hit * 100.0
            - r["oracleShouldKeepFP"] * 4.0
            + r["verbal"] * 0.3
            + r["exclNv"] * 0.2
            + min(0.0, r["dV"]) * 2.5
            + min(0.0, r["dNv"]) * 1.0
        )
        st = next(
            (
                x
                for x in sweep_rows
                if x["policy"] == r["policy"]
                and x["sessionName"] == "26_07_05__00:00:00"
            ),
            None,
        )
        if st:
            score += st["dNv"] * 1.5 + st["exclNv"] * 0.1
            score -= st["oracleShouldKeepFP"] * 2.0
            # Prefer recovering within-word merges on St too
            n_sm_st = max(1, st["oracleShouldMergeN"])
            score += (st["oracleShouldMergeHit"] / n_sm_st) * 20.0
        scored.append((score, r, st))
    scored.sort(key=lambda x: -x[0])
    for score, r, st in scored[:10]:
        print(
            f"  score={score:6.1f}  {r['policy']:22s}  EN dV={r['dV']:+.1f} "
            f"dNV={r['dNv']:+.1f} sm={r['oracleShouldMergeHit']}/{r['oracleShouldMergeN']} "
            f"fp={r['oracleShouldKeepFP']}"
            + (
                f"  | St dNV={st['dNv']:+.1f} sm={st['oracleShouldMergeHit']}/{st['oracleShouldMergeN']} "
                f"fp={st['oracleShouldKeepFP']}"
                if st
                else ""
            ),
            flush=True,
        )

    # Manual preference: gap under ~500ms word break → 350 or 400;
    # avoid 450 if should_keep FP climbs. Prefer short450 if it recovers
    # not_clearly_short without big EN verbal regression.
    if scored:
        # Prefer among top few: gap 300-400, minimize EN verbal drop, maximize sm hit
        top = scored[:6]
        preferred = None
        for score, r, st in top:
            g = r["params"]["max_gap_ms"]
            if 300 <= g <= 400 and r["oracleShouldKeepFP"] <= 3:
                preferred = (score, r, st)
                break
        if preferred is None:
            preferred = scored[0]
        best = preferred[1]
        report["recommended"] = {
            "policy": best["policy"],
            "params": {
                "short_piece_ms": best["params"]["short_piece_ms"],
                "max_gap_ms": best["params"]["max_gap_ms"],
                "require_clearly_short": best["params"]["require_clearly_short"],
            },
            "english": {
                k: best[k]
                for k in (
                    "nMerges",
                    "nCands",
                    "medianDur",
                    "exclNv",
                    "dNv",
                    "verbal",
                    "dV",
                    "oracleShouldMergeHit",
                    "oracleShouldMergeN",
                    "oracleShouldKeepFP",
                    "tooShort",
                )
            },
            "st75": preferred[2]
            and {
                k: preferred[2][k]
                for k in (
                    "nMerges",
                    "nCands",
                    "medianDur",
                    "exclNv",
                    "dNv",
                    "verbal",
                    "dV",
                    "oracleShouldMergeHit",
                    "oracleShouldMergeN",
                    "oracleShouldKeepFP",
                    "tooShort",
                )
            },
            "score": round(preferred[0], 2),
            "rationale": (
                "Within-word DJW sibling gaps (oracle should_merge) are almost all "
                "<300–400ms; inter-word (should_keep) median ~320–430ms with many ≥500ms. "
                "Raising max_gap_ms from 100 toward 350–400 recovers syllable glue without "
                "crossing the ~0.5s word-break regime. short_piece also matters for "
                "≥400ms syllable shards."
            ),
        }
        print(
            f"\nRECOMMENDED: {best['policy']} → {report['recommended']['params']}",
            flush=True,
        )

        # Verify top pick with real audio merge-back (energy on merged left differs slightly)
        print("\n========== VERIFY (real apply_merge_back) ==========")
        verify_policies = [
            ("prod_short400_gap100", MergeBackParams()),
            (
                best["policy"],
                MergeBackParams(
                    short_piece_ms=best["params"]["short_piece_ms"],
                    max_gap_ms=best["params"]["max_gap_ms"],
                    require_clearly_short=True,
                ),
            ),
        ]
        # Also verify short450_gap350 and short400_gap350 if different
        for alt_name, alt_p in (
            ("short450_gap350", MergeBackParams(short_piece_ms=450, max_gap_ms=350)),
            ("short400_gap350", MergeBackParams(short_piece_ms=400, max_gap_ms=350)),
            ("short450_gap400", MergeBackParams(short_piece_ms=450, max_gap_ms=400)),
        ):
            if alt_name not in {x[0] for x in verify_policies}:
                verify_policies.append((alt_name, alt_p))
        verify_rows = []
        for vname, vp in verify_policies:
            for kd in kit_data:
                p = MergeBackParams(
                    **{**asdict(vp), "session_median_tag_ms": kd["med"]}
                )
                merged, mstats = apply_merge_back(
                    kd["baseline"], kd["audio"], kd["sr"], p
                )
                before = score_kit(kd["tags_all"], kd["baseline"])
                after = score_kit(kd["tags_all"], merged)
                dh = dur_stats(merged)
                row = {
                    "policy": vname,
                    "sessionName": kd["sessionName"],
                    "nMerges": mstats["nMerges"],
                    "nCands": len(merged),
                    "medianDur": dh.get("median"),
                    "exclNv": after["manual_excl_nonverbal"]["iou50Pct"],
                    "dNv": round(
                        after["manual_excl_nonverbal"]["iou50Pct"]
                        - before["manual_excl_nonverbal"]["iou50Pct"],
                        1,
                    ),
                    "verbal": after["verbal"]["iou50Pct"],
                    "dV": round(
                        after["verbal"]["iou50Pct"] - before["verbal"]["iou50Pct"],
                        1,
                    ),
                    "tooShort": after["manual_excl_nonverbal"]["candTooShortPct"],
                    "reject": mstats["rejectReasons"],
                }
                verify_rows.append(row)
                print(
                    f"  {vname:22s} {kd['sessionName']:26s} merges={mstats['nMerges']:4d} "
                    f"nC={len(merged):4d} med={dh.get('median')} "
                    f"exclNV={row['exclNv']} ({row['dNv']:+.1f}) "
                    f"verb={row['verbal']} ({row['dV']:+.1f}) "
                    f"tooShort={row['tooShort']}",
                    flush=True,
                )
        report["verify"] = verify_rows

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nwrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
