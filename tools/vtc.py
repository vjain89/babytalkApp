"""Voice Type Classifier (VTC) stage for BabyTalk ML candidates.

VTC ([LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC)) does frame-level *role*
classification on child-centered audio:

    KCHI  key child (closest / target child)
    OCH   other child
    FEM   adult female
    MAL   adult male

Two ways it plugs into the pipeline (see ``vad_segments.build_candidates``):

``segmentation=vtc-first``
    VTC owns both *where* speech is and *who-type* it is. Energy VAD and ECAPA
    are skipped; VTC RTTM turns become the parent spans, then pause-split /
    syllable resegment / speechlike gate still run.

``diarization=vtc``
    Stage-1 energy VAD still finds regions; VTC roles cut those regions the way
    ECAPA clusters used to (role change = split). Prefer ``vtc-first`` for the
    role-native path; this mode is for A/B against VAD+ECAPA with the same
    stage-1 gate.

Requires a local checkout of LAAC-LSCP/VTC with ``uv sync`` already done
(default ``~/.cache/babytalk/VTC``, override with ``BABYTALK_VTC_ROOT``).
Audio is converted to 16 kHz mono WAV before inference. Results are cached
under ``~/.cache/babytalk/vtc_predictions/`` keyed by audio content hash.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install deps: pip install numpy\n" + str(e)) from e

TARGET_SR = 16_000
ROLE_LABELS = ("KCHI", "OCH", "FEM", "MAL")
ROLE_TO_SPEAKER = {
    "KCHI": "Baby",
    "OCH": "Other",
    "FEM": "Parent",
    "MAL": "Parent",
}
DEFAULT_VTC_ROOT = Path.home() / ".cache" / "babytalk" / "VTC"
CACHE_ROOT = Path.home() / ".cache" / "babytalk" / "vtc_predictions"


@dataclass
class VtcTurn:
    start_ms: float
    end_ms: float
    role: str  # KCHI / OCH / FEM / MAL
    confidence: float = 1.0

    @property
    def speaker(self) -> str:
        return ROLE_TO_SPEAKER.get(self.role, "Other")


@dataclass
class VtcResult:
    turns: list[VtcTurn]
    ok: bool = False
    error: str | None = None
    note: str | None = None
    backend: str = "vtc"
    cache_hit: bool = False
    output_dir: str | None = None
    elapsed_sec: float | None = None
    stats: dict | None = None


def vtc_root() -> Path:
    env = (os.environ.get("BABYTALK_VTC_ROOT") or "").strip()
    return Path(env).expanduser() if env else DEFAULT_VTC_ROOT


def vtc_available() -> tuple[bool, str]:
    """Return (ok, detail) describing whether VTC can be run."""
    root = vtc_root()
    if not root.is_dir():
        return False, f"VTC checkout missing at {root} (set BABYTALK_VTC_ROOT)"
    ckpt = root / "VTC-2" / "model" / "best.ckpt"
    if not ckpt.is_file() or ckpt.stat().st_size < 1_000_000:
        return False, f"VTC checkpoint missing/incomplete at {ckpt}"
    uv = shutil.which("uv")
    if not uv:
        return False, "uv not on PATH (needed to run VTC)"
    return True, f"ready at {root}"


def prefer_device() -> str:
    env = (os.environ.get("BABYTALK_VTC_DEVICE") or "").strip().lower()
    if env in ("cpu", "mps", "cuda", "gpu"):
        return env
    # Apple Silicon: MPS is usually fastest for this stack when available.
    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(audio)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32, copy=False)
    if sr == TARGET_SR:
        return x
    try:
        from scipy.signal import resample_poly
        import math

        g = math.gcd(int(sr), TARGET_SR)
        return resample_poly(x, TARGET_SR // g, int(sr) // g).astype(np.float32)
    except ImportError:
        n_out = int(round(len(x) * TARGET_SR / float(sr)))
        if n_out <= 1:
            return x[:1].astype(np.float32)
        idx = np.linspace(0.0, len(x) - 1.0, num=n_out)
        return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def write_wav_16k_mono(path: Path, audio16k: np.ndarray) -> None:
    """Write float audio as 16-bit PCM WAV (VTC / ffmpeg-friendly)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio16k, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(pcm.tobytes())


def _audio_hash(audio16k: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(b"vtc-16k-v1")
    h.update(int(TARGET_SR).to_bytes(4, "little"))
    # Hash a downsampled view so we don't hash multi-minute PCM twice slowly.
    step = max(1, len(audio16k) // 250_000)
    h.update(np.ascontiguousarray(audio16k[::step]).tobytes())
    h.update(len(audio16k).to_bytes(8, "little"))
    return h.hexdigest()[:16]


def parse_rttm(path: Path) -> list[VtcTurn]:
    """Parse a VTC RTTM file into role turns (ms)."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    turns: list[VtcTurn] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # SPEAKER <uri> 1 <start_s> <dur_s> <NA> <NA> <label> <NA> <NA>
        if len(parts) < 8:
            continue
        try:
            start_s = float(parts[3])
            dur_s = float(parts[4])
        except ValueError:
            continue
        label = parts[7].upper()
        if label not in ROLE_LABELS:
            # Some RTTMs put the label in field 8; tolerate that.
            label = parts[8].upper() if len(parts) > 8 else label
        if label not in ROLE_LABELS:
            continue
        if dur_s <= 0:
            continue
        turns.append(
            VtcTurn(
                start_ms=start_s * 1000.0,
                end_ms=(start_s + dur_s) * 1000.0,
                role=label,
            )
        )
    turns.sort(key=lambda t: (t.start_ms, t.end_ms))
    return turns


def _find_rttm(pred_dir: Path, stem: str) -> Path | None:
    for sub in ("rttm", "raw_rttm"):
        cand = pred_dir / sub / f"{stem}.rttm"
        if cand.exists():
            return cand
    # Flat fallback
    cand = pred_dir / f"{stem}.rttm"
    return cand if cand.exists() else None


def _vtc_env() -> dict:
    """Env for the VTC subprocess.

    TorchCodec needs Homebrew FFmpeg dylibs (and Torch's ``libc10``) on
    ``DYLD_LIBRARY_PATH``. Conda's FFmpeg/libiconv on the path breaks loading
    (missing ``_iconv``), so we deliberately put Homebrew + torch first and
    strip conda lib dirs out of the inherited DYLD path.
    """
    env = os.environ.copy()
    root = vtc_root()
    torch_lib = root / ".venv" / "lib" / "python3.13" / "site-packages" / "torch" / "lib"
    if not torch_lib.is_dir():
        # Fall back to whatever python the uv env uses.
        for cand in (root / ".venv" / "lib").glob("python*/site-packages/torch/lib"):
            torch_lib = cand
            break
    brew_lib = Path("/opt/homebrew/lib")
    parts = []
    if brew_lib.is_dir():
        parts.append(str(brew_lib))
    if torch_lib.is_dir():
        parts.append(str(torch_lib))

    def _clean(path_val: str | None) -> list[str]:
        out = []
        for p in (path_val or "").split(":"):
            if not p:
                continue
            # Drop conda/miniconda lib dirs — their libiconv breaks brew ffmpeg.
            low = p.lower()
            if "miniconda" in low or "anaconda" in low or "/conda/" in low:
                continue
            out.append(p)
        return out

    cleaned = _clean(env.get("DYLD_LIBRARY_PATH")) + _clean(
        env.get("DYLD_FALLBACK_LIBRARY_PATH")
    )
    for p in cleaned:
        if p not in parts:
            parts.append(p)
    joined = ":".join(parts)
    env["DYLD_LIBRARY_PATH"] = joined
    env["DYLD_FALLBACK_LIBRARY_PATH"] = joined
    # Prefer Homebrew ffmpeg on PATH over an older conda build.
    brew_bin = "/opt/homebrew/bin"
    path = env.get("PATH", "")
    if brew_bin not in path.split(":"):
        env["PATH"] = f"{brew_bin}:{path}"
    # Keep OpenMP from fighting itself under uv+torch.
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", env.get("OMP_NUM_THREADS", "4"))
    return env


def run_vtc_inference(
    audio: np.ndarray,
    sr: int,
    *,
    device: str | None = None,
    cache: bool = True,
    min_duration_on_s: float = 0.1,
    min_duration_off_s: float = 0.1,
) -> VtcResult:
    """Run VTC on an in-memory waveform; return role turns in source time."""
    import time

    ok, detail = vtc_available()
    if not ok:
        return VtcResult(turns=[], ok=False, error=detail)

    root = vtc_root()
    audio16k = to_mono_16k(audio, sr)
    if len(audio16k) < TARGET_SR // 10:
        return VtcResult(turns=[], ok=True, note="audio too short", stats={"nTurns": 0})

    key = _audio_hash(audio16k)
    cache_dir = CACHE_ROOT / key
    rttm = _find_rttm(cache_dir, "audio") if cache else None
    if rttm is not None:
        turns = parse_rttm(rttm)
        return VtcResult(
            turns=turns,
            ok=True,
            cache_hit=True,
            output_dir=str(cache_dir),
            note="cache hit",
            stats=_turn_stats(turns),
        )

    device = device or prefer_device()
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="babytalk-vtc-") as tmp:
        tmp_path = Path(tmp)
        wav_dir = tmp_path / "wavs"
        wav_dir.mkdir()
        wav_path = wav_dir / "audio.wav"
        write_wav_16k_mono(wav_path, audio16k)
        out_dir = tmp_path / "predictions"
        out_dir.mkdir()

        cmd = [
            shutil.which("uv") or "uv",
            "run",
            "scripts/infer.py",
            "--wavs",
            str(wav_dir),
            "--output",
            str(out_dir),
            "--device",
            device,
            "--min_duration_on_s",
            str(min_duration_on_s),
            "--min_duration_off_s",
            str(min_duration_off_s),
            "--batch_size",
            os.environ.get("BABYTALK_VTC_BATCH", "64"),
        ]
        run_env = _vtc_env()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
                env=run_env,
            )
        except OSError as e:
            return VtcResult(turns=[], ok=False, error=f"failed to launch VTC: {e}")

        if proc.returncode != 0:
            # Retry once on CPU if MPS/CUDA failed.
            err = (proc.stderr or proc.stdout or "").strip()
            if device != "cpu":
                cmd_cpu = list(cmd)
                idx = cmd_cpu.index("--device")
                cmd_cpu[idx + 1] = "cpu"
                proc = subprocess.run(
                    cmd_cpu,
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    check=False,
                    env=run_env,
                )
                device = "cpu"
                err = (proc.stderr or proc.stdout or err).strip()
            if proc.returncode != 0:
                tail = err[-1200:] if err else "(no stderr)"
                return VtcResult(
                    turns=[],
                    ok=False,
                    error=f"VTC infer exit {proc.returncode}: {tail}",
                )

        rttm = _find_rttm(out_dir, "audio")
        if rttm is None:
            return VtcResult(
                turns=[],
                ok=False,
                error="VTC finished but no RTTM was written",
                note=(proc.stdout or "")[-400:],
            )
        turns = parse_rttm(rttm)

        if cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            dest_rttm = cache_dir / "rttm"
            dest_rttm.mkdir(exist_ok=True)
            shutil.copy2(rttm, dest_rttm / "audio.rttm")
            # Also keep csv if present for debugging.
            csv = out_dir / "rttm.csv"
            if csv.exists():
                shutil.copy2(csv, cache_dir / "rttm.csv")

    elapsed = time.time() - t0
    return VtcResult(
        turns=turns,
        ok=True,
        cache_hit=False,
        output_dir=str(cache_dir) if cache else None,
        elapsed_sec=round(elapsed, 1),
        note=f"device={device}",
        stats={**_turn_stats(turns), "device": device},
    )


def _turn_stats(turns: list[VtcTurn]) -> dict:
    by_role: dict[str, int] = {}
    for t in turns:
        by_role[t.role] = by_role.get(t.role, 0) + 1
    return {
        "nTurns": len(turns),
        "byRole": by_role,
        "totalSpeechMs": round(sum(t.end_ms - t.start_ms for t in turns), 1),
    }


def intersect_turns_with_regions(
    turns: list[VtcTurn],
    regions: list[tuple[float, float]],
    *,
    min_ms: float = 250.0,
) -> list[VtcTurn]:
    """Keep the parts of VTC turns that overlap energy-VAD regions."""
    out: list[VtcTurn] = []
    for turn in turns:
        for rs, re_ in regions:
            s = max(turn.start_ms, rs)
            e = min(turn.end_ms, re_)
            if e - s >= min_ms:
                out.append(
                    VtcTurn(
                        start_ms=s,
                        end_ms=e,
                        role=turn.role,
                        confidence=turn.confidence,
                    )
                )
    out.sort(key=lambda t: (t.start_ms, t.end_ms))
    return out
