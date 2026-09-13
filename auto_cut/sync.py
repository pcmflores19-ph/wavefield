"""
Finds the time offset between two independently-recorded tracks.

Two recording setups need two different signals to match on:

  - Remote, no bleed (host and guest each on their own device): the tracks
    share no audio content at all, so alignment has to come from WHEN each
    person talks, not what either mic actually heard. `activity_signal`
    turns each track's speech intervals into a comparable on/off curve.
  - In-person, shared room: mics genuinely pick up each other's audio, so a
    loud moment shows up in every mic that heard it - this is what
    `level_signal` matches on, using the same per-frame loudness data
    voice_activity.py already computes for auto-mute. It's this app's
    equivalent of AudioAlign's acoustic fingerprint matching, built on data
    already on hand instead of a Chromaprint/Echoprint dependency.

Both signals are searched by the same engine (`_search`): a coarse pass at a
wide hop finds an approximate offset cheaply even over a two-hour recording,
then a fine pass refines around it. `find_offset` tries both signals and
returns whichever produced the more confident match, so the caller never has
to know or declare which recording setup produced the tracks.

Pure logic only - no Tk, no ffmpeg, no file I/O. Callers already have
everything this needs from the app's normal analysis pass
(voice_activity.speaking_intervals).
"""

from collections import namedtuple

import numpy as np

from voice_activity import HOP_SECONDS as LEVELS_HOP_SECONDS

# The fine grid both signal types are searched on. Coarse enough to keep the
# coarse-pass search cheap, fine enough that speech-pattern alignment doesn't
# need sub-syllable resolution.
HOP_SECONDS = 0.05

# The coarse pass runs at this much wider a hop - a two-hour recording is
# only ~7200 samples at 1s resolution, cheap to search over a wide window
# before ever touching the fine grid.
COARSE_HOP_SECONDS = 1.0

# How far around the coarse pass's best guess the fine pass searches, in
# seconds. Wide enough to correct any coarse-grid quantization error, narrow
# enough to stay cheap.
FINE_WINDOW_SECONDS = 2.0

# A match below this normalized score is not a match, regardless of how it
# compares to other candidate lags - guards against two near-silent or two
# near-constant tracks producing a confident-looking but meaningless peak.
MIN_SCORE = 0.15

# How far the best score must stand out from the pack (a robust z-score
# using median/MAD rather than mean/stdev, since most candidate lags score
# near zero and a few real near-misses can otherwise skew a plain stdev).
MIN_CONFIDENCE = 2.0

TrackActivity = namedtuple("TrackActivity", ["intervals", "duration", "levels_db"])
SyncResult = namedtuple("SyncResult", ["offset_seconds", "score", "confidence", "ambiguous", "method"])

_Candidate = namedtuple("_Candidate", ["method", "offset", "score", "confidence", "ambiguous"])


def activity_signal(intervals, duration, hop_seconds=HOP_SECONDS):
    """[0/1] per hop, 1 wherever any interval covers that instant."""
    n = max(1, int(np.ceil(duration / hop_seconds)))
    signal = np.zeros(n, dtype=np.float64)
    for start, end in intervals:
        a = max(0, int(np.floor(start / hop_seconds)))
        b = min(n, int(np.ceil(end / hop_seconds)))
        if b > a:
            signal[a:b] = 1.0
    return signal


def level_signal(levels_db, hop_seconds=HOP_SECONDS):
    """
    The per-frame loudness curve, downsampled from its native (finer) hop to
    the shared search grid via block-max - a loud instant anywhere in a
    coarser bin should still register as loud, unlike a block-mean, which
    would wash out a brief but distinctive peak.
    """
    levels_db = np.asarray(levels_db, dtype=np.float64)
    if levels_db.size == 0:
        return levels_db
    factor = max(1, round(hop_seconds / LEVELS_HOP_SECONDS))
    return _block_max(levels_db, factor)


def _block_max(x, factor):
    if factor <= 1:
        return x
    n = len(x) // factor
    if n == 0:
        return x[:0]
    return x[:n * factor].reshape(n, factor).max(axis=1)


def _block_mean(x, factor):
    if factor <= 1:
        return x
    n = len(x) // factor
    if n == 0:
        return x[:0]
    return x[:n * factor].reshape(n, factor).mean(axis=1)


def _center(x):
    if x.size == 0:
        return x
    return x - x.mean()


# A tiny overlap window can score deceptively high by pure chance - normalized
# cross-correlation divides out the very thing (sample count) that would
# otherwise make a coincidence look weak. Requiring a minimum fraction of the
# shorter signal's length keeps near-the-edge-of-the-search-range candidates
# from ever spuriously winning against a real, well-supported match.
MIN_OVERLAP_FRACTION = 0.2


def _score_at_lag(a, b, lag, min_overlap=0):
    """
    Normalized cross-correlation of a and b at integer sample `lag`, over
    only their actual overlap at that lag - not zero-padded, so a candidate
    near the edge of the search window isn't penalized for having less
    overlap than the true match. Scores 0.0 (never a winner) if that overlap
    is thinner than `min_overlap` samples.

    lag >= 0: b's index 0 aligns with a's index `lag` (b started later, or
    needs that many samples trimmed to line up). lag < 0: the reverse.
    """
    shift_a = max(0, -lag)
    shift_b = max(0, lag)
    n = min(len(a) - shift_a, len(b) - shift_b)
    if n <= 0 or n < min_overlap:
        return 0.0
    wa = a[shift_a:shift_a + n]
    wb = b[shift_b:shift_b + n]
    denom = np.linalg.norm(wa) * np.linalg.norm(wb)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(wa, wb) / denom)


def _search(a, b, hop_seconds, min_offset_seconds, max_offset_seconds):
    """Best-scoring integer-lag offset within [min, max], in seconds."""
    lo = int(round(min_offset_seconds / hop_seconds))
    hi = int(round(max_offset_seconds / hop_seconds))
    if hi < lo:
        lo, hi = hi, lo
    min_overlap = MIN_OVERLAP_FRACTION * min(len(a), len(b))
    offsets = np.arange(lo, hi + 1)
    scores = np.array([_score_at_lag(a, b, int(o), min_overlap) for o in offsets])
    best = int(np.argmax(scores))
    return float(offsets[best]) * hop_seconds, float(scores[best]), scores


def _confidence(best_score, scores):
    if best_score < MIN_SCORE:
        return 0.0, True
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median))) + 1e-6
    confidence = (best_score - median) / mad
    return confidence, confidence < MIN_CONFIDENCE


def _try_method(sig_a, sig_b, max_offset_seconds):
    """One signal type, coarse-then-fine. None if there's nothing to search."""
    if sig_a is None or sig_b is None or len(sig_a) == 0 or len(sig_b) == 0:
        return None

    # Mean, not max, for this downsampling - confirmed by direct
    # reproduction: for realistic bursty speech (short utterances close
    # together), block-max saturates nearly every coarse bin to 1 the
    # instant ANY speech falls in it, flattening the very timing structure
    # correlation needs to find the right neighborhood - the coarse pass
    # was landing tens of seconds from the true offset as a result, and the
    # fine pass's narrow refinement window around a wrong anchor could
    # never recover from it. Mean preserves how much of each bin was
    # speech, not just whether any of it was.
    coarse_factor = max(1, round(COARSE_HOP_SECONDS / HOP_SECONDS))
    coarse_a = _block_mean(_center(sig_a), coarse_factor)
    coarse_b = _block_mean(_center(sig_b), coarse_factor)
    if len(coarse_a) == 0 or len(coarse_b) == 0:
        return None
    coarse_offset, _coarse_score, _coarse_scores = _search(
        coarse_a, coarse_b, COARSE_HOP_SECONDS, -max_offset_seconds, max_offset_seconds)

    fine_a, fine_b = _center(sig_a), _center(sig_b)
    fine_offset, fine_score, fine_scores = _search(
        fine_a, fine_b, HOP_SECONDS,
        coarse_offset - FINE_WINDOW_SECONDS, coarse_offset + FINE_WINDOW_SECONDS)

    confidence, ambiguous = _confidence(fine_score, fine_scores)
    return fine_offset, fine_score, confidence, ambiguous


def find_offset(reference, target, max_offset_seconds=600.0):
    """
    The offset that best aligns `target` to `reference`, trying both the
    timing (speech on/off) and content (loudness curve) signals and keeping
    whichever is more confident. `reference`/`target` are TrackActivity
    (intervals, duration, levels_db) - levels_db may be None or empty (a
    project reopened via the fast reload path doesn't always have it - see
    recompute_scenes' own fallback in app.py for the same gap), in which
    case only the timing signal is tried for that pair.

    Sign convention: positive offset_seconds means `target` has extra
    leading material to TRIM; negative means it's missing lead-in and needs
    silence PADDED onto its front. ambiguous=True (offset_seconds is
    meaningless, always 0.0) only when neither signal produces a confident
    match - never applied automatically by anything in this module.
    """
    duration = max(reference.duration, target.duration)

    activity_a = activity_signal(reference.intervals, duration)
    activity_b = activity_signal(target.intervals, duration)
    timing = _try_method(activity_a, activity_b, max_offset_seconds)

    content = None
    if reference.levels_db is not None and target.levels_db is not None \
            and len(reference.levels_db) and len(target.levels_db):
        level_a = level_signal(reference.levels_db)
        level_b = level_signal(target.levels_db)
        content = _try_method(level_a, level_b, max_offset_seconds)

    candidates = []
    if timing is not None:
        candidates.append(_Candidate("timing", *timing))
    if content is not None:
        candidates.append(_Candidate("content", *content))

    if not candidates:
        return SyncResult(0.0, 0.0, 0.0, True, "none")

    confident = [c for c in candidates if not c.ambiguous]
    pool = confident if confident else candidates
    best = max(pool, key=lambda c: c.confidence)
    offset = 0.0 if best.ambiguous else best.offset
    return SyncResult(offset, best.score, best.confidence, best.ambiguous, best.method)
