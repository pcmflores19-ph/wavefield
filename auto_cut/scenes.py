"""
Which camera is on screen, and when.

A vodcast here is three recordings of the same conversation: V1 the host alone,
V2 the guest alone, and V3 the merged shot with both already in frame. V3 is
picture only - its audio is the same two voices again and would double every
word - so it never becomes a speaker track.

The decision needs no new analysis. `silence_detector.active_intervals_by_lane`
already works out who is genuinely talking at each moment, including two people
at once, for auto-mute. A camera cut asks the same question:

    host alone   -> V1
    guest alone  -> V2
    both, or neither -> V3

An earlier attempt at switching described it in FCPXML and asked DaVinci
Resolve to honour it; Resolve rearranged the clips instead, and the free
edition blocks the scripting API that would have made it work. This time the
cut is rendered by us in the video export, where nothing can second-guess it.
"""

import hashlib
import random

HOST, GUEST, BOTH = 0, 1, 2

# Below this a shot is not a shot. Without it, a "mm-hm" in the middle of the
# other person's sentence cuts away and back inside a few frames, which reads
# as a glitch rather than an edit.
DEFAULT_MIN_SHOT_SECONDS = 2.0

# Sitting on one face for minutes on end is the other way switching looks
# wrong. Past this, cut away to the merged shot briefly and come back - the
# standard cutaway, and the reason V3 exists. 0 disables it.
DEFAULT_MAX_SHOT_SECONDS = 25.0


def scene_timeline(active_by_lane, duration, min_shot_seconds=None,
                   hop_seconds=0.01, max_shot_seconds=None):
    """
    [(camera, start, end)] covering 0..duration with no gaps.

    `active_by_lane` is [host_intervals, guest_intervals] from
    active_intervals_by_lane - who is really talking, bleed already excluded.
    """
    if min_shot_seconds is None:
        min_shot_seconds = DEFAULT_MIN_SHOT_SECONDS
    if max_shot_seconds is None:
        max_shot_seconds = DEFAULT_MAX_SHOT_SECONDS
    if duration <= 0:
        return []

    host = list(active_by_lane[0]) if len(active_by_lane) > 0 else []
    guest = list(active_by_lane[1]) if len(active_by_lane) > 1 else []

    # Every instant where either state changes becomes a candidate boundary.
    edges = {0.0, float(duration)}
    for intervals in (host, guest):
        for start, end in intervals:
            edges.add(max(0.0, min(float(duration), start)))
            edges.add(max(0.0, min(float(duration), end)))
    bounds = sorted(edges)

    raw = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a <= 0:
            continue
        middle = (a + b) / 2.0
        host_on = _covers(host, middle)
        guest_on = _covers(guest, middle)
        if host_on and not guest_on:
            camera = HOST
        elif guest_on and not host_on:
            camera = GUEST
        else:
            camera = BOTH          # talking together, or nobody talking
        raw.append((camera, a, b))

    scenes = _enforce_minimum(_merge_runs(raw), min_shot_seconds)
    return _enforce_maximum(scenes, max_shot_seconds, min_shot_seconds)


def _covers(intervals, moment):
    for start, end in intervals:
        if start <= moment < end:
            return True
    return False


def _merge_runs(scenes):
    """Joins neighbouring blocks that are on the same camera."""
    merged = []
    for camera, start, end in scenes:
        if merged and merged[-1][0] == camera and abs(merged[-1][2] - start) < 1e-6:
            merged[-1] = (camera, merged[-1][1], end)
        else:
            merged.append((camera, start, end))
    return merged


def _enforce_minimum(scenes, min_shot_seconds):
    """
    Absorbs anything too short into the shot before it.

    Repeated until nothing changes: removing one short shot can leave its
    neighbours adjacent and on the same camera, which then merge into one and
    may reveal another short shot beside them.
    """
    if min_shot_seconds <= 0 or not scenes:
        return scenes

    while True:
        for index, (camera, start, end) in enumerate(scenes):
            if end - start >= min_shot_seconds or len(scenes) == 1:
                continue
            if index > 0:
                previous = scenes[index - 1]
                scenes[index - 1] = (previous[0], previous[1], end)
            else:
                following = scenes[1]
                scenes[1] = (following[0], start, following[2])
            del scenes[index]
            scenes = _merge_runs(scenes)
            break
        else:
            return scenes


def _shot_rng(camera, start, end):
    """
    A private RNG for one shot's split decisions, seeded from the shot's own
    identity rather than a global generator - so recomputing scenes for an
    unrelated reason never reshuffles cuts already made for this one. Uses
    sha256 rather than Python's built-in hash(), which is salted per process
    and would make the same shot split differently between runs.
    """
    key = f"{camera}:{start:.6f}:{end:.6f}"
    seed = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    return random.Random(seed)


def _enforce_maximum(scenes, max_shot_seconds, min_shot_seconds):
    """
    Breaks up any shot that outstays its welcome.

    A long stretch on one camera is rebuilt as alternating solo/merged pairs:
    a brief return to the person talking, then a longer hold on the merged
    view, sized so the merged view reads as about 70% of the stretch and the
    talking camera about 30% - the emphasis a long monologue should actually
    have, not a brief cutaway to the merged shot in an otherwise-solo video.
    Each piece is still bounded by max_shot_seconds on its own, so neither
    camera can hold the screen indefinitely just because it is the majority
    one. The draw is seeded from the shot's own (camera, start, end), so
    recomputing scenes elsewhere on the timeline never reshuffles cuts
    already made here.

    Shots already on V3 are left alone: there is nowhere more neutral to go.
    """
    if not max_shot_seconds or max_shot_seconds <= 0:
        return scenes

    # A 0 (or otherwise sub-1s) minimum is a valid setting for absorbing
    # short reaction shots, but piece length shouldn't inherit it - drawing
    # from [0, max] would let a 0.1s flicker of a shot through.
    chunk_floor = min(max(1.0, min_shot_seconds), max_shot_seconds)
    # The merged view holds the screen roughly 7x as long as each return to
    # the talking camera - 70/30 expressed as a length ratio within each
    # solo-then-merged pair, rather than decided cutaway by cutaway.
    solo_ratio = 0.3 / 0.7

    out = []
    for camera, start, end in scenes:
        if camera == BOTH or (end - start) <= max_shot_seconds:
            out.append((camera, start, end))
            continue

        rng = _shot_rng(camera, start, end)
        position = start
        while position < end:
            remaining = end - position
            if remaining <= max_shot_seconds:
                # Short enough to just be one more shot on the camera
                # that's already talking - no need to force in a cutaway
                # for a tail this short.
                out.append((camera, position, end))
                break

            both_len = rng.uniform(chunk_floor, max_shot_seconds)
            solo_len = max(chunk_floor, both_len * solo_ratio)
            if 0 < remaining - (solo_len + both_len) < chunk_floor:
                # Don't leave a dangling tail thinner than a piece is
                # allowed to be after this pair - fold it into the merged
                # piece instead.
                both_len = remaining - solo_len

            out.append((camera, position, position + solo_len))
            position += solo_len
            if position >= end:
                break
            both_len = min(both_len, end - position)
            out.append((BOTH, position, position + both_len))
            position += both_len
    return _merge_runs(out)


# ------------------------------------------------------------- hand editing

def apply_scene_edits(scenes, edits):
    """
    Replays ordered hand edits over the automatic timeline; the latest wins.

    Assignment rather than add/remove, so this cannot reuse
    silence_detector.apply_range_edits. Edits are
    ("scene", camera, start, end), where camera None means "back to automatic"
    - recorded as an edit of its own rather than deleting history, so undo
    still steps back through it.
    """
    result = list(scenes)
    for edit in edits or []:
        try:
            _kind, camera, start, end = edit
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        if camera is None:
            result = _restore_automatic(result, scenes, float(start), float(end))
        else:
            result = _assign(result, int(camera), float(start), float(end))
    return _merge_runs(result)


def _restore_automatic(current, base_scenes, start, end):
    """Replaces [start, end) in current with whatever was in base_scenes."""
    out = []
    for existing, a, b in current:
        if b <= start or a >= end:
            out.append((existing, a, b))
            continue
        if a < start:
            out.append((existing, a, start))
        if b > end:
            out.append((existing, end, b))
    for auto_cam, a, b in base_scenes:
        oa = max(a, start)
        ob = min(b, end)
        if ob > oa:
            out.append((auto_cam, oa, ob))
    out.sort(key=lambda s: s[1])
    return out


def _assign(scenes, camera, start, end):
    """Forces `camera` over [start, end), splitting whatever was there."""
    out = []
    for existing, a, b in scenes:
        if b <= start or a >= end:
            out.append((existing, a, b))
            continue
        if a < start:
            out.append((existing, a, start))
        if b > end:
            out.append((existing, end, b))
    out.append((camera, start, end))
    out.sort(key=lambda s: s[1])
    return out


# --------------------------------------------------- what the renderer needs

def apply_to_keep_ranges(scenes, keep_ranges):
    """
    Clips the scene timeline to the surviving edit.

    A keep range routinely spans several shots, so this intersection - not the
    scenes and not the keep ranges alone - is what the video export consumes.
    Returns [(camera, start, end)] in timeline order.
    """
    pieces = []
    for keep_start, keep_end in keep_ranges:
        for camera, start, end in scenes:
            a = max(start, keep_start)
            b = min(end, keep_end)
            if b > a:
                pieces.append((camera, a, b))
    pieces.sort(key=lambda p: p[1])
    return pieces


def summarize(scenes):
    """Seconds on each camera, for the UI."""
    totals = {HOST: 0.0, GUEST: 0.0, BOTH: 0.0}
    for camera, start, end in scenes:
        totals[camera] = totals.get(camera, 0.0) + (end - start)
    return {"cuts": max(0, len(scenes) - 1),
            "host": totals[HOST], "guest": totals[GUEST], "both": totals[BOTH]}
