import numpy as np

from sync import HOP_SECONDS, SyncResult, TrackActivity, find_offset


def _irregular_intervals(total_seconds, seed=0):
    """
    A non-periodic speech pattern (irregular gaps) - a periodic one would be
    ambiguous by construction, since every multiple of the period scores
    identically.
    """
    rng = np.random.RandomState(seed)
    intervals = []
    t = 0.0
    speaking = True
    while t < total_seconds:
        length = rng.uniform(1.5, 6.0) if speaking else rng.uniform(0.5, 4.0)
        end = min(total_seconds, t + length)
        if speaking:
            intervals.append((t, end))
        t = end
        speaking = not speaking
    return intervals


def _shift_intervals(intervals, offset):
    return [(max(0.0, s + offset), e + offset) for s, e in intervals if e + offset > 0]


def _bursty_intervals(total_seconds, seed):
    """
    Real conversational speech: frequent short utterances (0.3-2s) with
    short gaps (0.2-1.5s) - much denser than _irregular_intervals' spread-out
    pattern. Regression coverage for a real bug: the coarse search pass used
    to block-MAX-downsample the activity signal, which for a pattern this
    dense saturates nearly every coarse bin to "speech happened here" and
    destroys the timing structure needed to find the right neighborhood -
    confirmed by direct reproduction to land tens of seconds from the true
    offset, with the fine pass's narrow window around that wrong anchor
    never able to recover. Fixed by using block-mean for that downsampling
    instead (see _block_mean in sync.py).
    """
    rng = np.random.RandomState(seed)
    intervals = []
    t = 0.0
    speaking = True
    while t < total_seconds:
        length = rng.uniform(0.3, 2.0) if speaking else rng.uniform(0.2, 1.5)
        end = min(total_seconds, t + length)
        if speaking:
            intervals.append((t, end))
        t = end
        speaking = not speaking
    return intervals


def test_recovers_offset_for_realistic_bursty_speech():
    duration = 600.0
    failures = []
    for seed in range(20):
        true_offset = np.random.RandomState(seed + 1000).uniform(3, 90)
        host = _bursty_intervals(duration, seed)
        guest = _shift_intervals(host, true_offset)
        reference = TrackActivity(host, duration, None)
        target = TrackActivity(guest, duration + true_offset, None)

        result = find_offset(reference, target, max_offset_seconds=120.0)
        error = abs(result.offset_seconds - true_offset)
        if result.ambiguous or error > 0.5:
            failures.append((seed, true_offset, result.offset_seconds, result.ambiguous))
    assert not failures, f"failed on {len(failures)}/20 bursty cases: {failures}"


def test_recovers_a_known_positive_offset_via_timing():
    duration = 300.0
    ref_intervals = _irregular_intervals(duration, seed=1)
    target_intervals = _shift_intervals(ref_intervals, 12.3)
    reference = TrackActivity(ref_intervals, duration, None)
    target = TrackActivity(target_intervals, duration + 12.3, None)

    result = find_offset(reference, target, max_offset_seconds=60.0)
    assert not result.ambiguous
    assert result.method == "timing"
    assert abs(result.offset_seconds - 12.3) < HOP_SECONDS * 2


def test_recovers_a_known_negative_offset_via_timing():
    duration = 300.0
    ref_intervals = _irregular_intervals(duration, seed=2)
    target_intervals = _shift_intervals(ref_intervals, -8.7)
    reference = TrackActivity(ref_intervals, duration, None)
    target = TrackActivity(target_intervals, duration, None)

    result = find_offset(reference, target, max_offset_seconds=60.0)
    assert not result.ambiguous
    assert result.method == "timing"
    assert abs(result.offset_seconds - (-8.7)) < HOP_SECONDS * 2


def test_recovers_offset_larger_than_the_fine_window():
    # Bigger than FINE_WINDOW_SECONDS (2.0) alone could correct for -
    # exercises the coarse pass actually finding the right neighborhood.
    duration = 400.0
    ref_intervals = _irregular_intervals(duration, seed=3)
    target_intervals = _shift_intervals(ref_intervals, 45.6)
    reference = TrackActivity(ref_intervals, duration, None)
    target = TrackActivity(target_intervals, duration + 45.6, None)

    result = find_offset(reference, target, max_offset_seconds=120.0)
    assert not result.ambiguous
    assert abs(result.offset_seconds - 45.6) < HOP_SECONDS * 2


def _synthetic_levels(total_seconds, peak_times, hop_seconds=0.010, width=0.3):
    n = int(total_seconds / hop_seconds)
    t = np.arange(n) * hop_seconds
    levels = np.full(n, -60.0)
    for peak in peak_times:
        bump = -60.0 + 54.0 * np.exp(-((t - peak) ** 2) / (2 * width ** 2))
        levels = np.maximum(levels, bump)
    return levels


def test_recovers_a_known_offset_via_content_when_timing_is_unrelated():
    duration = 200.0
    peak_times = [10.0, 35.0, 60.0, 61.5, 90.0, 140.0, 141.2, 180.0]
    ref_levels = _synthetic_levels(duration, peak_times)
    offset = 6.4
    target_levels = _synthetic_levels(duration, [p + offset for p in peak_times])

    # Intervals deliberately uncorrelated with the level peaks and with each
    # other, so a confident result can only come from the content signal.
    reference = TrackActivity(_irregular_intervals(duration, seed=11), duration, ref_levels)
    target = TrackActivity(_irregular_intervals(duration, seed=97), duration, target_levels)

    result = find_offset(reference, target, max_offset_seconds=60.0)
    assert not result.ambiguous
    assert result.method == "content"
    assert abs(result.offset_seconds - offset) < HOP_SECONDS * 2


def test_uncorrelated_tracks_of_similar_length_are_ambiguous():
    # Regression: a tiny overlap window near the edge of the search range
    # can score deceptively high by pure chance under normalized
    # cross-correlation (dividing by a near-zero sample count inflates a
    # coincidence). Two genuinely unrelated tracks of similar length must
    # not produce a confident match just because their true offset would
    # sit near the edge of max_offset_seconds.
    duration = 120.0
    reference = TrackActivity(_irregular_intervals(duration, seed=41), duration, None)
    target = TrackActivity(_irregular_intervals(duration, seed=42), duration, None)

    result = find_offset(reference, target, max_offset_seconds=600.0)
    assert result.ambiguous
    assert result.offset_seconds == 0.0


def test_ambiguous_when_no_real_overlap():
    reference = TrackActivity([(0.0, 5.0), (10.0, 15.0)], 20.0, None)
    target = TrackActivity([(1000.0, 1005.0)], 1010.0, None)

    result = find_offset(reference, target, max_offset_seconds=60.0)
    assert result.ambiguous
    assert result.offset_seconds == 0.0


def test_ambiguous_on_near_silent_target():
    duration = 120.0
    reference = TrackActivity(_irregular_intervals(duration, seed=5), duration, None)
    target = TrackActivity([(59.0, 59.2)], duration, None)

    result = find_offset(reference, target, max_offset_seconds=60.0)
    assert result.ambiguous


def test_falls_back_to_timing_only_when_levels_missing():
    duration = 200.0
    ref_intervals = _irregular_intervals(duration, seed=7)
    target_intervals = _shift_intervals(ref_intervals, 5.0)
    reference = TrackActivity(ref_intervals, duration, None)
    target = TrackActivity(target_intervals, duration + 5.0, np.array([]))

    result = find_offset(reference, target, max_offset_seconds=30.0)
    assert not result.ambiguous
    assert result.method == "timing"
    assert abs(result.offset_seconds - 5.0) < HOP_SECONDS * 2


def test_deterministic_same_input_same_output():
    duration = 150.0
    ref_intervals = _irregular_intervals(duration, seed=9)
    target_intervals = _shift_intervals(ref_intervals, 3.3)
    reference = TrackActivity(ref_intervals, duration, None)
    target = TrackActivity(target_intervals, duration, None)

    first = find_offset(reference, target, max_offset_seconds=30.0)
    second = find_offset(reference, target, max_offset_seconds=30.0)
    assert first == second
