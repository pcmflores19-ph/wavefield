import scenes
from scenes import BOTH, GUEST, HOST, _enforce_maximum


def test_both_dominates_duration_on_a_long_stretch():
    # A long monologue should read as mostly the merged view (~70%) with
    # brief returns to the talking camera (~30%) - not the other way round.
    shot = [(HOST, 0.0, 2000.0)]
    out = _enforce_maximum(shot, max_shot_seconds=25.0, min_shot_seconds=2.0)

    total = sum(end - start for _, start, end in out)
    both_total = sum(end - start for camera, start, end in out if camera == BOTH)
    both_fraction = both_total / total
    assert 0.6 <= both_fraction <= 0.8
    assert GUEST not in {camera for camera, _, _ in out}


def test_no_shot_exceeds_max_even_after_merging():
    for total in (60.0, 500.0, 2000.0):
        shot = [(HOST, 0.0, total)]
        out = _enforce_maximum(shot, max_shot_seconds=25.0, min_shot_seconds=2.0)
        for camera, start, end in out:
            assert end - start <= 25.0 + 1e-6, (
                f"a {end - start:.2f}s shot exceeds max_shot_seconds=25 "
                f"(stretch total={total}s)")


def test_varied_lengths_within_bounds():
    shot = [(HOST, 0.0, 500.0)]
    out = _enforce_maximum(shot, max_shot_seconds=10.0, min_shot_seconds=2.0)

    solo_lengths = [end - start for camera, start, end in out if camera == HOST]
    both_lengths = [end - start for camera, start, end in out if camera == BOTH]
    assert all(length >= 2.0 - 1e-6 for length in solo_lengths)
    assert len({round(length, 2) for length in both_lengths}) > 1


def test_deterministic_same_input_same_output():
    shot = [(HOST, 0.0, 200.0)]
    first = _enforce_maximum(list(shot), max_shot_seconds=10.0, min_shot_seconds=2.0)
    second = _enforce_maximum(list(shot), max_shot_seconds=10.0, min_shot_seconds=2.0)
    assert first == second

    active_by_lane = [[(0.0, 300.0)], []]
    timeline_a = scenes.scene_timeline(active_by_lane, 300.0,
                                        min_shot_seconds=2.0, max_shot_seconds=10.0)
    timeline_b = scenes.scene_timeline(active_by_lane, 300.0,
                                        min_shot_seconds=2.0, max_shot_seconds=10.0)
    assert timeline_a == timeline_b


def test_no_regression_on_short_and_both_shots():
    short_shot = [(HOST, 0.0, 5.0)]
    assert _enforce_maximum(short_shot, max_shot_seconds=10.0, min_shot_seconds=2.0) == short_shot

    long_both = [(BOTH, 0.0, 100.0)]
    assert _enforce_maximum(long_both, max_shot_seconds=10.0, min_shot_seconds=2.0) == long_both


def test_apply_to_keep_ranges_intersects_one_keep_range_with_several_scenes():
    # A single keep range spanning three back-to-back scenes should come
    # back as three pieces, each clipped to the keep range's own bounds.
    base_scenes = [(HOST, 0.0, 10.0), (GUEST, 10.0, 20.0), (BOTH, 20.0, 30.0)]
    pieces = scenes.apply_to_keep_ranges(base_scenes, [(5.0, 25.0)])

    assert pieces == [(HOST, 5.0, 10.0), (GUEST, 10.0, 20.0), (BOTH, 20.0, 25.0)]


def test_apply_to_keep_ranges_drops_scenes_outside_every_keep_range():
    base_scenes = [(HOST, 0.0, 10.0), (GUEST, 10.0, 20.0)]
    # Nothing in [10, 20) survives the cut, so GUEST must not appear at all.
    pieces = scenes.apply_to_keep_ranges(base_scenes, [(0.0, 5.0), (25.0, 30.0)])

    assert pieces == [(HOST, 0.0, 5.0)]


def test_apply_to_keep_ranges_handles_multiple_keep_ranges_in_one_scene():
    base_scenes = [(HOST, 0.0, 100.0)]
    pieces = scenes.apply_to_keep_ranges(base_scenes, [(10.0, 20.0), (50.0, 60.0)])

    assert pieces == [(HOST, 10.0, 20.0), (HOST, 50.0, 60.0)]

    unbounded = [(HOST, 0.0, 100.0)]
    assert _enforce_maximum(unbounded, max_shot_seconds=0, min_shot_seconds=2.0) == unbounded
    assert _enforce_maximum(unbounded, max_shot_seconds=None, min_shot_seconds=2.0) == unbounded


def test_no_sub_minimum_or_degenerate_tail():
    for total in (37.3, 41.0, 12.001, 199.9):
        shot = [(HOST, 0.0, total)]
        out = _enforce_maximum(shot, max_shot_seconds=10.0, min_shot_seconds=2.0)
        for camera, start, end in out:
            length = end - start
            assert length > 0
            if camera != BOTH:
                assert length >= 2.0 - 1e-6
