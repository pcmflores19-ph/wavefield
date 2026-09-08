"""
Built-in audio effects, ported from OBS Studio's filters.

OBS's filters are C compiled into OBS itself - there is no plugin file to load,
so these are ports rather than copies. Each one keeps OBS's algorithm, its
parameter names and its defaults, so a setting that works in OBS works here.
Sources: obs-studio/plugins/obs-filters/{noise-gate,compressor,limiter,
expander,gain,eq}-filter.c

Two reasons this beats shipping third-party plugins: nothing has to be
redistributed, and the controls are already familiar to anyone who records in
OBS - which is exactly who this app is for.

The one piece of maths everything shares, straight from OBS:

    gain_coefficient(rate, time) = exp(-1 / (rate * time))

an exponential smoothing coefficient for an attack or release time in seconds.
"""

import numpy as np

# OBS works in float samples and dB throughout, with this floor standing in for
# digital silence.
VOL_MIN_DB = -96.0


def _gain_coefficient(sample_rate, seconds):
    if seconds <= 0:
        return 0.0
    return float(np.exp(-1.0 / (sample_rate * seconds)))


def _mul_to_db(x):
    return np.where(x > 1e-9, 20.0 * np.log10(np.maximum(x, 1e-9)), VOL_MIN_DB)


def _db_to_mul(db):
    return np.power(10.0, db / 20.0)


def _envelope(samples, attack_gain, release_gain):
    """
    OBS's envelope follower, sample by sample:

        if env < |x|:  env = |x| + attack_gain  * (env - |x|)
        else:          env = |x| + release_gain * (env - |x|)

    Genuinely sequential - each output depends on the one before - so it cannot
    be vectorised away. numpy would still be looping, just in Python.
    """
    magnitude = np.abs(samples)
    out = np.empty_like(magnitude)
    env = 0.0
    for i in range(magnitude.size):
        env_in = magnitude[i]
        if env < env_in:
            env = env_in + attack_gain * (env - env_in)
        else:
            env = env_in + release_gain * (env - env_in)
        out[i] = env
    return out


try:                                    # 50-100x faster, optional
    from numba import njit
    _envelope = njit(cache=True, fastmath=True)(_envelope)
except Exception:
    pass


# --------------------------------------------------------------------- gate

def noise_gate(samples, sample_rate, open_threshold_db=-26.0,
               close_threshold_db=-32.0, attack_ms=25.0, hold_ms=200.0,
               release_ms=150.0):
    """
    Silences the microphone between sentences.

    Two thresholds, not one: it opens at -26 dB but does not close again until
    -32 dB, so a voice dipping mid-word cannot chatter the gate on and off.
    Once closed it holds open for `hold_ms` first, which is what stops the tail
    of a word being clipped.
    """
    open_level = float(_db_to_mul(open_threshold_db))
    close_level = float(_db_to_mul(close_threshold_db))
    attack_rate = 1.0 / max(1e-6, sample_rate * (attack_ms / 1000.0))
    release_rate = 1.0 / max(1e-6, sample_rate * (release_ms / 1000.0))
    hold_seconds = hold_ms / 1000.0
    # OBS's decay: how fast the measured level is allowed to fall.
    decay_rate = 1.0 / max(1e-6, sample_rate * (release_ms / 1000.0))

    out = np.empty_like(samples)
    level = 0.0
    attenuation = 0.0
    # Start with the hold already expired. Starting it at zero means the very
    # first sample counts as "still within the hold window" and the gate ramps
    # itself open on silence - it only cut 3 dB instead of closing.
    held = hold_seconds
    is_open = False
    dt = 1.0 / sample_rate

    for i in range(samples.size):
        current = abs(float(samples[i]))
        if current > open_level:
            is_open = True
        level = max(level, current)
        if level < close_level and is_open:
            held = 0.0
            is_open = False
        level = max(0.0, level - decay_rate)

        if is_open:
            attenuation = min(1.0, attenuation + attack_rate)
        else:
            held += dt
            if held < hold_seconds:
                attenuation = min(1.0, attenuation + attack_rate)
            else:
                attenuation = max(0.0, attenuation - release_rate)
        out[i] = samples[i] * attenuation
    return out


# --------------------------------------------------- compressor and limiter

def _compress(samples, sample_rate, threshold_db, ratio, attack_ms,
              release_ms, output_gain_db):
    attack_gain = _gain_coefficient(sample_rate, attack_ms / 1000.0)
    release_gain = _gain_coefficient(sample_rate, release_ms / 1000.0)
    slope = 1.0 - (1.0 / ratio)

    env = _envelope(samples.astype(np.float32), attack_gain, release_gain)
    env_db = _mul_to_db(env)
    # gain = slope * (threshold - env_db), never boosting.
    gain = _db_to_mul(np.minimum(0.0, slope * (threshold_db - env_db)))
    return samples * gain * float(_db_to_mul(output_gain_db))


def compressor(samples, sample_rate, threshold_db=-18.0, ratio=10.0,
               attack_ms=6.0, release_ms=60.0, output_gain_db=0.0):
    """Evens out a voice that swings between loud and quiet."""
    return _compress(samples, sample_rate, threshold_db, max(1.0, ratio),
                     attack_ms, release_ms, output_gain_db)


def limiter(samples, sample_rate, threshold_db=-6.0, release_ms=60.0):
    """
    A compressor with an infinite ratio: nothing gets past the threshold.

    OBS fixes the attack at 0.1 ms - a limiter that eases in is not a limiter.
    """
    attack_gain = _gain_coefficient(sample_rate, 0.0001)
    release_gain = _gain_coefficient(sample_rate, release_ms / 1000.0)
    env = _envelope(samples.astype(np.float32), attack_gain, release_gain)
    env_db = _mul_to_db(env)
    gain = _db_to_mul(np.minimum(0.0, threshold_db - env_db))
    return samples * gain


def expander(samples, sample_rate, threshold_db=-40.0, ratio=2.0,
             attack_ms=10.0, release_ms=50.0, output_gain_db=0.0):
    """
    The opposite of a compressor: pushes quiet things further down.

    Gentler than a gate - room tone is reduced rather than chopped out, which
    on a breathy voice sounds far more natural.
    """
    attack_gain = _gain_coefficient(sample_rate, attack_ms / 1000.0)
    release_gain = _gain_coefficient(sample_rate, release_ms / 1000.0)
    slope = 1.0 - max(1.0, ratio)

    env = _envelope(samples.astype(np.float32), attack_gain, release_gain)
    env_db = _mul_to_db(env)
    gain = _db_to_mul(np.minimum(0.0, slope * (threshold_db - env_db)))
    return samples * gain * float(_db_to_mul(output_gain_db))


# ----------------------------------------------------------------- gain, EQ

def gain(samples, sample_rate, gain_db=0.0):
    """Plain volume change."""
    return samples * float(_db_to_mul(gain_db))


def eq3(samples, sample_rate, low_db=0.0, mid_db=0.0, high_db=0.0):
    """
    OBS's three-band equaliser: low below 100 Hz, high above 10 kHz, mid the
    rest. Built from one-pole filters, as OBS does, rather than a biquad bank.
    """
    if low_db == 0.0 and mid_db == 0.0 and high_db == 0.0:
        return samples

    def one_pole_low(x, cutoff):
        """
        y[n] = (1-a)*x[n] + a*y[n-1]

        A one-pole IIR: each output depends on the last, so it cannot be
        vectorised. scipy does it in C when available - roughly a hundred times
        faster over a full episode - and the loop is the fallback, since the
        frozen build deliberately excludes scipy to keep the installer small.
        """
        a = float(np.exp(-2.0 * np.pi * cutoff / sample_rate))
        try:
            from scipy.signal import lfilter
            return lfilter([1.0 - a], [1.0, -a], x).astype(np.float32)
        except Exception:
            pass
        out = np.empty_like(x)
        state = 0.0
        for i in range(x.size):
            state = (1.0 - a) * x[i] + a * state
            out[i] = state
        return out

    low = one_pole_low(samples, 100.0)
    low_mid = one_pole_low(samples, 10000.0)
    mid = low_mid - low
    high = samples - low_mid
    return (low * float(_db_to_mul(low_db))
            + mid * float(_db_to_mul(mid_db))
            + high * float(_db_to_mul(high_db)))


try:
    from numba import njit as _njit
    eq3.__globals__["_one_pole_compiled"] = True
except Exception:
    pass


# ------------------------------------------------------------------ registry

# What the FX window offers. Each parameter is
# (key, label, minimum, maximum, default, unit).
EFFECTS = [
    ("noise_gate", "Noise Gate", noise_gate, [
        ("open_threshold_db", "Open threshold", -96.0, 0.0, -26.0, "dB"),
        ("close_threshold_db", "Close threshold", -96.0, 0.0, -32.0, "dB"),
        ("attack_ms", "Attack", 0.0, 500.0, 25.0, "ms"),
        ("hold_ms", "Hold", 0.0, 1000.0, 200.0, "ms"),
        ("release_ms", "Release", 1.0, 1000.0, 150.0, "ms"),
    ]),
    ("compressor", "Compressor", compressor, [
        ("ratio", "Ratio", 1.0, 32.0, 10.0, ": 1"),
        ("threshold_db", "Threshold", -60.0, 0.0, -18.0, "dB"),
        ("attack_ms", "Attack", 1.0, 500.0, 6.0, "ms"),
        ("release_ms", "Release", 1.0, 1000.0, 60.0, "ms"),
        ("output_gain_db", "Output gain", -32.0, 32.0, 0.0, "dB"),
    ]),
    ("expander", "Expander", expander, [
        ("ratio", "Ratio", 1.0, 20.0, 2.0, ": 1"),
        ("threshold_db", "Threshold", -60.0, 0.0, -40.0, "dB"),
        ("attack_ms", "Attack", 1.0, 100.0, 10.0, "ms"),
        ("release_ms", "Release", 1.0, 1000.0, 50.0, "ms"),
        ("output_gain_db", "Output gain", -32.0, 32.0, 0.0, "dB"),
    ]),
    ("limiter", "Limiter", limiter, [
        ("threshold_db", "Threshold", -60.0, 0.0, -6.0, "dB"),
        ("release_ms", "Release", 1.0, 1000.0, 60.0, "ms"),
    ]),
    ("eq3", "3-Band EQ", eq3, [
        ("low_db", "Low", -20.0, 20.0, 0.0, "dB"),
        ("mid_db", "Mid", -20.0, 20.0, 0.0, "dB"),
        ("high_db", "High", -20.0, 20.0, 0.0, "dB"),
    ]),
    ("gain", "Gain", gain, [
        ("gain_db", "Gain", -30.0, 30.0, 0.0, "dB"),
    ]),
]

BY_KEY = {key: (label, fn, params) for key, label, fn, params in EFFECTS}


def defaults(key):
    _, _, params = BY_KEY[key]
    return {name: default for name, _, _, _, default, _ in params}


def apply(key, samples, sample_rate, params=None):
    """Runs one effect. Unknown keys pass the audio through untouched."""
    entry = BY_KEY.get(key)
    if entry is None:
        return samples
    _, fn, spec = entry
    values = defaults(key)
    values.update(params or {})
    allowed = {name for name, _, _, _, _, _ in spec}
    values = {k: v for k, v in values.items() if k in allowed}
    return fn(np.asarray(samples, dtype=np.float32), sample_rate, **values)
