"""
Physical-simulation tests for src.pulse / src.wavelet.

Rationale
---------
A single Wavelet evaluated through Pulse gives
    y(x, t) = c * cos(kc*x + phi_c + wc*t)   (+ analogous sine term)
which is a plane wave with phase velocity v = -w/k. Rather than eyeballing
plots, each test below checks a physical invariant that ANY correct
implementation of a traveling/standing wave must satisfy:

  1. The 1-D wave equation:      d2y/dt2 = v^2 * d2y/dx2
  2. Phase velocity:             x_peak(t) tracks x_peak(0) - (w/k)*t
  3. Spatial periodicity:        y(x + 2*pi/k, t) == y(x, t)
  4. Temporal periodicity:       y(x, t + 2*pi/w) == y(x, t)
  5. Amplitude bound:            |y| <= sum(|c_i| + |s_i|)   (triangle ineq.)
  6. t=0 consistency:            Pulse(ws=0) == StandingPulse with same coeffs
  7. Superposition / linearity   (documents a real bug, see note below)
  8. Random generator sanity     (coefficients/k in [0,1), single nonzero term)

Run with: pytest test_physics_validation.py -v
"""
import numpy as np
import pytest

from src.pulse import Pulse, StandingPulse
from src.wavelet import Wavelet

RTOL = 1e-6
ATOL = 1e-6


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def make_single_cosine_pulse(c=1.3, kc=0.7, phi_c=0.2, wc=0.9):
    """A pulse with exactly one traveling cosine component."""
    return Pulse(
        coefficients=np.array([[c, 0.0]]),
        wave_vectors=np.array([[kc, 0.0]]),
        phases=np.array([[phi_c, 0.0]]),
        ws=np.array([[wc, 0.0]]),
    )


def make_single_sine_pulse(s=0.8, ks=0.5, phi_s=0.4, ws_=0.6):
    """A pulse with exactly one traveling sine component."""
    return Pulse(
        coefficients=np.array([[0.0, s]]),
        wave_vectors=np.array([[0.0, ks]]),
        phases=np.array([[0.0, phi_s]]),
        ws=np.array([[0.0, ws_]]),
    )


def central_diff_2nd(f_vals, h):
    """Second derivative of a sampled 1-D array via central differences."""
    return (f_vals[2:] - 2 * f_vals[1:-1] + f_vals[:-2]) / h**2


# --------------------------------------------------------------------------
# 1. Wave equation: d2y/dt2 = v^2 * d2y/dx2, v = w/k
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pulse,k,w", [
    (make_single_cosine_pulse(kc=0.7, wc=0.9), 0.7, 0.9),
    (make_single_sine_pulse(ks=0.5, ws_=0.6), 0.5, 0.6),
])
def test_wave_equation_satisfied(pulse, k, w):
    # NOTE on step size: Wavelet.__call__ rounds every sample to 6 decimal
    # places (`round(..., self.truncate)`). A central second difference's
    # true signal scales as O(h^2), so h must be large enough that h^2
    # comfortably exceeds the ~5e-7 rounding noise floor -- otherwise you're
    # measuring quantization artifacts, not the wave equation. h=0.05 keeps
    # the numerical-differentiation truncation error negligible while
    # staying far above the rounding noise floor.
    v = w / k
    h = 0.05

    x0, t0 = 3.0, 2.0
    xs = x0 + np.array([-h, 0, h])
    ts = t0 + np.array([-h, 0, h])

    # d2y/dx2 at (x0, t0)
    y_x = np.array([pulse(np.array([x]), t0)[0] for x in xs])
    d2y_dx2 = central_diff_2nd(y_x, h)[0]

    # d2y/dt2 at (x0, t0)
    y_t = np.array([pulse(np.array([x0]), t)[0] for t in ts])
    d2y_dt2 = central_diff_2nd(y_t, h)[0]

    assert d2y_dt2 == pytest.approx(v**2 * d2y_dx2, rel=1e-2, abs=1e-3)


# --------------------------------------------------------------------------
# 2. Phase velocity: peak drifts at -w/k
# --------------------------------------------------------------------------

def test_phase_velocity_matches_dispersion():
    # NOTE: an infinite plane wave has infinitely many equal-height peaks,
    # so tracking "the" peak via a global argmax is ill-posed -- which peak
    # wins the argmax jumps around with the sampling grid as t changes
    # (that's what produced the wildly wrong slope before). Instead use the
    # chain rule directly: for y = f(k*x + w*t), (dy/dt)/(dy/dx) = w/k at
    # any point where dy/dx != 0. This is local, well-posed, and needs no
    # peak-finding at all.
    kc, wc = 0.6, -1.1
    pulse = make_single_cosine_pulse(c=1.0, kc=kc, phi_c=0.3, wc=wc)
    v_expected = -wc / kc

    x0, t0 = 4.0, 1.0
    h = 1e-3

    dy_dx = (pulse(np.array([x0 + h]), t0)[0] - pulse(np.array([x0 - h]), t0)[0]) / (2 * h)
    dy_dt = (pulse(np.array([x0]), t0 + h)[0] - pulse(np.array([x0]), t0 - h)[0]) / (2 * h)

    assert dy_dx != pytest.approx(0.0, abs=1e-4), "chosen point sits at a node; pick a different x0/t0"
    # Phase velocity is the speed of a constant-phase level curve y(x,t)=const.
    # Implicit differentiation of that constraint gives dx/dt = -(dy/dt)/(dy/dx)
    # -- note the minus sign, which is easy to drop (I did, first pass).
    v_measured = -dy_dt / dy_dx
    assert v_measured == pytest.approx(v_expected, rel=1e-2, abs=1e-2)


# --------------------------------------------------------------------------
# 3 & 4. Periodicity in x and in t
# --------------------------------------------------------------------------

def test_spatial_periodicity():
    kc, wc = 0.8, 0.3
    pulse = make_single_cosine_pulse(kc=kc, wc=wc)
    period_x = 2 * np.pi / kc

    x = np.linspace(-10, 10, 200)
    t = 1.7
    y1 = pulse(x, t)
    y2 = pulse(x + period_x, t)
    np.testing.assert_allclose(y1, y2, rtol=RTOL, atol=ATOL)


def test_temporal_periodicity():
    kc, wc = 0.8, 0.3
    pulse = make_single_cosine_pulse(kc=kc, wc=wc)
    period_t = 2 * np.pi / wc

    x = np.linspace(-10, 10, 50)
    t0 = 0.4
    y1 = pulse(x, t0)
    y2 = pulse(x, t0 + period_t)
    np.testing.assert_allclose(y1, y2, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------
# 5. Amplitude bound (triangle inequality on superposed wavelets)
# --------------------------------------------------------------------------

def test_amplitude_bound():
    coeffs = np.array([[1.2, 0.7], [0.3, 0.9]])
    kss = np.array([[0.5, 0.4], [0.2, 0.6]])
    phases = np.array([[0.0, 0.0], [0.0, 0.0]])
    standing = StandingPulse(coefficients=coeffs, wave_vectors=kss, phases=phases)

    bound = np.sum(np.abs(coeffs))
    x = np.linspace(-30, 30, 2000)
    y = standing(x)
    assert np.max(np.abs(y)) <= bound + ATOL


# --------------------------------------------------------------------------
# 6. t=0 consistency between Pulse and StandingPulse
# --------------------------------------------------------------------------

def test_pulse_matches_standing_pulse_at_t0():
    coeffs = np.array([[1.1, 0.4]])
    kss = np.array([[0.3, 0.7]])
    phases = np.array([[0.1, 0.2]])

    standing = StandingPulse(coefficients=coeffs, wave_vectors=kss, phases=phases)
    moving = Pulse(coefficients=coeffs, wave_vectors=kss, phases=phases,
                    ws=np.array([[0.0, 0.0]]))

    x = np.linspace(-20, 20, 500)
    np.testing.assert_allclose(standing(x), moving(x, t=0), rtol=RTOL, atol=ATOL)

    # And a genuinely moving pulse should still equal the standing pattern
    # exactly at t=0, for any nonzero ws.
    moving2 = Pulse(coefficients=coeffs, wave_vectors=kss, phases=phases,
                     ws=np.array([[1.3, -0.8]]))
    np.testing.assert_allclose(standing(x), moving2(x, t=0), rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------
# 7. Superposition / linearity
# --------------------------------------------------------------------------

def test_superposition_pointwise_evaluation():
    """
    Two independently-evaluated pulses summed pointwise must equal a single
    pulse built from the union of both wavelet sets, evaluated once. This is
    just linearity of the wave equation and should always hold if you build
    the combined Pulse directly via from_wavelets(..., ws=...).
    """
    p1 = make_single_cosine_pulse(c=1.0, kc=0.5, phi_c=0.0, wc=0.4)
    p2 = make_single_sine_pulse(s=0.6, ks=0.3, phi_s=0.1, ws_=-0.2)

    combined = Pulse.from_wavelets(
        [Wavelet(1.0, 0, 0.5, 0, 0.0, 0), Wavelet(0, 0.6, 0, 0.3, 0, 0.1)],
        ws=np.array([[0.4, 0.0], [0.0, -0.2]]),
    )

    x = np.linspace(-10, 10, 100)
    t = 0.9
    lhs = p1(x, t) + p2(x, t)
    rhs = combined(x, t)
    np.testing.assert_allclose(lhs, rhs, rtol=RTOL, atol=ATOL)


def test_pulse_plus_pulse_stays_time_dependent():
    p1 = make_single_cosine_pulse(c=1.0, kc=0.5, phi_c=0.0, wc=0.4)
    p2 = make_single_sine_pulse(s=0.6, ks=0.3, phi_s=0.1, ws_=-0.2)

    combined = p1 + p2
    assert isinstance(combined, Pulse), (
        "p1 + p2 lost its time-dependence: got a StandingPulse instead of a Pulse"
    )

    # and it should actually match pointwise summation at t != 0
    x = np.linspace(-10, 10, 100)
    t = 0.9
    np.testing.assert_allclose(combined(x, t), p1(x, t) + p2(x, t), rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------
# 8. Random generator sanity (bounds + single-term structure)
# --------------------------------------------------------------------------

def test_random_cosine_wavelet_is_physically_sane():
    for _ in range(50):
        w = Wavelet.generate_random_cosine_wavelet()
        c, s, kc, ks, phi_c, phi_s = w.sextuplet
        assert 0.0 <= c < 1.0
        assert 0.0 <= kc < 1.0
        assert 0.0 <= phi_c < 1.0
        assert s == 0.0 and ks == 0.0 and phi_s == 0.0


def test_random_sine_wavelet_is_physically_sane():
    for _ in range(50):
        w = Wavelet.generate_random_sine_wavelet()
        c, s, kc, ks, phi_c, phi_s = w.sextuplet
        assert 0.0 <= s < 1.0
        assert 0.0 <= ks < 1.0
        assert 0.0 <= phi_s < 1.0
        assert c == 0.0 and kc == 0.0 and phi_c == 0.0


def test_random_standing_pulse_bounded():
    for _ in range(20):
        pulse = StandingPulse.generate_random_standing_pulse(n_cos=2, n_sin=2)
        bound = sum(abs(w.sextuplet[0]) + abs(w.sextuplet[1]) for w in pulse.wavelets)
        x = np.linspace(-30, 30, 500)
        assert np.max(np.abs(pulse(x))) <= bound + ATOL


# --------------------------------------------------------------------------
# 9. Regression: generate_random_pulse must not silently drop wavelets
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_cos,n_sin", [(1, 1), (2, 1), (1, 3), (3, 3)])
def test_generate_random_pulse_keeps_all_wavelets(n_cos, n_sin):
    """
    Regression test: generate_random_pulse used to default `ws` to shape
    (1, 2) regardless of n_cos + n_sin. Because transform() passes ndarrays
    through unchanged (no shape validation), wavelets_at(t) would silently
    zip() down to just 1 wavelet -- 2 of 3, or worse, would vanish with no
    error. This checks the full requested wavelet count survives evaluation.
    """
    pulse = Pulse.generate_random_pulse(n_cos=n_cos, n_sin=n_sin)
    expected = n_cos + n_sin
    assert pulse.time_weights.shape[0] == expected
    assert len(pulse.wavelets_at(0.5)) == expected


def test_transform_does_not_validate_ndarray_shape():
    """
    Documents a real hazard in transform(): if `value` is already an
    ndarray, it is returned unchanged with NO shape check against the
    requested `shape`. A caller who passes a mis-shaped ndarray gets no
    error -- just silent zip()-truncation downstream (see the test above).
    This test pins down current (unsafe) behavior; if transform() is later
    hardened to validate/reshape, this test should be updated to expect
    a raised error instead.
    """
    from src.utils.types import transform
    mis_shaped = np.ones((1, 2))
    result = transform(mis_shaped, (3, 2))
    assert result.shape == (1, 2), (
        "transform() silently let a (1,2) array through where (3,2) was "
        "requested -- this is the root cause of the wavelet-dropping bug."
    )


# --------------------------------------------------------------------------
# 10. Group velocity of a two-wavelet beating pulse
# --------------------------------------------------------------------------

def test_group_velocity_of_beating_pulse():
    """
    Single-wavelet tests only confirm phase velocity of one plane wave.
    A pulse built from two close-frequency components beats: the fast
    carrier moves at phase velocity ~ -w_mean/k_mean, but the slow envelope
    moves at group velocity v_g = -dw/dk (dw = w1-w2, dk = k1-k2). This is
    the physical quantity that actually matters for wave-packet transport,
    and it can only be wrong if multi-wavelet superposition itself is wrong
    (single-wavelet tests can't catch that class of bug).

    Measured via Hilbert-transform envelope extraction + cross-correlation
    between two time snapshots, with sub-sample parabolic interpolation on
    the correlation peak. dt is kept small relative to the envelope's
    spatial period (2*pi/dk) to avoid aliasing onto the wrong beat cycle.
    """
    hilbert = pytest.importorskip("scipy.signal").hilbert

    c = 1.0
    k1, k2 = 1.0, 0.9
    w1, w2 = 1.35, 1.2
    dk, dw = k1 - k2, w1 - w2
    v_g_expected = -dw / dk

    pulse = Pulse(
        coefficients=np.array([[c, 0], [c, 0]]),
        wave_vectors=np.array([[k1, 0], [k2, 0]]),
        phases=np.array([[0, 0], [0, 0]]),
        ws=np.array([[w1, 0], [w2, 0]]),
    )

    dx = 0.01
    x = np.arange(-150, 150, dx)
    t0, dt = 0.0, 2.0  # envelope period ~2*pi/dk ~= 62.8; dt's shift must stay well under that

    env0 = np.abs(hilbert(pulse(x, t0)))
    env1 = np.abs(hilbert(pulse(x, t0 + dt)))

    # trim edges: Hilbert transform (FFT-based) has boundary artifacts
    trim = int(10.0 / dx)
    env0, env1 = env0[trim:-trim], env1[trim:-trim]

    corr = np.correlate(env1 - env1.mean(), env0 - env0.mean(), mode="full")
    peak = np.argmax(corr)
    y_m1, y_0, y_p1 = corr[peak - 1], corr[peak], corr[peak + 1]
    sub_sample = 0.5 * (y_m1 - y_p1) / (y_m1 - 2 * y_0 + y_p1)
    lag_samples = (peak - (len(env0) - 1)) + sub_sample

    v_measured = (lag_samples * dx) / dt
    assert v_measured == pytest.approx(v_g_expected, rel=0.03, abs=0.03)


# --------------------------------------------------------------------------
# 11. Mixed-type __add__ must raise, not silently corrupt or crash obscurely
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 11. StandingPulse + Pulse: a static pattern plus a moving wave is a moving
#     wave. This exercises the "StandingPulse is structurally a Pulse with
#     ws=0" design: both classes now carry time_weights, so combining them
#     is well-defined and physically meaningful (a background field plus a
#     traveling disturbance), not an error case.
# --------------------------------------------------------------------------

def test_standing_pulse_plus_pulse_is_a_moving_pulse():
    static = StandingPulse(coefficients=np.array([[1.0, 0]]), wave_vectors=np.array([[0.4, 0]]),
                            phases=np.array([[0.2, 0]]))
    moving = make_single_cosine_pulse(c=0.8, kc=0.6, phi_c=0.0, wc=0.9)

    combined = static + moving
    assert isinstance(combined, Pulse)

    x = np.linspace(-15, 15, 40)
    t = 1.7
    expected = static(x) + moving(x, t)
    np.testing.assert_allclose(combined(x, t), expected, rtol=RTOL, atol=ATOL)


def test_pulse_plus_standing_pulse_is_commutative():
    static = StandingPulse(coefficients=np.array([[1.0, 0]]), wave_vectors=np.array([[0.4, 0]]),
                            phases=np.array([[0.2, 0]]))
    moving = make_single_cosine_pulse(c=0.8, kc=0.6, phi_c=0.0, wc=0.9)

    r1 = static + moving
    r2 = moving + static
    assert isinstance(r2, Pulse)

    x = np.linspace(-15, 15, 40)
    t = 1.7
    np.testing.assert_allclose(r1(x, t), r2(x, t), rtol=RTOL, atol=ATOL)


def test_standing_pulse_plus_standing_pulse_stays_standing():
    """Two genuinely static operands should NOT get upgraded to a Pulse --
    only combinations involving actual motion (nonzero ws, or a Pulse
    operand) should return a Pulse."""
    s1 = StandingPulse(coefficients=np.array([[1.0, 0]]), wave_vectors=np.array([[0.4, 0]]),
                        phases=np.array([[0.2, 0]]))
    s2 = StandingPulse(coefficients=np.array([[0.3, 0]]), wave_vectors=np.array([[0.2, 0]]),
                        phases=np.array([[0.1, 0]]))
    combined = s1 + s2
    assert isinstance(combined, StandingPulse) and not isinstance(combined, Pulse)


# --------------------------------------------------------------------------
# 12. Exact destructive interference
# --------------------------------------------------------------------------

def test_destructive_interference_is_exactly_zero():
    """Two identical-(k,w) wavelets, pi out of phase, must cancel exactly."""
    k, w = 0.5, 0.3
    w1 = Pulse(coefficients=np.array([[1.0, 0]]), wave_vectors=np.array([[k, 0]]),
               phases=np.array([[0.0, 0]]), ws=np.array([[w, 0]]))
    w2 = Pulse(coefficients=np.array([[1.0, 0]]), wave_vectors=np.array([[k, 0]]),
               phases=np.array([[np.pi, 0]]), ws=np.array([[w, 0]]))
    combined = w1 + w2

    x = np.linspace(-20, 20, 50)
    for t in [0.0, 1.23, -3.4, 100.0]:
        y = combined(x, t)
        assert np.max(np.abs(y)) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# 13. Real standing wave from two counter-propagating pulses (fixed nodes)
# --------------------------------------------------------------------------

def test_counter_propagating_pulses_form_standing_wave_with_fixed_nodes():
    """
    Two equal-amplitude Pulses with the SAME k but OPPOSITE w are counter-
    propagating traveling waves. Their sum is 2*cos(kx)*cos(wt) -- a genuine
    standing wave with nodes fixed in x for all time. This is the physical
    phenomenon the class name "StandingPulse" evokes, but the StandingPulse
    class itself has no time axis at all -- this test constructs the real
    thing from two Pulses and checks the defining property: a node stays
    exactly zero for every t, not just at t=0.
    """
    k, w = 0.6, 0.8
    right = Pulse(coefficients=np.array([[1.0, 0]]), wave_vectors=np.array([[k, 0]]),
                  phases=np.array([[0.0, 0]]), ws=np.array([[w, 0]]))
    left = Pulse(coefficients=np.array([[1.0, 0]]), wave_vectors=np.array([[k, 0]]),
                 phases=np.array([[0.0, 0]]), ws=np.array([[-w, 0]]))
    standing = right + left

    # analytic node: 2*cos(k*x)*cos(w*t) == 0 for all t requires cos(k*x)==0
    node_x = np.array([(np.pi / 2) / k])
    for t in [0.0, 0.7, 1.3, 5.0, -2.2]:
        assert standing(node_x, t)[0] == pytest.approx(0.0, abs=1e-9)

    # and a non-node point should NOT be uniformly zero (sanity check that
    # the test isn't trivially passing because everything is zero)
    antinode_x = np.array([0.0])  # cos(k*0)=1, so amplitude here is 2*cos(w*t)
    values_at_antinode = [standing(antinode_x, t)[0] for t in [0.0, 0.3, 0.9, 1.5]]
    assert np.std(values_at_antinode) > 0.1


# --------------------------------------------------------------------------
# 14. Rigid-shape translation invariance of a single traveling wave
# --------------------------------------------------------------------------

def test_traveling_wave_is_a_rigidly_translating_shape():
    """A true traveling wave satisfies y(x, t+dt) == y(x - v*dt, t) exactly:
    the whole profile shifts by v*dt without changing shape."""
    kc, wc = 0.7, -1.1
    v = -wc / kc
    pulse = make_single_cosine_pulse(c=1.3, kc=kc, phi_c=0.4, wc=wc)

    x = np.linspace(-20, 20, 200)
    t1, dt = 2.0, 0.6
    y_t2 = pulse(x, t1 + dt)
    y_t1_shifted = pulse(x - v * dt, t1)
    np.testing.assert_allclose(y_t2, y_t1_shifted, rtol=RTOL, atol=1e-8)


# --------------------------------------------------------------------------
# 15. Spectral correctness: FFT of the pulse must show energy ONLY at the
#     wavenumbers that were put in, with the right amplitudes
# --------------------------------------------------------------------------

def test_spatial_spectrum_matches_input_wavenumbers_and_amplitudes():
    """
    This is the most direct test that the library does what it claims: build
    a two-wavelet pulse, FFT it in x, and check the spectrum has exactly two
    peaks, at exactly the requested k's, with exactly the requested
    amplitudes -- and nothing anywhere else (no leakage/cross-talk/aliasing
    from however the superposition is implemented internally).

    k1, k2 are chosen as exact integer multiples of the FFT's frequency
    resolution (2*pi/L) so the domain is an exact integer number of periods
    for both components -- this avoids spectral leakage that would otherwise
    smear energy across neighboring bins and make the amplitude check noisy.
    """
    c1, c2 = 1.3, 0.7
    L, N = 400.0, 8192
    df = 2 * np.pi / L
    k1, k2 = 51 * df, 134 * df

    pulse = StandingPulse(
        coefficients=np.array([[c1, 0], [c2, 0]]),
        wave_vectors=np.array([[k1, 0], [k2, 0]]),
        phases=np.array([[0.0, 0], [0.0, 0]]),
    )

    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = x[1] - x[0]
    y = pulse(x)

    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(N, d=dx) * 2 * np.pi
    amp = np.abs(Y) * 2 / N

    idx1 = np.argmin(np.abs(freqs - k1))
    idx2 = np.argmin(np.abs(freqs - k2))
    assert amp[idx1] == pytest.approx(c1, rel=1e-4)
    assert amp[idx2] == pytest.approx(c2, rel=1e-4)

    mask = np.ones_like(amp, dtype=bool)
    mask[idx1] = mask[idx2] = False
    assert amp[mask].max() < 1e-4, "energy leaked to a wavenumber that wasn't in the pulse"


# --------------------------------------------------------------------------
# 16. Parseval / energy: total spatial energy equals the sum of each
#     component's independent contribution (orthogonality across k)
# --------------------------------------------------------------------------

def test_energy_is_additive_across_orthogonal_wavenumbers():
    """
    For distinct wavenumbers integrated over an exact common period, cross
    terms in integral(y^2 dx) vanish by orthogonality -- so total energy
    must equal the sum of each component's independent energy
    (integral of (c*cos(kx))^2 dx = c^2/2 * L per component), not something
    that depends on how the components interact. This is a stronger
    statement than the amplitude-bound test: it checks the *exact* energy
    value, not just an upper bound.
    """
    c1, c2 = 1.3, 0.7
    L, N = 400.0, 8192
    df = 2 * np.pi / L
    k1, k2 = 51 * df, 134 * df

    pulse = StandingPulse(
        coefficients=np.array([[c1, 0], [c2, 0]]),
        wave_vectors=np.array([[k1, 0], [k2, 0]]),
        phases=np.array([[0.0, 0], [0.0, 0]]),
    )
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = x[1] - x[0]
    y = pulse(x)

    energy_measured = np.sum(y**2) * dx
    energy_expected = (c1**2 / 2 + c2**2 / 2) * L
    assert energy_measured == pytest.approx(energy_expected, rel=1e-6)


# --------------------------------------------------------------------------
# 17. k=0 edge case: should collapse to a spatially-uniform oscillator,
#     not silently produce NaN/Inf
# --------------------------------------------------------------------------

def test_zero_wavenumber_is_spatially_uniform_not_nan():
    """
    k=0 means infinite wavelength: the wave should be identical at every x
    (a pure temporal oscillation, no spatial variation) -- this is a valid
    physical limit, not a singularity, for the wave VALUE itself. (Only a
    derived quantity like phase velocity = w/k is undefined at k=0; the
    wave amplitude is perfectly well-defined.) This guards against a future
    change accidentally introducing a division by k somewhere in the
    evaluation path.
    """
    w = Wavelet(c=1.0, s=0, kc=0.0, ks=0, phi_c=0.5, phi_s=0)
    x = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    y = w(x)
    assert np.all(np.isfinite(y)), "k=0 produced non-finite values"
    assert np.allclose(y, y[0]), "k=0 wave should be identical at every x"
    assert y[0] == pytest.approx(np.cos(0.5), rel=RTOL)