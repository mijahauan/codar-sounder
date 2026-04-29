"""IQ source for the codar-sounder daemon.

Two backends are supported:

  * ``RadiodIQSource`` — the production path.  Subscribes to a radiod
    multicast IQ channel via ka9q-python's ``RadiodStream`` /
    ``ManagedStream`` (the same pattern psk-recorder, wspr-recorder,
    hf-timestd, hfdl-recorder use).  ka9q-python is imported lazily so
    the rest of the package remains usable on hosts that don't have it
    installed.

  * ``SyntheticIQSource`` — a deterministic synthetic chirp generator
    used by the end-to-end smoke test and by daemon dry-runs on hosts
    that have no live radiod.  Pretends to be a radiod channel emitting
    one configurable target at a fixed group range, plus optional
    additive complex noise.

Both backends present the same interface: an iterable that yields
contiguous ``numpy.complex64`` chunks of ``cpi_n_samples`` length each.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Iterator, Optional

import numpy as np

log = logging.getLogger(__name__)

# Speed of light, km/s — keep in sync with core/invert.py and core/dechirp.py.
C_KM_PER_S = 299_792.458


# ---------------------------------------------------------------------------
# Synthetic source — used for tests and dry-runs on hosts without radiod.
# ---------------------------------------------------------------------------

class SyntheticIQSource:
    """Generate a synthetic CODAR-like FMCW IQ stream with one target.

    The generator emits exactly the kind of signal the dechirp engine
    expects: a continuously-transmitted FMCW chirp with one delayed
    echo.  Group range, amplitude and noise floor are operator-set.

    Real-time pacing: the iterator sleeps so that one CPI takes
    approximately ``cpi_seconds`` of wall time — useful for daemon
    dry-runs where you want to see the JSONL records accumulate at a
    realistic cadence.  Set ``realtime=False`` for tests that want the
    chunks as fast as numpy can generate them.
    """

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        sweep_rate_hz_per_s: float,
        sweep_repetition_hz: float,
        cpi_seconds: float,
        target_group_range_km: float,
        target_amplitude: float = 1.0,
        noise_db: float = -40.0,
        realtime: bool = False,
        range_wobble_km: float = 30.0,
        wobble_period_s: float = 600.0,
        seed: int = 0,
    ):
        self.sample_rate_hz = float(sample_rate_hz)
        self.sweep_rate_hz_per_s = float(sweep_rate_hz_per_s)
        self.sweep_repetition_hz = float(sweep_repetition_hz)
        self.cpi_seconds = float(cpi_seconds)
        self.target_group_range_km = float(target_group_range_km)
        self.target_amplitude = float(target_amplitude)
        self.noise_amplitude = (
            10 ** (noise_db / 20.0) if noise_db > -200 else 0.0
        )
        # Real ionospheric F-region peaks wander 10–50 km on TID and
        # diurnal timescales.  A perfectly stationary synthetic target
        # gets removed by the daemon's rolling-median clutter mask
        # (correctly!) and disappears, which is *not* the
        # representative smoke-test scenario.  Adding a slow sinusoidal
        # wobble keeps the target visible to the F-region trace
        # extractor while letting the clutter mask still suppress
        # truly stationary energy (direct path, 0-Hz DC).
        self.range_wobble_km = float(range_wobble_km)
        self.wobble_period_s = max(float(wobble_period_s), 1.0)
        self.realtime = realtime
        self._rng = np.random.default_rng(seed=seed)
        self._t_offset_s = 0.0
        self._stopped = threading.Event()

    @property
    def cpi_n_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.cpi_seconds))

    def _generate_one_cpi(self) -> np.ndarray:
        n = self.cpi_n_samples
        sweep_period = 1.0 / self.sweep_repetition_hz
        # Sinusoidal wobble centred on target_group_range_km — see __init__.
        wobble = 0.0
        if self.range_wobble_km > 0 and self.wobble_period_s > 0:
            import math
            wobble = self.range_wobble_km * math.sin(
                2 * math.pi * self._t_offset_s / self.wobble_period_s
            )
        delay_s = (self.target_group_range_km + wobble) / C_KM_PER_S

        t = self._t_offset_s + np.arange(n) / self.sample_rate_hz
        # Sweep wraps every sweep_period — modulo gives the in-sweep time.
        t_in_sweep_target = (t - delay_s) % sweep_period
        valid = t >= delay_s
        phase = (
            2.0 * np.pi * 0.5 * self.sweep_rate_hz_per_s * t_in_sweep_target ** 2
        )
        rx = self.target_amplitude * np.exp(1j * phase).astype(np.complex64)
        rx[~valid] = 0
        if self.noise_amplitude > 0:
            rx += self.noise_amplitude * (
                self._rng.standard_normal(n)
                + 1j * self._rng.standard_normal(n)
            ).astype(np.complex64)
        self._t_offset_s += n / self.sample_rate_hz
        return rx

    def __iter__(self) -> Iterator[np.ndarray]:
        while not self._stopped.is_set():
            t0 = time.monotonic()
            yield self._generate_one_cpi()
            if self.realtime:
                elapsed = time.monotonic() - t0
                remaining = self.cpi_seconds - elapsed
                if remaining > 0:
                    self._stopped.wait(remaining)

    def stop(self) -> None:
        self._stopped.set()


# ---------------------------------------------------------------------------
# radiod source — production path via ka9q-python.
# ---------------------------------------------------------------------------

class RadiodIQSource:
    """Subscribe to a radiod IQ channel via ka9q-python.

    Lazy-imports ``ka9q``-package modules at construction time so that
    a host without ka9q-python installed can still load the rest of
    codar-sounder (e.g. ``inventory --json``, ``validate --json``).

    Raises:
        ModuleNotFoundError: if ka9q-python is not importable.  The
            daemon catches this and falls back to ``SyntheticIQSource``
            with operator-visible warnings.
    """

    def __init__(
        self,
        *,
        radiod_status_dns: str,
        channel_name: str,
        sample_rate_hz: float,
        cpi_seconds: float,
    ):
        self.radiod_status_dns = radiod_status_dns
        self.channel_name = channel_name
        self.sample_rate_hz = float(sample_rate_hz)
        self.cpi_seconds = float(cpi_seconds)
        self._stream = None
        self._stopped = threading.Event()
        self._import_ka9q()  # raises ModuleNotFoundError if missing

    @property
    def cpi_n_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.cpi_seconds))

    def _import_ka9q(self) -> None:
        # Lazy import — ka9q-python is large and may not be present.
        # We deliberately import only what we need so import cost stays low.
        global _ka9q_RadiodStream, _ka9q_RadiodControl
        from ka9q.stream import RadiodStream                 # type: ignore[import-not-found]
        from ka9q.control import RadiodControl              # type: ignore[import-not-found]
        _ka9q_RadiodStream = RadiodStream
        _ka9q_RadiodControl = RadiodControl

    def __iter__(self) -> Iterator[np.ndarray]:
        # The full RadiodStream wiring (control-channel resolution,
        # SSRC negotiation, packet resequencing, gap-fill) is
        # nontrivial and lives in ka9q-python.  v0.2 wires it up
        # straightforwardly: subscribe, read CPI-sized chunks, yield
        # them as complex64.  Detailed integration with ManagedStream
        # for auto-recovery is deferred to v0.3.
        log.info(
            "RadiodIQSource: subscribing to %s channel=%s sample_rate=%d Hz",
            self.radiod_status_dns, self.channel_name, int(self.sample_rate_hz),
        )
        stream = _ka9q_RadiodStream(                        # type: ignore[name-defined]
            status=self.radiod_status_dns,
            ssrc=self.channel_name,
        )
        self._stream = stream

        n_samples = self.cpi_n_samples
        buf = np.empty(n_samples, dtype=np.complex64)
        filled = 0
        for samples in stream:
            chunk = np.asarray(samples, dtype=np.complex64)
            while chunk.size > 0:
                if self._stopped.is_set():
                    return
                take = min(chunk.size, n_samples - filled)
                buf[filled:filled + take] = chunk[:take]
                filled += take
                chunk = chunk[take:]
                if filled == n_samples:
                    yield buf.copy()
                    filled = 0

    def stop(self) -> None:
        self._stopped.set()
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception as exc:
                log.warning("RadiodStream close failed: %s", exc)


# ---------------------------------------------------------------------------
# Factory — try radiod first, fall back to synthetic with a clear warning.
# ---------------------------------------------------------------------------

def make_iq_source(
    *,
    radiod_status_dns: str,
    channel_name: str,
    sample_rate_hz: float,
    cpi_seconds: float,
    sweep_rate_hz_per_s: float,
    sweep_repetition_hz: float,
    fallback_target_group_range_km: float = 500.0,
    force_synthetic: bool = False,
):
    """Construct an IQ source, falling back to synthetic if radiod isn't available.

    The fallback case is *deliberately noisy in the log* so an operator
    who expects real data sees the warning rather than wondering why
    the JSONL records all show the same target.
    """
    if not force_synthetic:
        try:
            return RadiodIQSource(
                radiod_status_dns=radiod_status_dns,
                channel_name=channel_name,
                sample_rate_hz=sample_rate_hz,
                cpi_seconds=cpi_seconds,
            )
        except ModuleNotFoundError as exc:
            log.warning(
                "ka9q-python not importable (%s) — falling back to "
                "SyntheticIQSource.  Install ka9q-python and restart for "
                "live data.", exc,
            )

    log.warning(
        "Using SyntheticIQSource: target group range = %.1f km, "
        "no real ionospheric data will be produced.",
        fallback_target_group_range_km,
    )
    return SyntheticIQSource(
        sample_rate_hz=sample_rate_hz,
        sweep_rate_hz_per_s=sweep_rate_hz_per_s,
        sweep_repetition_hz=sweep_repetition_hz,
        cpi_seconds=cpi_seconds,
        target_group_range_km=fallback_target_group_range_km,
        realtime=True,
    )
