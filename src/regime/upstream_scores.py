"""src/regime/upstream_scores.py

PR2: Upstream score builders for ProbabilityGate.

IMPORTANT
---------
These functions produce *directional* scores (positive=bullish, negative=bearish).
Hilbert phase is a *state*; we must explicitly map it to direction.

Also includes safe 1H -> 15m alignment helpers that avoid future leakage.
Default behavior assumes 1H timestamps represent the *bar open* time and will
shift scores by 1 bar so the 1H score is only available after the 1H bar closes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dt_index(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Return a copy sorted by datetime index."""
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        return out.sort_index()

    if ts_col not in df.columns:
        raise KeyError(f"Expected '{ts_col}' column or DatetimeIndex")

    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col])
    return out.sort_values(ts_col).set_index(ts_col)


def _hilbert_analytic_signal(x: np.ndarray) -> np.ndarray:
    """Analytic signal via FFT (SciPy-free Hilbert transform).

    Returns complex array z = x + j*H{x}.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    X = np.fft.fft(x)

    h = np.zeros(n)
    if n % 2 == 0:
        # even
        h[0] = 1
        h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        # odd
        h[0] = 1
        h[1 : (n + 1) // 2] = 2

    return np.fft.ifft(X * h)


def _ema_series(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


# ---------------------------------------------------------------------------
# Hilbert score
# ---------------------------------------------------------------------------

@dataclass
class HilbertScoreConfig:
    close_col: str = "close"
    ts_col: str = "timestamp"
    detrend_span: int = 48          # 1H 기준: 약 2일 (48h)
    smooth_span: int = 6            # score smoothing
    lead_phase: float = np.pi / 4   # 45도 선행
    amp_weight: bool = True
    amp_weight_ref_span: int = 96   # amplitude reference EMA span


def make_score_hilbert_1h(df_1h: pd.DataFrame, cfg: Optional[HilbertScoreConfig] = None) -> pd.Series:
    """Hilbert phase -> *directional* score.

    Mapping (explicit, directional):
      - phase = angle(analytic_signal)
      - sine = sin(phase)
      - lead = sin(phase + lead_phase)
      - score = lead - sine   (rising => +, falling => -)

    Notes:
      - Hilbert phase is NOT direction by itself.
      - We detrend first to reduce phase drift.
      - Optional amplitude weighting reduces impact when cycle strength is weak.
    """
    cfg = cfg or HilbertScoreConfig()
    df = _to_dt_index(df_1h, ts_col=cfg.ts_col)

    if cfg.close_col not in df.columns:
        raise KeyError(f"df_1h missing '{cfg.close_col}'")

    close = df[cfg.close_col].astype(float)

    # detrend (simple): remove slow EMA baseline
    base = _ema_series(close, span=cfg.detrend_span)
    x = (close - base).to_numpy(dtype=float)

    z = _hilbert_analytic_signal(x)
    phase = np.unwrap(np.angle(z))

    sine = np.sin(phase)
    lead = np.sin(phase + float(cfg.lead_phase))
    score = lead - sine

    if cfg.amp_weight:
        amp = np.abs(z)
        # robust reference: EMA of amp (avoids median over whole sample)
        amp_ref = _ema_series(pd.Series(amp, index=df.index), span=cfg.amp_weight_ref_span).to_numpy(dtype=float)
        w = np.tanh(amp / (amp_ref + 1e-12))
        score = score * w

    s = pd.Series(score, index=df.index, name="score_hilbert_1h")
    s = _ema_series(s, span=cfg.smooth_span)
    return s


# ---------------------------------------------------------------------------
# HMM 2-state score
# ---------------------------------------------------------------------------

@dataclass
class HMMScoreConfig:
    ts_col: str = "timestamp"
    p_markup_col: str = "p_markup"
    p_markdown_col: str = "p_markdown"
    use_logit: bool = False
    smooth_span: int = 3


def make_score_hmm_1h(df_1h: pd.DataFrame, cfg: Optional[HMMScoreConfig] = None) -> pd.Series:
    """2-state posterior -> directional score.

    score = p_markup - p_markdown (in [-1, +1])
    or logit(p_markup / p_markdown) if use_logit.
    """
    cfg = cfg or HMMScoreConfig()
    df = _to_dt_index(df_1h, ts_col=cfg.ts_col)

    for col in (cfg.p_markup_col, cfg.p_markdown_col):
        if col not in df.columns:
            raise KeyError(f"df_1h missing '{col}'")

    p_up = df[cfg.p_markup_col].astype(float)
    p_dn = df[cfg.p_markdown_col].astype(float)

    if cfg.use_logit:
        eps = 1e-6
        raw = np.log((p_up + eps) / (p_dn + eps))
    else:
        raw = p_up - p_dn

    s = pd.Series(raw.to_numpy(dtype=float), index=df.index, name="score_hmm_1h")
    s = _ema_series(s, span=cfg.smooth_span)
    return s


# ---------------------------------------------------------------------------
# Safe alignment: 1H score -> 15m bars (no lookahead)
# ---------------------------------------------------------------------------

def align_score_1h_to_15m(
    score_1h: pd.Series,
    df_15m: pd.DataFrame,
    *,
    df_15m_ts_col: str = "timestamp",
    timestamp_semantics: Literal["open", "close"] = "open",
) -> pd.Series:
    """Align 1H score to 15m bars using forward-fill, with optional safety shift.

    Parameters
    ----------
    score_1h:
        1H score indexed by datetime.
    df_15m:
        15m dataframe with datetime index or a timestamp column.
    timestamp_semantics:
        - 'open'  (default, safest): assumes 1H timestamps are bar OPEN times.
          We shift by 1 bar so the score becomes usable only after the 1H bar closes.
        - 'close': assumes 1H timestamps are bar CLOSE times; no shift.

    Returns
    -------
    score_15m aligned to df_15m index, forward-filled.
    """
    if not isinstance(score_1h.index, pd.DatetimeIndex):
        raise TypeError("score_1h must have a DatetimeIndex")

    s = score_1h.sort_index().copy()
    if timestamp_semantics == "open":
        # shift by 1 bar (1H) to avoid using the still-forming 1H bar
        s = s.shift(1)

    df15 = _to_dt_index(df_15m, ts_col=df_15m_ts_col)
    s15 = s.reindex(df15.index, method="ffill")
    return s15.rename("score_raw")
