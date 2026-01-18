# -*- coding: utf-8 -*-
"""Feature Store - centralized feature computation with FAIL-CLOSED warmup validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import hashlib

import numpy as np
import pandas as pd


TfRole = Literal["trigger", "anchor", "context"]


class FeatureStoreError(RuntimeError):
    pass


class FeatureDefinitionError(FeatureStoreError):
    pass


class FeatureWarmupError(FeatureStoreError):
    pass


def _stable_sig(params: Dict[str, Any]) -> str:
    """Stable signature for conflict detection."""
    items = sorted((str(k), repr(v)) for k, v in (params or {}).items())
    blob = "|".join([f"{k}={v}" for k, v in items]).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _ensure_dt_index(df_or_s: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    idx = df_or_s.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise FeatureDefinitionError(
            f"Expected DatetimeIndex, got {type(idx).__name__}. "
            "Convert your dataframe index to timestamps before attaching."
        )
    if idx.tz is not None:
        # Keep it simple and deterministic across platforms.
        df_or_s = df_or_s.copy()
        df_or_s.index = df_or_s.index.tz_convert(None)
    return df_or_s


@dataclass(frozen=True)
class FeatureSpec:
    """Defines a single feature and its causal/warmup contract."""

    name: str
    role: TfRole
    compute: Callable[["FeatureContext"], Union[pd.Series, pd.DataFrame]]

    deps: Tuple[str, ...] = ()
    params: Dict[str, Any] = field(default_factory=dict)

    # If set, the feature is considered invalid for the first warmup_bars of its ROLE TF.
    warmup_duration: Optional[str] = None

    # Human-readable description of causal/alignment contract.
    causal_contract: str = "same_tf"

    def signature(self) -> str:
        return _stable_sig(self.params)


@dataclass
class FeatureResult:
    value: Union[pd.Series, pd.DataFrame]
    valid: pd.Series  # bool mask aligned to value.index


@dataclass
class FeatureContext:
    """Runtime context passed to feature compute() functions."""

    config: Any
    dfs: Dict[TfRole, pd.DataFrame]
    tfs: Dict[TfRole, str]
    store: "FeatureStore"

    def df(self, role: TfRole) -> pd.DataFrame:
        return self.dfs[role]

    def tf(self, role: TfRole) -> str:
        return self.tfs[role]

    def get(self, name: str) -> FeatureResult:
        return self.store.get(name)


class FeatureStore:
    """A small registry + cache for feature computation.

    Goals:
      - Remove duplicated computations.
      - Enforce *one* definition per feature name (fail-closed on ambiguity).
      - Track warmup validity and allow callers to reject on invalid.

    This is intentionally lightweight: no DAG framework, just enough guardrails.
    """

    def __init__(self, config: Any, *, duration_to_bars: Optional[Callable[[str, str], int]] = None):
        self.config = config
        self._duration_to_bars = duration_to_bars

        self._specs: Dict[str, FeatureSpec] = {}
        self._spec_sigs: Dict[str, str] = {}
        self._cache: Dict[str, FeatureResult] = {}

        self._dfs: Dict[TfRole, pd.DataFrame] = {}
        self._tfs: Dict[TfRole, str] = {}

    # -------------------- attach --------------------

    def attach(self, role: TfRole, df: pd.DataFrame, timeframe: str) -> None:
        """Attach DataFrame for a role. Clears cache for affected features."""
        df = _ensure_dt_index(df)

        # Clear cache for all features that use this role (conservative approach)
        # This prevents stale data when re-attaching different data
        if role in self._dfs:
            # Role already attached - clear all cached features for this role
            to_clear = [
                name for name, spec in self._specs.items()
                if spec.role == role
            ]
            for name in to_clear:
                self._cache.pop(name, None)
            # Also clear features that depend on cleared features (transitive)
            self._clear_dependent_cache(to_clear)

        self._dfs[role] = df
        self._tfs[role] = timeframe

    def _clear_dependent_cache(self, cleared_names: list) -> None:
        """Clear cache for features that depend on already-cleared features."""
        cleared_set = set(cleared_names)
        changed = True
        while changed:
            changed = False
            for name, spec in self._specs.items():
                if name not in cleared_set and name in self._cache:
                    if any(dep in cleared_set for dep in spec.deps):
                        self._cache.pop(name, None)
                        cleared_set.add(name)
                        changed = True

    # -------------------- register --------------------

    def register(self, spec: FeatureSpec) -> None:
        if spec.name in self._specs:
            old_sig = self._spec_sigs[spec.name]
            new_sig = spec.signature()
            if old_sig != new_sig:
                raise FeatureDefinitionError(
                    f"Feature '{spec.name}' defined twice with different params/signature "
                    f"(old={old_sig}, new={new_sig}). Use different names (e.g. ema_20 vs ema_50)."
                )
            # Same signature: treat as idempotent.
            return

        if spec.role not in ("trigger", "anchor", "context"):
            raise FeatureDefinitionError(f"Unknown role: {spec.role}")

        self._specs[spec.name] = spec
        self._spec_sigs[spec.name] = spec.signature()

    # -------------------- compute/get --------------------

    def get(self, name: str) -> FeatureResult:
        if name in self._cache:
            return self._cache[name]

        if name not in self._specs:
            raise FeatureDefinitionError(f"Unknown feature '{name}'. Did you forget to register it?")

        spec = self._specs[name]

        # Compute dependencies first (DAG).
        for dep in spec.deps:
            self.get(dep)

        if spec.role not in self._dfs:
            raise FeatureDefinitionError(
                f"Role '{spec.role}' not attached for feature '{name}'. Call store.attach() first."
            )

        ctx = FeatureContext(config=self.config, dfs=self._dfs, tfs=self._tfs, store=self)
        out = spec.compute(ctx)
        if not isinstance(out, (pd.Series, pd.DataFrame)):
            raise FeatureDefinitionError(
                f"Feature '{name}' compute() must return pd.Series or pd.DataFrame, got {type(out)}"
            )
        out = _ensure_dt_index(out)

        # Valid mask: prefer explicit 'valid' column for DataFrames.
        if isinstance(out, pd.DataFrame) and "valid" in out.columns:
            valid = out["valid"].astype(bool)
        else:
            valid = pd.Series(True, index=out.index)

        # Apply warmup gate if configured.
        if spec.warmup_duration:
            if self._duration_to_bars is None:
                raise FeatureDefinitionError(
                    f"Feature '{name}' specifies warmup_duration='{spec.warmup_duration}', "
                    "but FeatureStore has no duration_to_bars() function wired."
                )
            bars = self._duration_to_bars(spec.warmup_duration, self._tfs[spec.role])
            if bars > 0:
                valid = valid.copy()
                valid.iloc[:bars] = False

        res = FeatureResult(value=out, valid=valid.reindex(out.index, fill_value=False))
        self._cache[name] = res
        return res

    def get_series(self, name: str) -> pd.Series:
        res = self.get(name)
        if not isinstance(res.value, pd.Series):
            raise FeatureDefinitionError(f"Feature '{name}' is not a Series.")
        return res.value

    def get_df(self, name: str) -> pd.DataFrame:
        res = self.get(name)
        if not isinstance(res.value, pd.DataFrame):
            raise FeatureDefinitionError(f"Feature '{name}' is not a DataFrame.")
        return res.value

    def is_valid_at(self, name: str, ts: pd.Timestamp) -> bool:
        res = self.get(name)
        if ts not in res.valid.index:
            return False
        return bool(res.valid.loc[ts])

    def value_at(self, name: str, ts: pd.Timestamp, *, fail_on_warmup: bool = True) -> Any:
        """Get feature value at specific timestamp with FAIL-CLOSED warmup check."""
        res = self.get(name)
        if ts not in res.value.index:
            raise FeatureWarmupError(f"Timestamp {ts} not found in feature '{name}' index.")
        if fail_on_warmup and not self.is_valid_at(name, ts):
            raise FeatureWarmupError(
                f"Feature '{name}' is not valid at {ts} (warmup period). "
                "Use fail_on_warmup=False to suppress this error."
            )
        if isinstance(res.value, pd.Series):
            return res.value.loc[ts]
        return res.value.loc[ts]  # Returns row for DataFrame

    # -------------------- aliases --------------------

    def register_default_specs(self) -> None:
        """Alias for register_default_prob_gate_bundle() for backward compatibility."""
        self.register_default_prob_gate_bundle()

    # -------------------- builtins for this repo --------------------

    def register_default_prob_gate_bundle(self) -> None:
        """Registers the minimal built-in set used by the current backtester.

        Names are explicit (avoid name collisions):
          - hilbert_score_1h
          - hilbert_score_aligned
          - prob_gate_bundle
          - ret_n
          - ema_short
          - drift_regime_aligned
        """

        # Lazy imports: keep this module importable in unit tests without TA-Lib, etc.
        from src.regime.upstream_scores import make_score_hilbert_1h
        from src.regime.upstream_scores import align_score_1h_to_15m
        from src.regime.prob_gate import ProbGateConfig, ProbabilityGate

        def _ret_n(ctx: FeatureContext) -> pd.Series:
            n = int(getattr(ctx.config, "prob_gate_short_ret_bars", 3))
            close = ctx.df("trigger")["close"].astype(float)
            return (close.pct_change(n) * 100.0).rename(f"ret_{n}")

        def _ema_short(ctx: FeatureContext) -> pd.Series:
            span = int(getattr(ctx.config, "prob_gate_short_ema_period", 20))
            close = ctx.df("trigger")["close"].astype(float)
            return close.ewm(span=span, adjust=False).mean().rename(f"ema_{span}")

        def _hilbert_score_1h(ctx: FeatureContext) -> pd.Series:
            df = ctx.df("context")
            # make_score_hilbert_1h expects 1H df.
            s = make_score_hilbert_1h(df)
            return s.rename("score_raw_1h")

        def _hilbert_score_aligned(ctx: FeatureContext) -> pd.Series:
            s1h = ctx.get("hilbert_score_1h").value
            df_tr = ctx.df("trigger")
            # 'open' semantics = shift by 1H to avoid using still-forming 1H candle.
            return align_score_1h_to_15m(s1h, df_tr, timestamp_semantics="open").rename("score_raw")

        def _drift_regime_aligned(ctx: FeatureContext) -> pd.Series:
            # Drift regime is only needed if enabled, but we always compute deterministically.
            df1h = ctx.df("context").copy()
            close = df1h["close"].astype(float)
            ema = close.ewm(span=200, adjust=False).mean()

            enter_pct = float(getattr(ctx.config, "prob_gate_drift_enter_pct", 0.012))
            exit_pct = float(getattr(ctx.config, "prob_gate_drift_exit_pct", 0.008))
            min_bars = int(getattr(ctx.config, "prob_gate_drift_min_bars", 3))

            use_slope = bool(getattr(ctx.config, "prob_gate_use_drift_slope", True))
            slope_bars = int(getattr(ctx.config, "prob_gate_drift_slope_bars", 24))
            slope_val = (ema.diff(slope_bars)).fillna(0.0)

            pct_diff = (close / ema) - 1.0

            regime = []
            current = "RANGE"
            cand = "RANGE"
            streak = 0

            for i in range(len(df1h)):
                v = float(pct_diff.iloc[i])
                s = float(slope_val.iloc[i])

                # Hysteresis hold
                if current == "UPTREND" and v >= exit_pct:
                    regime.append(current)
                    cand = current
                    streak = 0
                    continue
                if current == "DOWNTREND" and v <= -exit_pct:
                    regime.append(current)
                    cand = current
                    streak = 0
                    continue

                # Candidate from enter band
                if v > enter_pct:
                    new_cand = "UPTREND"
                    if use_slope and s <= 0:
                        new_cand = "RANGE"
                elif v < -enter_pct:
                    new_cand = "DOWNTREND"
                    if use_slope and s >= 0:
                        new_cand = "RANGE"
                else:
                    new_cand = "RANGE"

                if new_cand == cand:
                    streak += 1
                else:
                    cand = new_cand
                    streak = 1

                if streak >= min_bars:
                    current = cand

                regime.append(current)

            s_reg = pd.Series(regime, index=df1h.index, name="drift_regime_1h")

            # Causal alignment: shift by 1 bar on 1H, then ffill to trigger TF.
            s_reg = s_reg.shift(1)
            aligned = s_reg.reindex(ctx.df("trigger").index, method="ffill")
            return aligned.rename("drift_regime")

        def _prob_gate_bundle(ctx: FeatureContext) -> pd.DataFrame:
            df_tr = ctx.df("trigger")

            score = ctx.get("hilbert_score_aligned").value

            # Durations instead of magic numbers; scale with trigger tf.
            n_atr_dur = getattr(ctx.config, "prob_gate_atr_duration", "1d")
            vol_dur = getattr(ctx.config, "prob_gate_vol_duration", "2d")
            if self._duration_to_bars is None:
                raise FeatureDefinitionError("FeatureStore missing duration_to_bars() for prob_gate windows")
            n_atr = int(self._duration_to_bars(str(n_atr_dur), ctx.tf("trigger")))
            vol_window = int(self._duration_to_bars(str(vol_dur), ctx.tf("trigger")))

            gate_cfg = ProbGateConfig(
                temp_mode=getattr(ctx.config, "prob_gate_temp_mode", "vol"),
                p_shrink=float(getattr(ctx.config, "prob_gate_p_shrink", 0.6)),
                thr_long=float(getattr(ctx.config, "prob_gate_thr_long", 0.55)),
                thr_short=float(getattr(ctx.config, "prob_gate_thr_short", 0.65)),
                # Windows
                n_atr=n_atr,
                vol_window=vol_window,
            )
            gate = ProbabilityGate(gate_cfg)

            res = gate.compute(
                score.values,
                df_tr["close"].astype(float).values,
                df_tr["high"].astype(float).values,
                df_tr["low"].astype(float).values,
            ).copy()
            res.index = df_tr.index

            # Attach PR4.3 timing helpers.
            ret_n = ctx.get("ret_n").value
            ema_short = ctx.get("ema_short").value
            res["ret_n"] = ret_n.values
            res["ema_short"] = ema_short.values
            res["close_trigger"] = df_tr["close"].astype(float).values

            # Attach drift regime (aligned).
            res["drift_regime"] = ctx.get("drift_regime_aligned").value.values

            # Keep backward-compatible alias if callers expect 'close_5m'.
            res["close_5m"] = res["close_trigger"]

            return res

        # Extract config params for signature tracking (conflict detection)
        ret_n_bars = int(getattr(self.config, "prob_gate_short_ret_bars", 3))
        ema_span = int(getattr(self.config, "prob_gate_short_ema_period", 20))
        drift_enter_pct = float(getattr(self.config, "prob_gate_drift_enter_pct", 0.012))
        drift_exit_pct = float(getattr(self.config, "prob_gate_drift_exit_pct", 0.008))
        drift_min_bars = int(getattr(self.config, "prob_gate_drift_min_bars", 3))
        drift_use_slope = bool(getattr(self.config, "prob_gate_use_drift_slope", True))
        drift_slope_bars = int(getattr(self.config, "prob_gate_drift_slope_bars", 24))
        gate_temp_mode = str(getattr(self.config, "prob_gate_temp_mode", "vol"))
        gate_p_shrink = float(getattr(self.config, "prob_gate_p_shrink", 0.6))
        gate_thr_long = float(getattr(self.config, "prob_gate_thr_long", 0.55))
        gate_thr_short = float(getattr(self.config, "prob_gate_thr_short", 0.65))
        gate_atr_dur = str(getattr(self.config, "prob_gate_atr_duration", "1d"))
        gate_vol_dur = str(getattr(self.config, "prob_gate_vol_duration", "2d"))

        # Register specs with explicit params for conflict detection.
        self.register(FeatureSpec(
            name="ret_n",
            role="trigger",
            compute=_ret_n,
            params={"n": ret_n_bars},
            warmup_duration=getattr(self.config, "warmup_duration", None),
            causal_contract="same_tf: close.pct_change(n)",
        ))

        self.register(FeatureSpec(
            name="ema_short",
            role="trigger",
            compute=_ema_short,
            params={"span": ema_span},
            warmup_duration=getattr(self.config, "warmup_duration", None),
            causal_contract="same_tf: ewm(span)",
        ))

        self.register(FeatureSpec(
            name="hilbert_score_1h",
            role="context",
            compute=_hilbert_score_1h,
            params={},  # Uses upstream defaults
            warmup_duration="6h",
            causal_contract="context_tf hilbert score (closed bars)",
        ))

        self.register(FeatureSpec(
            name="hilbert_score_aligned",
            role="trigger",
            compute=_hilbert_score_aligned,
            params={"timestamp_semantics": "open"},
            deps=("hilbert_score_1h",),
            warmup_duration="6h",
            causal_contract="context->trigger ffill + shift(1H) (open semantics)",
        ))

        self.register(FeatureSpec(
            name="drift_regime_aligned",
            role="trigger",
            compute=_drift_regime_aligned,
            params={
                "enter_pct": drift_enter_pct,
                "exit_pct": drift_exit_pct,
                "min_bars": drift_min_bars,
                "use_slope": drift_use_slope,
                "slope_bars": drift_slope_bars,
            },
            warmup_duration="24h",
            causal_contract="context EMA200 regime w/ hysteresis; shift(1H) + ffill",
        ))

        self.register(FeatureSpec(
            name="prob_gate_bundle",
            role="trigger",
            compute=_prob_gate_bundle,
            params={
                "temp_mode": gate_temp_mode,
                "p_shrink": gate_p_shrink,
                "thr_long": gate_thr_long,
                "thr_short": gate_thr_short,
                "atr_duration": gate_atr_dur,
                "vol_duration": gate_vol_dur,
            },
            deps=("hilbert_score_aligned", "ret_n", "ema_short", "drift_regime_aligned"),
            warmup_duration=None,  # ProbabilityGate returns its own 'valid'
            causal_contract="ProbabilityGate over aligned hilbert score + OHLC",
        ))
