# -*- coding: utf-8 -*-
"""Tests for FeatureStore - FAIL-CLOSED warmup validation."""

import pytest
import pandas as pd
import numpy as np

from src.features.feature_store import (
    FeatureStore,
    FeatureSpec,
    FeatureDefinitionError,
    FeatureWarmupError,
)


def _make_df(index, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + rng.standard_normal(len(index)).cumsum()
    high = close + rng.random(len(index))
    low = close - rng.random(len(index))
    return pd.DataFrame({"close": close, "high": high, "low": low}, index=index)


def _duration_to_bars(duration: str, tf: str) -> int:
    """Simple duration parser for tests."""
    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    tf_min = tf_minutes.get(tf, 5)

    if duration.endswith("h"):
        return int(duration[:-1]) * 60 // tf_min
    elif duration.endswith("d"):
        return int(duration[:-1]) * 1440 // tf_min
    elif duration.endswith("m"):
        return int(duration[:-1]) // tf_min
    return 0


class DummyConfig:
    trigger_tf = "5m"
    anchor_tf = "15m"
    context_tf = "1h"
    warmup_duration = "1h"  # 12 bars @ 5m

    # prob gate params
    prob_gate_temp_mode = "vol"
    prob_gate_p_shrink = 0.6
    prob_gate_thr_long = 0.55
    prob_gate_thr_short = 0.65
    prob_gate_atr_duration = "1d"
    prob_gate_vol_duration = "2d"

    prob_gate_short_ret_bars = 3
    prob_gate_short_ema_period = 20

    prob_gate_use_drift_thr = True
    prob_gate_drift_ema_period = 200
    prob_gate_drift_range_pct = 0.01
    prob_gate_drift_enter_pct = 0.012
    prob_gate_drift_exit_pct = 0.008
    prob_gate_drift_min_bars = 3
    prob_gate_use_drift_slope = True
    prob_gate_drift_slope_bars = 24


class TestFeatureStoreBasic:
    """Basic FeatureStore functionality tests."""

    def test_register_and_cache(self):
        """Test that features are computed once and cached."""
        cfg = DummyConfig()
        idx = pd.date_range("2021-01-01", periods=100, freq="5min")
        df = _make_df(idx, seed=1)

        store = FeatureStore(cfg, duration_to_bars=_duration_to_bars)
        store.attach("trigger", df, timeframe="5m")

        # Register a simple feature
        store.register(FeatureSpec(
            name="test_ema",
            role="trigger",
            compute=lambda ctx: ctx.df("trigger")["close"].ewm(span=10).mean(),
        ))

        # First compute
        res1 = store.get("test_ema")
        # Second call should return cached
        res2 = store.get("test_ema")

        assert res1 is res2  # Same object (cached)
        assert len(res1.value) == 100

    def test_signature_conflict_detection(self):
        """Test that same name + different params raises error."""
        cfg = DummyConfig()
        idx = pd.date_range("2021-01-01", periods=50, freq="5min")
        df = _make_df(idx, seed=2)

        store = FeatureStore(cfg)
        store.attach("trigger", df, timeframe="5m")

        # Register first version
        store.register(FeatureSpec(
            name="ema_test",
            role="trigger",
            params={"span": 20},
            compute=lambda ctx: ctx.df("trigger")["close"].ewm(span=20).mean(),
        ))

        # Same params (idempotent) - should succeed
        store.register(FeatureSpec(
            name="ema_test",
            role="trigger",
            params={"span": 20},
            compute=lambda ctx: ctx.df("trigger")["close"].ewm(span=20).mean(),
        ))

        # Different params - should fail
        with pytest.raises(FeatureDefinitionError, match="different params"):
            store.register(FeatureSpec(
                name="ema_test",
                role="trigger",
                params={"span": 50},  # Different!
                compute=lambda ctx: ctx.df("trigger")["close"].ewm(span=50).mean(),
            ))

    def test_unknown_feature_raises(self):
        """Test that accessing unregistered feature raises error."""
        cfg = DummyConfig()
        store = FeatureStore(cfg)

        with pytest.raises(FeatureDefinitionError, match="Unknown feature"):
            store.get("nonexistent_feature")


class TestFeatureStoreWarmup:
    """FAIL-CLOSED warmup validation tests."""

    def test_warmup_marks_invalid(self):
        """Test that warmup period is marked as invalid."""
        cfg = DummyConfig()
        idx = pd.date_range("2021-01-01", periods=100, freq="5min")
        df = _make_df(idx, seed=3)

        store = FeatureStore(cfg, duration_to_bars=_duration_to_bars)
        store.attach("trigger", df, timeframe="5m")

        store.register(FeatureSpec(
            name="warmup_test",
            role="trigger",
            compute=lambda ctx: ctx.df("trigger")["close"].rolling(10).mean(),
            warmup_duration="1h",  # 12 bars @ 5m
        ))

        res = store.get("warmup_test")

        # First 12 bars should be invalid
        assert res.valid.iloc[0] == False
        assert res.valid.iloc[11] == False
        # Bar 12+ should be valid
        assert res.valid.iloc[12] == True
        assert res.valid.iloc[-1] == True

    def test_value_at_fail_closed(self):
        """Test that value_at raises on warmup period by default."""
        cfg = DummyConfig()
        idx = pd.date_range("2021-01-01", periods=50, freq="5min")
        df = _make_df(idx, seed=4)

        store = FeatureStore(cfg, duration_to_bars=_duration_to_bars)
        store.attach("trigger", df, timeframe="5m")

        store.register(FeatureSpec(
            name="warmup_fail_test",
            role="trigger",
            compute=lambda ctx: ctx.df("trigger")["close"].rolling(5).mean(),
            warmup_duration="1h",
        ))

        # Access during warmup should fail
        early_ts = idx[5]
        with pytest.raises(FeatureWarmupError, match="not valid"):
            store.value_at("warmup_fail_test", early_ts)

        # Access after warmup should succeed
        late_ts = idx[20]
        val = store.value_at("warmup_fail_test", late_ts)
        assert not np.isnan(val)

    def test_value_at_bypass_warmup(self):
        """Test that fail_on_warmup=False bypasses check."""
        cfg = DummyConfig()
        idx = pd.date_range("2021-01-01", periods=50, freq="5min")
        df = _make_df(idx, seed=5)

        store = FeatureStore(cfg, duration_to_bars=_duration_to_bars)
        store.attach("trigger", df, timeframe="5m")

        store.register(FeatureSpec(
            name="warmup_bypass_test",
            role="trigger",
            compute=lambda ctx: ctx.df("trigger")["close"],
            warmup_duration="1h",
        ))

        # Should not raise even during warmup
        early_ts = idx[5]
        val = store.value_at("warmup_bypass_test", early_ts, fail_on_warmup=False)
        assert val is not None


class TestFeatureStoreDependencies:
    """Test feature dependency resolution."""

    def test_dependency_chain(self):
        """Test that dependencies are computed first."""
        cfg = DummyConfig()
        idx = pd.date_range("2021-01-01", periods=50, freq="5min")
        df = _make_df(idx, seed=6)

        store = FeatureStore(cfg, duration_to_bars=_duration_to_bars)
        store.attach("trigger", df, timeframe="5m")

        compute_order = []

        def _base(ctx):
            compute_order.append("base")
            return ctx.df("trigger")["close"]

        def _derived(ctx):
            compute_order.append("derived")
            base = ctx.get("base_feature").value
            return base * 2

        store.register(FeatureSpec(name="base_feature", role="trigger", compute=_base))
        store.register(FeatureSpec(
            name="derived_feature",
            role="trigger",
            compute=_derived,
            deps=("base_feature",),
        ))

        # Access derived first
        store.get("derived_feature")

        # Base should be computed before derived
        assert compute_order == ["base", "derived"]


class TestFeatureStoreRoles:
    """Test multi-timeframe role handling."""

    def test_multi_role_attach(self):
        """Test attaching multiple TF roles."""
        cfg = DummyConfig()
        idx_5m = pd.date_range("2021-01-01", periods=100, freq="5min")
        idx_1h = pd.date_range("2021-01-01", periods=10, freq="1h")

        df_5m = _make_df(idx_5m, seed=7)
        df_1h = _make_df(idx_1h, seed=8)

        store = FeatureStore(cfg)
        store.attach("trigger", df_5m, timeframe="5m")
        store.attach("context", df_1h, timeframe="1h")

        # Feature on trigger TF
        store.register(FeatureSpec(
            name="trigger_close",
            role="trigger",
            compute=lambda ctx: ctx.df("trigger")["close"],
        ))

        # Feature on context TF
        store.register(FeatureSpec(
            name="context_close",
            role="context",
            compute=lambda ctx: ctx.df("context")["close"],
        ))

        res_trig = store.get("trigger_close")
        res_ctx = store.get("context_close")

        assert len(res_trig.value) == 100
        assert len(res_ctx.value) == 10

    def test_missing_role_raises(self):
        """Test that accessing feature for unattached role raises."""
        cfg = DummyConfig()
        store = FeatureStore(cfg)
        # Only attach trigger, not context
        idx = pd.date_range("2021-01-01", periods=50, freq="5min")
        store.attach("trigger", _make_df(idx), timeframe="5m")

        store.register(FeatureSpec(
            name="needs_context",
            role="context",  # Not attached!
            compute=lambda ctx: ctx.df("context")["close"],
        ))

        with pytest.raises(FeatureDefinitionError, match="not attached"):
            store.get("needs_context")


class TestFeatureStoreGetHelpers:
    """Test get_series and get_df helpers."""

    def test_get_series(self):
        """Test get_series returns Series."""
        cfg = DummyConfig()
        idx = pd.date_range("2021-01-01", periods=50, freq="5min")
        df = _make_df(idx, seed=9)

        store = FeatureStore(cfg)
        store.attach("trigger", df, timeframe="5m")

        store.register(FeatureSpec(
            name="series_feature",
            role="trigger",
            compute=lambda ctx: ctx.df("trigger")["close"],
        ))

        s = store.get_series("series_feature")
        assert isinstance(s, pd.Series)

    def test_get_df(self):
        """Test get_df returns DataFrame."""
        cfg = DummyConfig()
        idx = pd.date_range("2021-01-01", periods=50, freq="5min")
        df = _make_df(idx, seed=10)

        store = FeatureStore(cfg)
        store.attach("trigger", df, timeframe="5m")

        store.register(FeatureSpec(
            name="df_feature",
            role="trigger",
            compute=lambda ctx: ctx.df("trigger")[["close", "high"]],
        ))

        result = store.get_df("df_feature")
        assert isinstance(result, pd.DataFrame)
        assert "close" in result.columns
        assert "high" in result.columns

    def test_get_series_on_df_raises(self):
        """Test get_series on DataFrame raises."""
        cfg = DummyConfig()
        idx = pd.date_range("2021-01-01", periods=50, freq="5min")
        df = _make_df(idx, seed=11)

        store = FeatureStore(cfg)
        store.attach("trigger", df, timeframe="5m")

        store.register(FeatureSpec(
            name="df_not_series",
            role="trigger",
            compute=lambda ctx: ctx.df("trigger")[["close", "high"]],
        ))

        with pytest.raises(FeatureDefinitionError, match="not a Series"):
            store.get_series("df_not_series")


class TestFeatureStoreStrictExceptions:
    """Verify EXACT exception types are raised (not just any exception)."""

    def test_conflict_raises_definition_error_not_generic(self):
        """Ensure conflict raises FeatureDefinitionError, not TypeError/KeyError."""
        cfg = DummyConfig()
        store = FeatureStore(cfg)
        idx = pd.date_range("2021-01-01", periods=50, freq="5min")
        store.attach("trigger", _make_df(idx), timeframe="5m")

        store.register(FeatureSpec(
            name="conflict_test",
            role="trigger",
            params={"span": 10},
            compute=lambda ctx: ctx.df("trigger")["close"],
        ))

        # This should raise FeatureDefinitionError specifically
        with pytest.raises(FeatureDefinitionError) as exc_info:
            store.register(FeatureSpec(
                name="conflict_test",
                role="trigger",
                params={"span": 99},  # Different params
                compute=lambda ctx: ctx.df("trigger")["close"],
            ))

        # Verify it's exactly FeatureDefinitionError, not a subclass of something else
        assert type(exc_info.value).__name__ == "FeatureDefinitionError"
        assert "different params" in str(exc_info.value)

    def test_warmup_raises_warmup_error_not_generic(self):
        """Ensure warmup violation raises FeatureWarmupError, not KeyError."""
        cfg = DummyConfig()
        store = FeatureStore(cfg, duration_to_bars=_duration_to_bars)
        idx = pd.date_range("2021-01-01", periods=50, freq="5min")
        store.attach("trigger", _make_df(idx), timeframe="5m")

        store.register(FeatureSpec(
            name="warmup_strict_test",
            role="trigger",
            compute=lambda ctx: ctx.df("trigger")["close"],
            warmup_duration="1h",
        ))

        with pytest.raises(FeatureWarmupError) as exc_info:
            store.value_at("warmup_strict_test", idx[5])

        assert type(exc_info.value).__name__ == "FeatureWarmupError"
        assert "not valid" in str(exc_info.value)

    def test_missing_timestamp_raises_warmup_error(self):
        """Ensure missing timestamp raises FeatureWarmupError with clear message."""
        cfg = DummyConfig()
        store = FeatureStore(cfg)
        idx = pd.date_range("2021-01-01", periods=50, freq="5min")
        store.attach("trigger", _make_df(idx), timeframe="5m")

        store.register(FeatureSpec(
            name="ts_test",
            role="trigger",
            compute=lambda ctx: ctx.df("trigger")["close"],
        ))

        # Timestamp not in index
        bad_ts = pd.Timestamp("2099-01-01")
        with pytest.raises(FeatureWarmupError, match="not found"):
            store.value_at("ts_test", bad_ts)

    def test_unknown_role_raises_definition_error(self):
        """Ensure invalid role raises FeatureDefinitionError."""
        cfg = DummyConfig()
        store = FeatureStore(cfg)

        with pytest.raises(FeatureDefinitionError, match="Unknown role"):
            store.register(FeatureSpec(
                name="bad_role_test",
                role="invalid_role",  # Not trigger/anchor/context
                compute=lambda ctx: ctx.df("trigger")["close"],
            ))


class TestFeatureStoreCacheContract:
    """Test cache behavior and attach contract."""

    def test_attach_clears_cache(self):
        """Verify that re-attach clears the cache to prevent stale data."""
        cfg = DummyConfig()
        store = FeatureStore(cfg)

        idx1 = pd.date_range("2021-01-01", periods=50, freq="5min")
        df1 = _make_df(idx1, seed=100)
        df1["close"] = 100.0  # All 100

        store.attach("trigger", df1, timeframe="5m")
        store.register(FeatureSpec(
            name="cache_test",
            role="trigger",
            # Return Series, not scalar (FeatureStore requires Series/DataFrame)
            compute=lambda ctx: ctx.df("trigger")["close"],
        ))

        # First compute
        res1 = store.get("cache_test")
        assert res1.value.iloc[0] == 100.0

        # Re-attach with different data
        idx2 = pd.date_range("2021-01-01", periods=50, freq="5min")
        df2 = _make_df(idx2, seed=200)
        df2["close"] = 200.0  # All 200

        store.attach("trigger", df2, timeframe="5m")

        # Should recompute with new data (not return cached 100.0)
        res2 = store.get("cache_test")
        assert res2.value.iloc[0] == 200.0, "Cache was not cleared on re-attach!"

    def test_cache_is_shared_across_gets(self):
        """Verify same result object is returned from cache."""
        cfg = DummyConfig()
        store = FeatureStore(cfg)
        idx = pd.date_range("2021-01-01", periods=50, freq="5min")
        store.attach("trigger", _make_df(idx), timeframe="5m")

        call_count = [0]

        def _counting_compute(ctx):
            call_count[0] += 1
            return ctx.df("trigger")["close"]

        store.register(FeatureSpec(
            name="count_test",
            role="trigger",
            compute=_counting_compute,
        ))

        store.get("count_test")
        store.get("count_test")
        store.get("count_test")

        assert call_count[0] == 1, "Compute called multiple times despite caching!"
