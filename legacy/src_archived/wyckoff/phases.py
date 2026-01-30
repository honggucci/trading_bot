"""
Wyckoff Phase Detection & Bidirectional Accumulation Strategies
v2.0 - Physics/Math Enhanced (FFT, Hilbert, Maxwell-Boltzmann, Entropy)

=== 6개 Wyckoff 패턴 (Schematics) ===
1. Accumulation: 바닥권 매집 (롱 준비)
2. Re-Accumulation: 상승 중 매집 (추가 롱)
3. Distribution: 고점권 분산 (숏 준비)
4. Re-Distribution: 하락 중 분산 (추가 숏)
5. Markup: 상승 추세 (롱 홀드)
6. Markdown: 하락 추세 (숏 홀드)

Origin: WPCN wpcn/_03_common/_03_wyckoff/phases.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

try:
    from scipy import signal as scipy_signal
    from scipy.stats import entropy as scipy_entropy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .types import Theta
from .indicators import atr, rsi, zscore
from .box import box_engine_freeze


# ============================================================================
# Direction Ratios
# ============================================================================

DIRECTION_RATIOS_FUTURES = {
    'accumulation': {'long': 0.70, 'short': 0.30},
    'distribution': {'long': 0.30, 'short': 0.70},
    're_accumulation': {'long': 0.80, 'short': 0.20},
    're_distribution': {'long': 0.20, 'short': 0.80},
    'markup': {'long': 0.90, 'short': 0.10},
    'markdown': {'long': 0.10, 'short': 0.90},
    'range': {'long': 0.50, 'short': 0.50},
    'unknown': {'long': 0.50, 'short': 0.50}
}

DIRECTION_RATIOS = DIRECTION_RATIOS_FUTURES


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class WyckoffPhase:
    """Wyckoff 페이즈 상태"""
    phase: str  # 'A', 'B', 'C', 'D', 'E', 'unknown'
    sub_phase: str
    direction: str  # 'accumulation', 'distribution', 're_accumulation', 're_distribution', 'unknown'
    confidence: float
    start_idx: int
    box_low: float
    box_high: float


@dataclass
class AccumulationSignal:
    """축적 신호"""
    t_signal: Any
    side: str
    event: str
    entry_price: float
    stop_price: float
    tp_price: float
    position_pct: float
    phase: str
    meta: Optional[Dict[str, Any]] = None


# ============================================================================
# Physics-Enhanced Phase Detector (Simplified)
# ============================================================================

class PhysicsEnhancedPhaseDetector:
    """물리/수학 기반 Phase A 감지 엔진 (Simplified)"""

    def __init__(self, lookback: int = 100):
        self.lookback = lookback

    def detect_dominant_cycle(self, prices: np.ndarray) -> Tuple[int, float]:
        """FFT 기반 지배 사이클 감지"""
        prices = prices[~np.isnan(prices)]
        if len(prices) < 32 or not HAS_SCIPY:
            return 20, 0.0

        detrended = prices - np.linspace(prices[0], prices[-1], len(prices))
        fft_result = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(detrended))

        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_power = np.abs(fft_result[pos_mask]) ** 2

        if len(pos_power) == 0:
            return 20, 0.0

        max_idx = np.argmax(pos_power)
        dominant_freq = pos_freqs[max_idx]

        if dominant_freq > 0:
            cycle_length = int(1 / dominant_freq)
            cycle_length = max(5, min(cycle_length, len(prices) // 2))
        else:
            cycle_length = 20

        total_power = np.sum(pos_power)
        dominant_power = pos_power[max_idx] / total_power if total_power > 0 else 0

        return cycle_length, dominant_power

    def compute_hilbert_phase(self, prices: np.ndarray) -> Tuple[float, str]:
        """Hilbert Transform 기반 순간 위상 계산"""
        prices = prices[~np.isnan(prices)]
        if len(prices) < 10 or not HAS_SCIPY:
            return 0.0, 'unknown'

        detrended = prices - np.linspace(prices[0], prices[-1], len(prices))
        analytic = scipy_signal.hilbert(detrended)
        instantaneous_phase = np.angle(analytic)
        current_phase = instantaneous_phase[-1]
        phase_normalized = (current_phase + np.pi) % (2 * np.pi)

        if phase_normalized < np.pi / 4 or phase_normalized >= 7 * np.pi / 4:
            position = 'bottom'
        elif phase_normalized < 3 * np.pi / 4:
            position = 'rising'
        elif phase_normalized < 5 * np.pi / 4:
            position = 'top'
        else:
            position = 'falling'

        return current_phase, position

    def compute_mb_volatility_regime(self, returns: np.ndarray) -> Tuple[str, float]:
        """Maxwell-Boltzmann 분포 기반 변동성 레짐 분류"""
        returns = returns[~np.isnan(returns)]
        if len(returns) < 20:
            return 'medium', 0.5

        current_vol = np.nanstd(returns[-20:])
        historical_vol = np.abs(returns)
        mean_vol = np.nanmean(historical_vol)
        std_vol = np.nanstd(historical_vol)

        if std_vol > 0:
            z_score = (current_vol - mean_vol) / std_vol
            percentile = 0.5 * (1 + np.tanh(z_score * 0.5))
        else:
            percentile = 0.5

        if percentile < 0.25:
            regime = 'low'
        elif percentile < 0.50:
            regime = 'medium'
        elif percentile < 0.85:
            regime = 'high'
        else:
            regime = 'extreme'

        return regime, percentile

    def compute_entropy(self, returns: np.ndarray, bins: int = 20) -> float:
        """Shannon Entropy 계산"""
        returns = returns[~np.isnan(returns)]
        if len(returns) < 10 or not HAS_SCIPY:
            return 0.5

        if np.min(returns) == np.max(returns):
            return 0.5

        try:
            hist, _ = np.histogram(returns, bins=bins, density=True)
        except ValueError:
            return 0.5

        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.5

        hist = hist / np.sum(hist)
        max_entropy = np.log(bins)
        current_entropy = scipy_entropy(hist)
        normalized_entropy = current_entropy / max_entropy if max_entropy > 0 else 0

        return normalized_entropy

    def compute_energy_spike(self, volume: np.ndarray, volatility: np.ndarray) -> Tuple[float, bool]:
        """에너지 계산: E = Volume × Volatility²"""
        if len(volume) < 20 or len(volatility) < 20:
            return 1.0, False

        if np.isnan(volume[-1]) or np.isnan(volatility[-1]):
            return 1.0, False

        current_energy = volume[-1] * (volatility[-1] ** 2)
        historical_energy = volume[-20:] * (volatility[-20:] ** 2)
        mean_energy = np.nanmean(historical_energy)

        if mean_energy > 0:
            energy_ratio = current_energy / mean_energy
        else:
            energy_ratio = 1.0

        is_spike = energy_ratio > 2.0
        return energy_ratio, is_spike

    def enhanced_phase_a_detection(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        returns: np.ndarray,
        vol_ratio: float,
        current_rsi: float,
        idx: int
    ) -> Tuple[bool, bool, float, Dict[str, Any]]:
        """물리/수학 강화 Phase A 감지"""
        meta = {}

        start_idx = max(0, idx - self.lookback)
        price_slice = close[start_idx:idx+1]
        vol_slice = volume[start_idx:idx+1]
        ret_slice = returns[start_idx:idx+1]

        if len(price_slice) < 20:
            return False, False, 0.0, meta

        # Physics/Math features
        cycle_length, cycle_power = self.detect_dominant_cycle(price_slice)
        phase_rad, phase_position = self.compute_hilbert_phase(price_slice)
        vol_regime, vol_percentile = self.compute_mb_volatility_regime(ret_slice)
        entropy = self.compute_entropy(ret_slice)
        volatility = np.abs(ret_slice)
        energy_ratio, is_energy_spike = self.compute_energy_spike(vol_slice, volatility)

        meta.update({
            'fft_cycle_length': cycle_length,
            'hilbert_position': phase_position,
            'mb_regime': vol_regime,
            'entropy': entropy,
            'energy_ratio': energy_ratio,
        })

        # SC (Selling Climax) Detection
        sc_score = 0.0
        sc_required = 0
        recent_ret = ret_slice[-6:] if len(ret_slice) >= 6 else ret_slice

        if np.nanmin(recent_ret) < -0.018:
            sc_score += 0.20
            sc_required += 1
        elif np.nanmin(recent_ret) < -0.012:
            sc_score += 0.10

        if vol_ratio > 1.5:
            sc_score += 0.15
            sc_required += 1
        elif vol_ratio > 1.2:
            sc_score += 0.05

        if current_rsi < 35:
            sc_score += 0.15
            sc_required += 1
        elif current_rsi < 40:
            sc_score += 0.05

        if phase_position == 'bottom':
            sc_score += 0.15
        elif phase_position == 'falling':
            sc_score += 0.05

        if vol_regime == 'extreme':
            sc_score += 0.15
        elif vol_regime == 'high':
            sc_score += 0.05

        if entropy > 0.8:
            sc_score += 0.10

        if is_energy_spike:
            sc_score += 0.10

        is_sc = sc_score >= 0.65 and sc_required >= 2

        # BC (Buying Climax) Detection
        bc_score = 0.0
        bc_required = 0

        if np.nanmax(recent_ret) > 0.018:
            bc_score += 0.20
            bc_required += 1
        elif np.nanmax(recent_ret) > 0.012:
            bc_score += 0.10

        if vol_ratio > 1.5:
            bc_score += 0.15
            bc_required += 1
        elif vol_ratio > 1.2:
            bc_score += 0.05

        if current_rsi > 65:
            bc_score += 0.15
            bc_required += 1
        elif current_rsi > 60:
            bc_score += 0.05

        if phase_position == 'top':
            bc_score += 0.15
        elif phase_position == 'rising':
            bc_score += 0.05

        if vol_regime == 'extreme':
            bc_score += 0.15
        elif vol_regime == 'high':
            bc_score += 0.05

        if entropy > 0.8:
            bc_score += 0.10

        if is_energy_spike:
            bc_score += 0.10

        is_bc = bc_score >= 0.65 and bc_required >= 2

        confidence = max(sc_score, bc_score)
        meta['sc_score'] = sc_score
        meta['bc_score'] = bc_score

        return is_sc, is_bc, confidence, meta

    def enhanced_phase_b_detection(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        returns: np.ndarray,
        vol_ratio: float,
        current_rsi: float,
        z_score: float,
        in_box: bool,
        idx: int
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """물리/수학 강화 Phase B (횡보) 감지"""
        meta = {}

        start_idx = max(0, idx - self.lookback)
        price_slice = close[start_idx:idx+1]
        vol_slice = volume[start_idx:idx+1]
        ret_slice = returns[start_idx:idx+1]

        if len(price_slice) < 40:
            return False, 0.0, meta

        entropy = self.compute_entropy(ret_slice)
        vol_regime, vol_percentile = self.compute_mb_volatility_regime(ret_slice)
        volatility = np.abs(ret_slice)
        energy_ratio, is_energy_spike = self.compute_energy_spike(vol_slice, volatility)

        meta.update({
            'entropy': entropy,
            'mb_regime': vol_regime,
            'energy_ratio': energy_ratio,
        })

        # Phase B Score
        b_score = 0.0

        if in_box:
            b_score += 0.15
        if abs(z_score) < 1.5:
            b_score += 0.10
        if vol_ratio < 1.2:
            b_score += 0.10

        if entropy < 0.6:
            b_score += 0.15
        elif entropy < 0.75:
            b_score += 0.05

        if vol_regime in ['low', 'medium']:
            b_score += 0.15
        elif vol_regime == 'high':
            b_score -= 0.10

        if energy_ratio < 0.8:
            b_score += 0.10
        elif energy_ratio > 1.5:
            b_score -= 0.10

        if 40 <= current_rsi <= 60:
            b_score += 0.05

        meta['b_score'] = b_score
        is_phase_b = b_score >= 0.50 and in_box

        return is_phase_b, b_score, meta

    def enhanced_phase_c_detection(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        returns: np.ndarray,
        current_close: float,
        box_low: float,
        box_high: float,
        atr_val: float,
        vol_ratio: float,
        idx: int
    ) -> Tuple[bool, bool, float, Dict[str, Any]]:
        """물리/수학 강화 Phase C (Spring/UTAD) 감지"""
        meta = {}

        start_idx = max(0, idx - self.lookback)
        price_slice = close[start_idx:idx+1]
        vol_slice = volume[start_idx:idx+1]
        ret_slice = returns[start_idx:idx+1]

        if len(price_slice) < 20:
            return False, False, 0.0, meta

        # Breakout strength
        if current_close < box_low:
            breakout_strength = (box_low - current_close) / atr_val if atr_val > 0 else 0
            breakout_dir = 'down'
        elif current_close > box_high:
            breakout_strength = (current_close - box_high) / atr_val if atr_val > 0 else 0
            breakout_dir = 'up'
        else:
            breakout_strength = 0.0
            breakout_dir = 'none'

        meta['breakout_strength'] = breakout_strength
        meta['breakout_direction'] = breakout_dir

        # Spring Detection
        spring_score = 0.0
        if breakout_dir == 'down':
            if breakout_strength > 0.1:
                spring_score += 0.20
            if breakout_strength < 0.5:
                spring_score += 0.10
            if vol_ratio < 1.5:
                spring_score += 0.20

        is_spring = spring_score >= 0.30 and breakout_dir == 'down'

        # UTAD Detection
        utad_score = 0.0
        if breakout_dir == 'up':
            if breakout_strength > 0.1:
                utad_score += 0.20
            if breakout_strength < 0.5:
                utad_score += 0.10
            if vol_ratio < 1.5:
                utad_score += 0.20

        is_utad = utad_score >= 0.30 and breakout_dir == 'up'

        confidence = max(spring_score, utad_score)
        meta['spring_score'] = spring_score
        meta['utad_score'] = utad_score

        return is_spring, is_utad, confidence, meta


# Global instance
_physics_detector = PhysicsEnhancedPhaseDetector()


# ============================================================================
# Main Phase Detection Function
# ============================================================================

def detect_wyckoff_phase(
    df: pd.DataFrame,
    theta: Theta,
    lookback: int = 50
) -> pd.DataFrame:
    """
    Wyckoff 페이즈를 감지합니다. (6개 패턴 + 5개 Phase)

    Returns:
        DataFrame with columns: phase, sub_phase, direction, confidence, box_low, box_high
    """
    _atr = atr(df, theta.atr_len)
    box = box_engine_freeze(df, theta)
    _rsi = rsi(df['close'], 14)
    z = zscore(df['close'], 20)

    ema_50 = df['close'].ewm(span=50, adjust=False).mean()
    ema_200 = df['close'].ewm(span=200, adjust=False).mean()

    n = len(df)
    phases = pd.DataFrame(index=df.index)
    phases['phase'] = 'B'
    phases['sub_phase'] = 'Range'
    phases['direction'] = 'range'
    phases['confidence'] = 0.4
    phases['box_low'] = box['box_low']
    phases['box_high'] = box['box_high']

    vol_ma = df['volume'].rolling(20, min_periods=5).mean()
    vol_ratio = df['volume'] / (vol_ma + 1e-12)
    ret = df['close'].pct_change()

    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    vol_ratio_arr = vol_ratio.values
    ret_arr = ret.values
    atr_arr = _atr.values
    rsi_arr = _rsi.values
    z_arr = z.values
    bl_arr = box['box_low'].values
    bh_arr = box['box_high'].values
    bw_arr = box['box_width'].values
    ema_50_arr = ema_50.values
    ema_200_arr = ema_200.values

    current_phase = 'B'
    phase_start = 0
    direction = 'range'

    for i in range(lookback, n):
        if np.isnan(atr_arr[i]) or np.isnan(bl_arr[i]):
            continue

        close = close_arr[i]
        bl, bh = bl_arr[i], bh_arr[i]
        bw = bw_arr[i]
        at = atr_arr[i]
        curr_rsi = rsi_arr[i] if not np.isnan(rsi_arr[i]) else 50
        curr_z = z_arr[i] if not np.isnan(z_arr[i]) else 0
        vol_r = vol_ratio_arr[i]

        # Phase A Detection
        is_sc, is_bc, phys_confidence, phys_meta = _physics_detector.enhanced_phase_a_detection(
            close=close_arr,
            volume=df['volume'].values,
            returns=ret_arr,
            vol_ratio=vol_r,
            current_rsi=curr_rsi,
            idx=i
        )

        if is_sc:
            current_phase = 'A'
            direction = 'accumulation'
            phase_start = i
            phases.iloc[i, phases.columns.get_loc('phase')] = 'A'
            phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'SC'
            phases.iloc[i, phases.columns.get_loc('direction')] = 'accumulation'
            phases.iloc[i, phases.columns.get_loc('confidence')] = min(0.95, 0.5 + phys_confidence)
            continue

        if is_bc:
            current_phase = 'A'
            direction = 'distribution'
            phase_start = i
            phases.iloc[i, phases.columns.get_loc('phase')] = 'A'
            phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'BC'
            phases.iloc[i, phases.columns.get_loc('direction')] = 'distribution'
            phases.iloc[i, phases.columns.get_loc('confidence')] = min(0.95, 0.5 + phys_confidence)
            continue

        # Phase B Detection
        margin = 0.1 * bw if bw > 0 else 0
        in_box = (bl - margin) <= close <= (bh + margin)

        ema50 = ema_50_arr[i] if not np.isnan(ema_50_arr[i]) else close
        ema200 = ema_200_arr[i] if not np.isnan(ema_200_arr[i]) else close
        is_uptrend = ema50 > ema200
        is_downtrend = ema50 < ema200

        is_phase_b, b_confidence, b_meta = _physics_detector.enhanced_phase_b_detection(
            close=close_arr,
            volume=df['volume'].values,
            returns=ret_arr,
            vol_ratio=vol_r,
            current_rsi=curr_rsi,
            z_score=curr_z,
            in_box=in_box,
            idx=i
        )

        if is_phase_b:
            if is_uptrend and i >= 20:
                recent_lows = low_arr[i-20:i+1]
                first_half_low = np.min(recent_lows[:10])
                second_half_low = np.min(recent_lows[10:])
                if second_half_low > first_half_low:
                    current_phase = 'B'
                    direction = 're_accumulation'
                    phases.iloc[i, phases.columns.get_loc('phase')] = 'B'
                    phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'ReAccum'
                    phases.iloc[i, phases.columns.get_loc('direction')] = 're_accumulation'
                    phases.iloc[i, phases.columns.get_loc('confidence')] = min(0.95, 0.5 + b_confidence)
                    continue

            if is_downtrend and i >= 20:
                recent_highs = high_arr[i-20:i+1]
                first_half_high = np.max(recent_highs[:10])
                second_half_high = np.max(recent_highs[10:])
                if second_half_high < first_half_high:
                    current_phase = 'B'
                    direction = 're_distribution'
                    phases.iloc[i, phases.columns.get_loc('phase')] = 'B'
                    phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'ReDistrib'
                    phases.iloc[i, phases.columns.get_loc('direction')] = 're_distribution'
                    phases.iloc[i, phases.columns.get_loc('confidence')] = min(0.95, 0.5 + b_confidence)
                    continue

            if current_phase in ['A', 'B']:
                current_phase = 'B'
                phases.iloc[i, phases.columns.get_loc('phase')] = 'B'
                phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'ST'
                if direction == 'range':
                    phases.iloc[i, phases.columns.get_loc('direction')] = 'accumulation'
                else:
                    phases.iloc[i, phases.columns.get_loc('direction')] = direction
                phases.iloc[i, phases.columns.get_loc('confidence')] = min(0.90, 0.4 + b_confidence)
                continue

        # Phase C Detection
        is_spring, is_utad, c_confidence, c_meta = _physics_detector.enhanced_phase_c_detection(
            close=close_arr,
            volume=df['volume'].values,
            returns=ret_arr,
            current_close=close,
            box_low=bl,
            box_high=bh,
            atr_val=at,
            vol_ratio=vol_r,
            idx=i
        )

        if is_spring:
            current_phase = 'C'
            phases.iloc[i, phases.columns.get_loc('phase')] = 'C'
            phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'Spring'
            phases.iloc[i, phases.columns.get_loc('direction')] = 'accumulation'
            phases.iloc[i, phases.columns.get_loc('confidence')] = min(0.95, 0.5 + c_confidence)
            continue

        if is_utad:
            current_phase = 'C'
            phases.iloc[i, phases.columns.get_loc('phase')] = 'C'
            phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'UTAD'
            phases.iloc[i, phases.columns.get_loc('direction')] = 'distribution'
            phases.iloc[i, phases.columns.get_loc('confidence')] = min(0.95, 0.5 + c_confidence)
            continue

        # Phase D/E Detection (Markup/Markdown)
        if i >= 50:
            trend_strength = (ema50 - ema200) / close if close > 0 else 0

            if trend_strength > 0.015 and close > ema50 and is_uptrend:
                current_phase = 'E'
                direction = 'markup'
                phases.iloc[i, phases.columns.get_loc('phase')] = 'E'
                phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'Markup'
                phases.iloc[i, phases.columns.get_loc('direction')] = 'markup'
                phases.iloc[i, phases.columns.get_loc('confidence')] = 0.60
                continue

            if trend_strength < -0.015 and close < ema50 and is_downtrend:
                current_phase = 'E'
                direction = 'markdown'
                phases.iloc[i, phases.columns.get_loc('phase')] = 'E'
                phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'Markdown'
                phases.iloc[i, phases.columns.get_loc('direction')] = 'markdown'
                phases.iloc[i, phases.columns.get_loc('confidence')] = 0.60
                continue

        # Default: maintain previous state
        if i > 0:
            prev_phase = phases.iloc[i-1]['phase']
            prev_direction = phases.iloc[i-1]['direction']
            prev_sub = phases.iloc[i-1]['sub_phase']
            prev_conf = phases.iloc[i-1]['confidence']

            if prev_direction == 'unknown':
                prev_direction = 'range'
                prev_sub = 'Range'

            phases.iloc[i, phases.columns.get_loc('phase')] = prev_phase
            phases.iloc[i, phases.columns.get_loc('sub_phase')] = prev_sub
            phases.iloc[i, phases.columns.get_loc('direction')] = prev_direction
            phases.iloc[i, phases.columns.get_loc('confidence')] = max(0.3, prev_conf - 0.01)

    return phases


def get_direction_ratios(direction: str, is_spot: bool = False) -> Dict[str, float]:
    """방향에 따른 포지션 비율 반환"""
    return DIRECTION_RATIOS.get(direction, DIRECTION_RATIOS['unknown'])
