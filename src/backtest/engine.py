# -*- coding: utf-8 -*-
"""
Backtest Engine - V2 vs V3 Comparison
=====================================

통합 신호 백테스트 엔진.

주요 기능:
1. Historical data replay
2. Signal generation (V2 / V3)
3. Position management
4. Performance metrics (Sharpe, MDD, Win Rate)
5. Comparison report

사용법:
```python
from src.backtest.engine import BacktestEngine

engine = BacktestEngine(initial_equity=100000)
results = engine.run(df_dict, signal_version='v3')
engine.print_report(results)
```
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Any
from datetime import datetime
import numpy as np
import pandas as pd

# HMM Gate import
try:
    from ..gate import load_hmm_gate, HMMEntryGate
    HMM_GATE_AVAILABLE = True
except ImportError:
    HMM_GATE_AVAILABLE = False

# RiskManager import
try:
    from ..risk import RiskManager, RiskConfig
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False

# ProbabilityGate import
try:
    from ..regime.prob_gate import ProbabilityGate, ProbGateConfig
    from ..regime.upstream_scores import make_score_hilbert_1h, align_score_1h_to_15m
    PROB_GATE_AVAILABLE = True
except ImportError:
    PROB_GATE_AVAILABLE = False


@dataclass
class Trade:
    """개별 거래"""
    entry_time: datetime
    entry_price: float
    side: Literal['long', 'short']
    size: float  # 계약 수량

    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""

    pnl: float = 0.0
    pnl_pct: float = 0.0

    # 메타데이터
    signal_confidence: float = 0.0
    regime: str = ""
    mtf_boost: float = 0.0


@dataclass
class BacktestResult:
    """백테스트 결과"""
    # 기본 통계
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # 수익률
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0

    # 승률
    win_rate: float = 0.0

    # 리스크 지표
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # 평균
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # 기간
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0

    # 거래 목록
    trades: List[Trade] = field(default_factory=list)

    # 에퀴티 커브
    equity_curve: List[float] = field(default_factory=list)

    # 메타
    signal_version: str = ""
    initial_equity: float = 0.0


class BacktestEngine:
    """백테스트 엔진"""

    def __init__(
        self,
        initial_equity: float = 100000,
        risk_per_trade: float = 1.0,  # % of equity
        commission: float = 0.0004,  # 0.04% (Binance futures)
        slippage: float = 0.0001,  # 0.01%
        risk_config: Optional[Any] = None,  # RiskConfig
        prob_gate_config: Optional[Any] = None,  # ProbGateConfig
    ):
        self.initial_equity = initial_equity
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage

        self.equity = initial_equity
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.position: Optional[Trade] = None

        # RiskManager 초기화
        self.risk_manager = None
        if RISK_MANAGER_AVAILABLE:
            config = risk_config or RiskConfig(
                max_position_pct=10.0,
                daily_loss_limit_pct=3.0,
                consecutive_loss_limit=5,
            )
            self.risk_manager = RiskManager(equity=initial_equity, config=config)

        # ProbabilityGate 초기화
        self.prob_gate = None
        self.prob_gate_config = prob_gate_config
        self._prob_gate_result = None  # Pre-computed gate result for all bars

    def reset(self):
        """상태 초기화"""
        self.equity = self.initial_equity
        self.trades = []
        self.equity_curve = [self.initial_equity]
        self.position = None

        # RiskManager 리셋
        if self.risk_manager:
            self.risk_manager = RiskManager(
                equity=self.initial_equity,
                config=self.risk_manager.config,
            )

        # ProbabilityGate 리셋
        self._prob_gate_result = None

    def run(
        self,
        df_15m: pd.DataFrame,
        signal_version: Literal['v2', 'v3'] = 'v3',
        tf_dataframes: Optional[Dict[str, pd.DataFrame]] = None,
        use_mock_hmm: bool = False,
        df_1h: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        백테스트 실행

        Args:
            df_15m: 15분봉 DataFrame (index=timestamp)
            signal_version: 'v2' or 'v3'
            tf_dataframes: Multi-TF 데이터 (v3용)
            use_mock_hmm: True면 MockHMMGate 강제 사용 (테스트용)
            df_1h: 1H DataFrame for ProbabilityGate (optional)

        Returns:
            BacktestResult
        """
        self.reset()

        # 필수 컬럼 확인
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df_15m.columns:
                raise ValueError(f"Missing column: {col}")

        # ATR 계산 (없으면)
        if 'atr' not in df_15m.columns:
            df_15m = self._add_atr(df_15m.copy())

        # ProbabilityGate 초기화 (df_1h 제공 시)
        if PROB_GATE_AVAILABLE and self.prob_gate_config is not None and df_1h is not None:
            try:
                self._init_prob_gate(df_15m, df_1h)
                print(f"[INFO] ProbabilityGate initialized (config: temp_mode={self.prob_gate_config.temp_mode}, p_shrink={self.prob_gate_config.p_shrink})")
            except Exception as e:
                print(f"[WARN] Failed to initialize ProbabilityGate: {e}")
                self._prob_gate_result = None

        # HMM Gate 로드 (실제 모델 우선, 없으면 Mock fallback)
        hmm_gate = None
        use_real_hmm_flag = False

        if use_mock_hmm:
            hmm_gate = MockHMMGate()
            print("[INFO] Using MockHMMGate (forced by use_mock_hmm=True)")
        elif HMM_GATE_AVAILABLE:
            try:
                hmm_gate = load_hmm_gate()
                use_real_hmm_flag = True
                print(f"[INFO] Real HMM Gate loaded ({len(hmm_gate.posterior_map):,} timestamps)")
            except FileNotFoundError as e:
                print(f"[WARN] HMM model files not found: {e}")
            except Exception as e:
                print(f"[WARN] Failed to load HMM Gate: {e}")

        if hmm_gate is None:
            hmm_gate = MockHMMGate()
            print("[WARN] Using MockHMMGate (always allows entry)")

        # TFPredictor 초기화 (v3용)
        tf_predictor = None
        if signal_version == 'v3' and tf_dataframes:
            try:
                from ..context.tf_predictor import TFPredictor
                tf_predictor = TFPredictor()
                tf_predictor.load_data(tf_dataframes)
            except ImportError:
                print("[WARN] TFPredictor not available, falling back to v2")
                signal_version = 'v2'

        # 메인 루프
        start_idx = 100  # 충분한 lookback

        for i in range(start_idx, len(df_15m)):
            current_bar = df_15m.iloc[i]
            current_time = df_15m.index[i]
            current_price = float(current_bar['close'])
            current_atr = float(current_bar['atr']) if 'atr' in current_bar else current_price * 0.02

            # 1. 포지션 관리 (청산 체크)
            if self.position:
                exit_signal = self._check_exit(current_price, current_bar)
                if exit_signal:
                    self._close_position(current_time, current_price, exit_signal)

            # 2. 신호 생성
            if self.position is None:
                df_slice = df_15m.iloc[:i+1].copy()

                signal = self._generate_signal(
                    df_slice,
                    hmm_gate,
                    current_time,
                    signal_version,
                    tf_predictor,
                )

                if signal and signal['allowed']:
                    self._open_position(
                        current_time,
                        current_price,
                        current_atr,
                        signal,
                    )

            # 3. 에퀴티 기록
            unrealized_pnl = 0.0
            if self.position:
                if self.position.side == 'long':
                    unrealized_pnl = (current_price - self.position.entry_price) * self.position.size
                else:
                    unrealized_pnl = (self.position.entry_price - current_price) * self.position.size

            self.equity_curve.append(self.equity + unrealized_pnl)

        # 종료 시 포지션 청산
        if self.position:
            final_price = float(df_15m['close'].iloc[-1])
            final_time = df_15m.index[-1]
            self._close_position(final_time, final_price, "END_OF_DATA")

        return self._calculate_results(signal_version, df_15m)

    def _generate_signal(
        self,
        df: pd.DataFrame,
        hmm_gate,
        current_time,
        version: str,
        tf_predictor=None,
    ) -> Optional[Dict[str, Any]]:
        """신호 생성 (ProbabilityGate AND 필터 포함)"""
        try:
            base_signal = None

            if version == 'v2':
                from ..anchor.unified_signal import check_unified_long_signal_v2

                result = check_unified_long_signal_v2(
                    df, hmm_gate, current_time
                )

                base_signal = {
                    'allowed': result.allowed,
                    'side': result.side,
                    'confidence': result.confidence,
                    'size_mult': result.size_mult,
                    'regime': result.regime,
                    'mtf_boost': 0.0,
                }

            elif version == 'v3':
                # V3: Legacy + MTF Boost
                from ..anchor.unified_signal import check_unified_long_signal_v2
                from ..anchor.legacy_pipeline import analyze_confluence

                # Legacy 신호
                legacy_result = check_unified_long_signal_v2(
                    df, hmm_gate, current_time
                )

                # MTF Boost
                mtf_boost = 0.0
                if tf_predictor and tf_predictor.hierarchy:
                    current_price = float(df['close'].iloc[-1])
                    prediction = tf_predictor.predict_next_move()
                    mtf_boost = self._calc_mtf_boost(current_price, prediction.confluence_zones)

                # 최종 점수
                final_confidence = legacy_result.confidence + mtf_boost

                base_signal = {
                    'allowed': legacy_result.allowed and final_confidence >= 0.3,
                    'side': legacy_result.side,
                    'confidence': final_confidence,
                    'size_mult': legacy_result.size_mult,
                    'regime': legacy_result.regime,
                    'mtf_boost': mtf_boost,
                }

            # ProbabilityGate AND 필터 적용
            if base_signal is not None and base_signal['allowed']:
                gate_allowed, p_bull = self._check_prob_gate(current_time, base_signal['side'])
                base_signal['prob_gate_allowed'] = gate_allowed
                base_signal['p_bull'] = p_bull
                # AND 필터: base_signal.allowed AND gate_allowed
                base_signal['allowed'] = base_signal['allowed'] and gate_allowed

            return base_signal

        except Exception as e:
            # 신호 생성 실패 시
            return None

        return None

    def _calc_mtf_boost(self, current_price: float, zones: List) -> float:
        """MTF Confluence Boost 계산"""
        boost = 0.0

        for zone in zones:
            zone_price = getattr(zone, 'price', None)
            zone_strength = getattr(zone, 'strength', 1)

            if zone_price is None:
                continue

            distance_pct = abs((zone_price - current_price) / current_price * 100)

            if distance_pct <= 3.0:
                if zone_strength >= 4:
                    boost = max(boost, 0.3)
                elif zone_strength >= 3:
                    boost = max(boost, 0.2)
                elif zone_strength >= 2:
                    boost = max(boost, 0.1)

        return min(boost, 0.4)

    def _open_position(
        self,
        time: datetime,
        price: float,
        atr: float,
        signal: Dict,
    ) -> bool:
        """포지션 진입 (리스크 체크 포함)

        Returns:
            True if position opened, False if blocked by risk manager
        """
        # 슬리피지 적용
        if signal['side'] == 'long':
            entry_price = price * (1 + self.slippage)
        else:
            entry_price = price * (1 - self.slippage)

        # 포지션 사이즈 계산
        risk_amount = self.equity * self.risk_per_trade / 100
        sl_distance = atr * 2.0  # ATR x 2
        size = risk_amount / sl_distance * signal.get('size_mult', 1.0)

        # RiskManager 체크
        if self.risk_manager:
            allowed, reason = self.risk_manager.can_open_position(
                size=size,
                price=entry_price,
                side=signal['side'],
            )
            if not allowed:
                # 리스크 체크 실패 - 진입 차단
                return False

            # 포지션 오픈 기록
            self.risk_manager.open_position()

        # 커미션
        commission_cost = entry_price * size * self.commission
        self.equity -= commission_cost

        self.position = Trade(
            entry_time=time,
            entry_price=entry_price,
            side=signal['side'],
            size=size,
            signal_confidence=signal['confidence'],
            regime=signal.get('regime', ''),
            mtf_boost=signal.get('mtf_boost', 0.0),
        )
        return True

    def _check_exit(self, current_price: float, current_bar) -> Optional[str]:
        """청산 조건 체크 (간단 버전)"""
        if self.position is None:
            return None

        entry = self.position.entry_price
        atr = float(current_bar.get('atr', entry * 0.02))

        # SL: ATR x 2
        sl_distance = atr * 2.0

        # TP: ATR x 3 (R:R 1.5)
        tp_distance = atr * 3.0

        if self.position.side == 'long':
            sl_price = entry - sl_distance
            tp_price = entry + tp_distance

            if current_price <= sl_price:
                return "STOP_LOSS"
            if current_price >= tp_price:
                return "TAKE_PROFIT"
        else:
            sl_price = entry + sl_distance
            tp_price = entry - tp_distance

            if current_price >= sl_price:
                return "STOP_LOSS"
            if current_price <= tp_price:
                return "TAKE_PROFIT"

        return None

    def _close_position(self, time: datetime, price: float, reason: str):
        """포지션 청산"""
        if self.position is None:
            return

        # 슬리피지 적용
        if self.position.side == 'long':
            exit_price = price * (1 - self.slippage)
            pnl = (exit_price - self.position.entry_price) * self.position.size
        else:
            exit_price = price * (1 + self.slippage)
            pnl = (self.position.entry_price - exit_price) * self.position.size

        # 커미션
        commission_cost = exit_price * self.position.size * self.commission
        pnl -= commission_cost

        # PnL %
        pnl_pct = pnl / self.initial_equity * 100

        # 업데이트
        self.position.exit_time = time
        self.position.exit_price = exit_price
        self.position.exit_reason = reason
        self.position.pnl = pnl
        self.position.pnl_pct = pnl_pct

        self.equity += pnl
        self.trades.append(self.position)

        # RiskManager에 거래 기록
        if self.risk_manager:
            self.risk_manager.record_trade(pnl=pnl, is_closed=True)
            # equity 동기화
            self.risk_manager.equity = self.equity

        self.position = None

    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """ATR 추가"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df['atr'] = tr.ewm(span=period, adjust=False).mean()
        return df

    def _init_prob_gate(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame):
        """ProbabilityGate 초기화 (1H Hilbert score 기반)

        1H Hilbert score를 계산하고 15m에 정렬한 뒤,
        전체 15m 구간에 대해 ProbabilityGate 결과를 미리 계산.
        """
        if not PROB_GATE_AVAILABLE:
            raise ImportError("ProbabilityGate not available")

        # 1. 1H Hilbert score 계산
        score_1h = make_score_hilbert_1h(df_1h)

        # 2. 15m에 정렬 (forward-fill + shift(1) for 'open' semantics)
        score_15m = align_score_1h_to_15m(score_1h, df_15m, timestamp_semantics='open')

        # 3. ProbabilityGate 계산
        self.prob_gate = ProbabilityGate(self.prob_gate_config)

        # 15m OHLC 추출
        close_15m = df_15m['close'].values
        high_15m = df_15m['high'].values
        low_15m = df_15m['low'].values
        score_raw = score_15m.values

        self._prob_gate_result = self.prob_gate.compute(score_raw, close_15m, high_15m, low_15m)
        self._prob_gate_result.index = df_15m.index

    def _check_prob_gate(self, current_time, signal_side: str) -> Tuple[bool, float]:
        """ProbabilityGate 체크

        Args:
            current_time: 현재 시간 (df_15m index)
            signal_side: 'long' or 'short'

        Returns:
            (allowed, p_bull) tuple
        """
        if self._prob_gate_result is None:
            return True, 0.5  # Gate 없으면 항상 허용

        if current_time not in self._prob_gate_result.index:
            return True, 0.5

        row = self._prob_gate_result.loc[current_time]

        if not row['valid']:
            return True, 0.5  # warmup 구간은 허용

        p_bull = row['p_bull']
        action_code = row['action_code']

        # Gate 로직: action_code와 signal_side 일치 여부
        if signal_side == 'long':
            allowed = action_code == 1  # LONG 허용
        else:  # short
            allowed = action_code == -1  # SHORT 허용

        return allowed, p_bull

    def _calculate_results(self, version: str, df: pd.DataFrame) -> BacktestResult:
        """결과 계산"""
        if not self.trades:
            return BacktestResult(
                signal_version=version,
                initial_equity=self.initial_equity,
                equity_curve=self.equity_curve,  # 거래 없어도 equity_curve는 반환
                start_date=df.index[0] if len(df) > 0 else None,
                end_date=df.index[-1] if len(df) > 0 else None,
                trading_days=len(set(df.index.date)) if len(df) > 0 else 0,
            )

        # 기본 통계
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in self.trades)
        total_pnl_pct = total_pnl / self.initial_equity * 100

        # 승률
        win_rate = len(winning) / len(self.trades) * 100 if self.trades else 0

        # 평균
        avg_win = np.mean([t.pnl_pct for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl_pct for t in losing]) if losing else 0

        # Profit Factor
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Drawdown
        equity_array = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array)
        drawdown_pct = drawdown / peak * 100

        max_dd = np.max(drawdown)
        max_dd_pct = np.max(drawdown_pct)

        # Sharpe Ratio (일간 수익률 기준)
        returns = np.diff(equity_array) / equity_array[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 96)  # 15m bars
        else:
            sharpe = 0.0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and np.std(downside_returns) > 0:
            sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252 * 96)
        else:
            sortino = 0.0

        # Calmar Ratio
        annual_return = total_pnl_pct * 365 / len(self.equity_curve) * 96
        calmar = annual_return / max_dd_pct if max_dd_pct > 0 else 0.0

        return BacktestResult(
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),

            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            avg_pnl_pct=np.mean([t.pnl_pct for t in self.trades]),

            win_rate=win_rate,

            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,

            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,

            start_date=df.index[0],
            end_date=df.index[-1],
            trading_days=len(set(df.index.date)),

            trades=self.trades,
            equity_curve=self.equity_curve,

            signal_version=version,
            initial_equity=self.initial_equity,
        )

    @staticmethod
    def compare(v2_result: BacktestResult, v3_result: BacktestResult) -> Dict:
        """V2 vs V3 비교"""
        comparison = {
            'metric': [],
            'v2': [],
            'v3': [],
            'diff': [],
            'winner': [],
        }

        metrics = [
            ('Total Trades', 'total_trades', False),
            ('Win Rate (%)', 'win_rate', True),
            ('Total PnL (%)', 'total_pnl_pct', True),
            ('Sharpe Ratio', 'sharpe_ratio', True),
            ('Sortino Ratio', 'sortino_ratio', True),
            ('Max Drawdown (%)', 'max_drawdown_pct', False),
            ('Profit Factor', 'profit_factor', True),
            ('Avg Win (%)', 'avg_win', True),
            ('Avg Loss (%)', 'avg_loss', False),
        ]

        for name, attr, higher_is_better in metrics:
            v2_val = getattr(v2_result, attr)
            v3_val = getattr(v3_result, attr)
            diff = v3_val - v2_val

            if higher_is_better:
                winner = 'V3' if diff > 0 else 'V2' if diff < 0 else 'TIE'
            else:
                winner = 'V2' if diff > 0 else 'V3' if diff < 0 else 'TIE'

            comparison['metric'].append(name)
            comparison['v2'].append(v2_val)
            comparison['v3'].append(v3_val)
            comparison['diff'].append(diff)
            comparison['winner'].append(winner)

        return comparison

    @staticmethod
    def print_report(result: BacktestResult):
        """결과 출력"""
        print("=" * 60)
        print(f"Backtest Report - {result.signal_version.upper()}")
        print("=" * 60)

        print(f"\nPeriod: {result.start_date} ~ {result.end_date}")
        print(f"Trading Days: {result.trading_days}")
        print(f"Initial Equity: ${result.initial_equity:,.0f}")

        print(f"\n--- Performance ---")
        print(f"Total Trades: {result.total_trades}")
        print(f"Win/Loss: {result.winning_trades}/{result.losing_trades}")
        print(f"Win Rate: {result.win_rate:.1f}%")

        print(f"\n--- Returns ---")
        print(f"Total PnL: ${result.total_pnl:,.0f} ({result.total_pnl_pct:+.2f}%)")
        print(f"Avg PnL: {result.avg_pnl_pct:+.2f}%")
        print(f"Avg Win: {result.avg_win:+.2f}%")
        print(f"Avg Loss: {result.avg_loss:+.2f}%")

        print(f"\n--- Risk Metrics ---")
        print(f"Max Drawdown: ${result.max_drawdown:,.0f} ({result.max_drawdown_pct:.1f}%)")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
        print(f"Calmar Ratio: {result.calmar_ratio:.2f}")
        print(f"Profit Factor: {result.profit_factor:.2f}")

        # Quant Researcher: 통계적 유의성 경고
        if result.total_trades < 30:
            print(f"\n[WARN] Low sample size ({result.total_trades} trades). Results may not be statistically significant.")
            print(f"       Recommended minimum: 30 trades for reliable metrics.")

        # Veteran Trader: 실전 트레이딩 경고
        print(f"\n[NOTE] Backtest assumptions:")
        print(f"       - Slippage: 0.01% (may be optimistic in volatile markets)")
        print(f"       - Liquidity: Assumes instant fill (may not hold for large positions)")
        print(f"       - Fees: Binance futures default (adjust for other exchanges)")

        print("=" * 60)

    @staticmethod
    def print_comparison(comparison: Dict):
        """비교 결과 출력"""
        print("\n" + "=" * 70)
        print("V2 vs V3 Comparison")
        print("=" * 70)

        print(f"\n{'Metric':<20} {'V2':>12} {'V3':>12} {'Diff':>12} {'Winner':>10}")
        print("-" * 70)

        for i in range(len(comparison['metric'])):
            metric = comparison['metric'][i]
            v2 = comparison['v2'][i]
            v3 = comparison['v3'][i]
            diff = comparison['diff'][i]
            winner = comparison['winner'][i]

            # 포맷팅
            if isinstance(v2, float):
                v2_str = f"{v2:.2f}"
                v3_str = f"{v3:.2f}"
                diff_str = f"{diff:+.2f}"
            else:
                v2_str = str(v2)
                v3_str = str(v3)
                diff_str = f"{diff:+d}"

            print(f"{metric:<20} {v2_str:>12} {v3_str:>12} {diff_str:>12} {winner:>10}")

        print("=" * 70)


class MockHMMGate:
    """Mock HMM Gate for backtesting"""

    def check_entry(self, ts, side):
        """항상 허용"""
        return MockGateDecision()


class MockGateDecision:
    """Mock gate decision"""
    allowed = True
    size_mult = 1.0
    state = 'accumulation'
    expected_var = 0.02
    markdown_prob = 0.1
    cooldown_active = False
    blocked_reason = None