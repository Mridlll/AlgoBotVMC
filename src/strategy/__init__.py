"""Trading strategy components."""

from .signals import SignalDetector, Signal, SignalType, SignalState, SimpleSignalDetector, Bias
from .risk import RiskManager
from .trade_manager import TradeManager
from .multi_timeframe import (
    MultiTimeframeCoordinator,
    MultiTimeframeSignal,
    TimeframeSignal,
    BiasResult,
    BiasDirection,
)
from .mtf_bias import MTFBiasCalculator, MTFBiasResult, HTFAnalysis
from .volatility_regime import VolatilityRegimeDetector, VolatilityRegime, RegimeAnalysis
from .regime import RegimeDetector, MarketRegime, RegimeResult, RegimeAction
from .mtf_divergence_scanner import (
    MTFDivergenceScanner,
    MTFScanResult,
    MTFDivergenceSignal,
    scan_for_divergences,
    STANDARD_TIMEFRAMES,
    TIMEFRAME_WEIGHTS,
)
from .partial_exits import (
    PartialExitManager,
    PartialExitPosition,
    ExitLeg,
    ExitReason,
)

__all__ = [
    "SignalDetector",
    "SimpleSignalDetector",
    "Signal",
    "SignalType",
    "SignalState",
    "Bias",
    "RiskManager",
    "TradeManager",
    "MultiTimeframeCoordinator",
    "MultiTimeframeSignal",
    "TimeframeSignal",
    "BiasResult",
    "BiasDirection",
    "MTFBiasCalculator",
    "MTFBiasResult",
    "HTFAnalysis",
    "VolatilityRegimeDetector",
    "VolatilityRegime",
    "RegimeAnalysis",
    "RegimeDetector",
    "MarketRegime",
    "RegimeResult",
    "RegimeAction",
    "MTFDivergenceScanner",
    "MTFScanResult",
    "MTFDivergenceSignal",
    "scan_for_divergences",
    "STANDARD_TIMEFRAMES",
    "TIMEFRAME_WEIGHTS",
    "PartialExitManager",
    "PartialExitPosition",
    "ExitLeg",
    "ExitReason",
]
