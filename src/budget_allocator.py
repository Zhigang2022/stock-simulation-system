from typing import List, Dict, Tuple, Any
import pandas as pd

class IntegratedBudgetAllocator:
    """
    Module D1: Unified Optimization & Capital Location Layer.
    
    Consolidates target weight assignments for the core strategic sleeve and 
    performs dynamic budget/liquidity verification for tactical ad-hoc orders.
    
    Attributes:
        top_percent (float): Quantile threshold (e.g., 0.10 for top 10%) to filter signals.
        allocation_type (str): Methodology to distribute weight ('equal' or 'score_weighted').
    """
    def __init__(self, top_percent: float = 0.10, allocation_type: str = "equal"):
        self.top_percent = top_percent
        self.allocation_type = allocation_type

    def allocate_capital(self, 
                         state: Any, 
                         current_prices: Any) -> Tuple[Dict[str, float], List[Any],List[Any]]:
        """
        Translates raw cross-sectional strategy scores into target weights and filters 
        tactical ad-hoc orders against available sub-sleeve capital constraints.

        Args:
            regular_signals (List[SignalPayload]): Raw strategic signals generated on rebalance dates. 
                Can be empty on non-rebalance days.
            ad_hoc_signals (List[SignalPayload]): Tactical risk overrides or instant exit/entry loops 
                evaluated daily.
            state (GlobalState): The persistent portfolio state containing cash and holdings.
            current_prices (pd.Series): Current timestamp market prices used for valuation checks.

        Returns:
            Tuple[Dict[str, float], List[SignalPayload]]:
                - core_target_weights (Dict[str, float]): Map of ticker to its strict percentage 
                  allocation within the strategic sleeve. 
                  NOTE: If `regular_signals` is empty, returns an empty dictionary `{}`. Downstream 
                  execution modules MUST interpret an empty dict as a 'HOLD' command for existing 
                  strategic positions, rather than a liquidation command.
                - sell_ad_hoc_orders (List[SignalPayload]): Validated, risk-checked subset of 
                  the input `ad_hoc_signals` cleared for immediate execution.
                - buy_ad_hoc_orders (List[SignalPayload]): Validated, risk-checked subset of 
                  the input `ad_hoc_signals` cleared for immediate execution.
        """
        # -------------------------------------------------------------
        # PART A: REGULAR REBALANCE WEIGHT CONFIGURATION (SLOW SLEEVE)
        # -------------------------------------------------------------
        core_target_weights: Dict[str, float] = {}
        signals=getattr(state,'signals')
        regular_signals=getattr(signals.regular_strategy_output,'signals',None)
        ad_hoc_signals=getattr(signals.adhoc_strategy_output,'signals',None) 
          
        if regular_signals:
            # Isolate scores
            scores_series = pd.Series({s.ticker: s.score for s in regular_signals})
            
            # Keep only top tier performers matching your quantile limit
            cutoff_quantile = 1.0 - self.top_percent
            threshold_score = scores_series.quantile(cutoff_quantile)
            top_candidates = scores_series[scores_series >= threshold_score]
            
            if not top_candidates.empty:
                if self.allocation_type == "equal":
                    weight = 1.0 / len(top_candidates)
                    core_target_weights = {ticker: weight for ticker in top_candidates.index}
                elif self.allocation_type == "score_weighted":
                    total_sum = top_candidates.sum()
                    if total_sum > 0:
                        core_target_weights = {ticker: score / total_sum for ticker, score in top_candidates.items()}

        # -------------------------------------------------------------
        # PART B: AD-HOC TRANSACTION BUDGET VALIDATION (FAST SLEEVE)
        # -------------------------------------------------------------
        sell_adhoc_orders: List[Any] = []
        buy_adhoc_orders: List[Any] = []
        
        if ad_hoc_signals:
            for signal in ad_hoc_signals:
                if signal.kind == "AD_HOC_EXIT":
                    # Exits always clear risk immediately, no budget allocation constraint applied
                    sell_adhoc_orders.append(signal)
                    
                elif signal.kind == "AD_HOC_BUY":
                    # TODO
                    # WARNING: Ensure synchronization with your downstream transaction executor module.
                    # Dynamic Budget check: Ensure tactical sleeve cash can fulfill trade
                    # Suppose an ad-hoc buy commands a fixed allocation (e.g., 10% of total tactical NAV)
                    estimated_cost = state.calculate_sleeve_value(state.tactical_cash, state.tactical_positions, current_prices) * 0.10
                    
                    if state.tactical_cash >= estimated_cost:
                        buy_adhoc_orders.append(signal)
                    else:
                        print(f"AD-HOC BUY REJECTED: Insufficient Tactical Sleeve Cash to allocate to {signal.ticker}")

        return core_target_weights, sell_adhoc_orders,buy_adhoc_orders