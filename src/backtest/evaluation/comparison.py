from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class StrategyRanking:
    """Strategy ranking results"""

    name: str
    composite_score: float
    rank: int
    scores: Dict[str, float]


class StrategyComparator:
    """
    Strategy comparison analysis class
    Implements multi-dimensional strategy evaluation and ranking
    """

    def __init__(self):
        # Evaluation dimension weights
        self.evaluation_weights = {
            "return": 0.25,  # Return performance
            "risk": 0.20,  # Risk control
            "stability": 0.20,  # Stability
            "efficiency": 0.15,  # Efficiency metrics
            "robustness": 0.20,  # Robustness
        }

    def rank_strategies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive ranking of strategies

        Args:
            results: Strategy results dictionary

        Returns:
            Dict: Ranking results
        """
        if len(results) < 2:
            return {"error": "Need at least 2 strategies for comparison"}

        # Calculate dimension scores
        dimension_scores = self._calculate_dimension_scores(results)

        # Calculate composite scores
        composite_scores = self._calculate_composite_scores(dimension_scores)

        # Generate rankings
        rankings = self._generate_rankings(composite_scores, dimension_scores)

        # Analyze results
        analysis = self._analyze_rankings(rankings, dimension_scores)

        return {
            "rankings": rankings,
            "dimension_scores": dimension_scores,
            "analysis": analysis,
            "methodology": self._get_methodology_description(),
        }

    def _calculate_dimension_scores(
        self, results: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate dimension scores"""
        dimension_scores = {}

        for strategy_name, result in results.items():
            metrics = result.performance_metrics

            # Return dimension score
            return_score = self._calculate_return_score(metrics)

            # Risk dimension score
            risk_score = self._calculate_risk_score(metrics)

            # Stability dimension score
            stability_score = self._calculate_stability_score(metrics)

            # Efficiency dimension score
            efficiency_score = self._calculate_efficiency_score(metrics)

            # Robustness dimension score
            robustness_score = self._calculate_robustness_score(metrics, result)

            dimension_scores[strategy_name] = {
                "return": return_score,
                "risk": risk_score,
                "stability": stability_score,
                "efficiency": efficiency_score,
                "robustness": robustness_score,
            }

        # Normalize scores (0-100 points)
        normalized_scores = self._normalize_scores(dimension_scores)

        return normalized_scores

    def _calculate_return_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate return score"""
        annual_return = metrics.get("annual_return", 0)
        total_return = metrics.get("total_return", 0)

        # Score based on annualized return
        return_score = min(
            100, max(0, annual_return * 500 + 50)
        )  # 10% annual return = 100 points

        # Total return adjustment
        if total_return > 0:
            return_score = min(100, return_score * (1 + total_return * 0.1))

        return return_score

    def _calculate_risk_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate risk score (lower risk = higher score)"""
        volatility = metrics.get("volatility", 0)
        max_drawdown = abs(metrics.get("max_drawdown", 0))
        var_95 = abs(metrics.get("var_95", 0))

        # Volatility score (annualized volatility below 15% = full score)
        vol_score = max(0, 100 - volatility * 500)

        # Maximum drawdown score (below 5% = full score)
        dd_score = max(0, 100 - max_drawdown * 2000)

        # VaR score
        var_score = max(0, 100 - var_95 * 5000)

        # Comprehensive risk score
        risk_score = vol_score * 0.4 + dd_score * 0.4 + var_score * 0.2

        return risk_score

    def _calculate_stability_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate stability score"""
        stability_ratio = metrics.get("stability", 0)
        win_rate = metrics.get("win_rate", 0)
        profit_loss_ratio = metrics.get("profit_loss_ratio", 0)

        # Stability ratio score
        stability_score = stability_ratio * 100

        # Win rate score
        win_rate_score = win_rate * 100

        # Profit/loss ratio score
        pl_score = min(100, profit_loss_ratio * 50)

        # Comprehensive stability score
        total_score = stability_score * 0.4 + win_rate_score * 0.3 + pl_score * 0.3

        return total_score

    def _calculate_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score"""
        sharpe_ratio = metrics.get("sharpe_ratio", 0)
        sortino_ratio = metrics.get("sortino_ratio", 0)
        calmar_ratio = metrics.get("calmar_ratio", 0)

        # Sharpe ratio score (above 2.0 = full score)
        sharpe_score = min(100, max(0, sharpe_ratio * 50))

        # Sortino ratio score
        sortino_score = min(100, max(0, sortino_ratio * 40))

        # Calmar ratio score
        calmar_score = min(100, max(0, calmar_ratio * 20))

        # Comprehensive efficiency score
        efficiency_score = sharpe_score * 0.5 + sortino_score * 0.3 + calmar_score * 0.2

        return efficiency_score

    def _calculate_robustness_score(
        self, metrics: Dict[str, Any], result: Any
    ) -> float:
        """Calculate robustness score"""
        skewness = metrics.get("skewness", 0)
        kurtosis = metrics.get("kurtosis", 0)
        tail_ratio = metrics.get("tail_ratio", 1)
        recovery_factor = metrics.get("recovery_factor", 0)

        # Skewness score (closer to 0 is better)
        skew_score = max(0, 100 - abs(skewness) * 50)

        # Kurtosis score (closer to normal distribution is better)
        kurt_score = max(0, 100 - abs(kurtosis) * 20)

        # Tail ratio score
        tail_score = max(0, 100 - abs(tail_ratio - 1) * 100)

        # Recovery factor score
        recovery_score = min(100, recovery_factor * 10)

        # Comprehensive robustness score
        robustness_score = (
            skew_score * 0.25
            + kurt_score * 0.25
            + tail_score * 0.25
            + recovery_score * 0.25
        )

        return robustness_score

    def _normalize_scores(
        self, dimension_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Normalize scores"""
        normalized = {}

        # Get all dimensions
        dimensions = list(next(iter(dimension_scores.values())).keys())

        for dim in dimensions:
            # Get scores for all strategies in this dimension
            dim_scores = [scores[dim] for scores in dimension_scores.values()]

            if len(set(dim_scores)) == 1:  # All scores are the same
                for strategy in dimension_scores:
                    if strategy not in normalized:
                        normalized[strategy] = {}
                    normalized[strategy][dim] = 50.0  # Give medium score
            else:
                # Min-Max normalization to 0-100 range
                min_score = min(dim_scores)
                max_score = max(dim_scores)

                for strategy, scores in dimension_scores.items():
                    if strategy not in normalized:
                        normalized[strategy] = {}

                    if max_score != min_score:
                        normalized_score = (
                            (scores[dim] - min_score) / (max_score - min_score) * 100
                        )
                    else:
                        normalized_score = 50.0

                    normalized[strategy][dim] = normalized_score

        return normalized

    def _calculate_composite_scores(
        self, dimension_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate composite scores"""
        composite_scores = {}

        for strategy, scores in dimension_scores.items():
            composite_score = sum(
                scores[dim] * self.evaluation_weights[dim] for dim in scores
            )
            composite_scores[strategy] = composite_score

        return composite_scores

    def _generate_rankings(
        self,
        composite_scores: Dict[str, float],
        dimension_scores: Dict[str, Dict[str, float]],
    ) -> List[StrategyRanking]:
        """Generate strategy rankings"""
        # Sort by composite score
        sorted_strategies = sorted(
            composite_scores.items(), key=lambda x: x[1], reverse=True
        )

        rankings = []
        for rank, (strategy_name, composite_score) in enumerate(sorted_strategies, 1):
            ranking = StrategyRanking(
                name=strategy_name,
                composite_score=composite_score,
                rank=rank,
                scores=dimension_scores[strategy_name],
            )
            rankings.append(ranking)

        return rankings

    def _analyze_rankings(
        self,
        rankings: List[StrategyRanking],
        dimension_scores: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """Analyze ranking results"""
        if not rankings:
            return {}

        top_strategy = rankings[0]

        # Find best strategy in each dimension
        dimension_leaders = {}
        for dim in self.evaluation_weights.keys():
            best_strategy = max(dimension_scores.items(), key=lambda x: x[1][dim])
            dimension_leaders[dim] = {
                "strategy": best_strategy[0],
                "score": best_strategy[1][dim],
            }

        # Analyze score distribution
        all_scores = [r.composite_score for r in rankings]
        score_analysis = {
            "mean": np.mean(all_scores),
            "std": np.std(all_scores),
            "range": max(all_scores) - min(all_scores),
            "cv": (
                np.std(all_scores) / np.mean(all_scores)
                if np.mean(all_scores) != 0
                else 0
            ),
        }

        return {
            "top_performer": {
                "name": top_strategy.name,
                "score": top_strategy.composite_score,
                "strengths": self._identify_strengths(top_strategy.scores),
                "weaknesses": self._identify_weaknesses(top_strategy.scores),
            },
            "dimension_leaders": dimension_leaders,
            "score_distribution": score_analysis,
            "competitive_landscape": self._analyze_competition(rankings),
        }

    def _identify_strengths(self, scores: Dict[str, float]) -> List[str]:
        """Identify strategy strengths"""
        strengths = []
        for dim, score in scores.items():
            if score >= 75:  # High score threshold
                strengths.append(f"{dim} performance excellent ({score:.1f} points)")
        return strengths

    def _identify_weaknesses(self, scores: Dict[str, float]) -> List[str]:
        """Identify strategy weaknesses"""
        weaknesses = []
        for dim, score in scores.items():
            if score <= 30:  # Low score threshold
                weaknesses.append(f"{dim} needs improvement ({score:.1f} points)")
        return weaknesses

    def _analyze_competition(self, rankings: List[StrategyRanking]) -> Dict[str, Any]:
        """Analyze competitive landscape"""
        if len(rankings) < 2:
            return {}

        score_gaps = []
        for i in range(len(rankings) - 1):
            gap = rankings[i].composite_score - rankings[i + 1].composite_score
            score_gaps.append(gap)

        return {
            "close_competition": any(gap < 5 for gap in score_gaps),
            "dominant_leader": score_gaps[0] > 15 if score_gaps else False,
            "avg_score_gap": np.mean(score_gaps) if score_gaps else 0,
            "max_score_gap": max(score_gaps) if score_gaps else 0,
        }

    def _get_methodology_description(self) -> Dict[str, Any]:
        """Get evaluation methodology description"""
        return {
            "evaluation_dimensions": {
                "return": "Return performance - Based on annualized return and total return",
                "risk": "Risk control - Based on volatility, max drawdown and VaR",
                "stability": "Stability - Based on stability ratio, win rate and profit/loss ratio",
                "efficiency": "Efficiency metrics - Based on Sharpe ratio, Sortino ratio and Calmar ratio",
                "robustness": "Robustness - Based on distribution characteristics and recovery ability",
            },
            "dimension_weights": self.evaluation_weights,
            "scoring_method": "Dimension scores normalized to 0-100 points, then weighted composite score calculated",
            "ranking_basis": "Ranked by composite score from high to low",
        }
