import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class StrategyRanking:
    """策略排名结果"""
    name: str
    composite_score: float
    rank: int
    scores: Dict[str, float]

class StrategyComparator:
    """
    策略比较分析类
    实现多维度策略评估和排名
    """
    
    def __init__(self):
        # 评估维度权重
        self.evaluation_weights = {
            'return': 0.25,      # 收益表现
            'risk': 0.20,        # 风险控制
            'stability': 0.20,   # 稳定性
            'efficiency': 0.15,  # 效率指标
            'robustness': 0.20   # 稳健性
        }
    
    def rank_strategies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        对策略进行综合排名
        
        Args:
            results: 策略结果字典
            
        Returns:
            Dict: 排名结果
        """
        if len(results) < 2:
            return {'error': '需要至少2个策略进行比较'}
        
        # 计算各维度得分
        dimension_scores = self._calculate_dimension_scores(results)
        
        # 计算综合得分
        composite_scores = self._calculate_composite_scores(dimension_scores)
        
        # 生成排名
        rankings = self._generate_rankings(composite_scores, dimension_scores)
        
        # 分析结果
        analysis = self._analyze_rankings(rankings, dimension_scores)
        
        return {
            'rankings': rankings,
            'dimension_scores': dimension_scores,
            'analysis': analysis,
            'methodology': self._get_methodology_description()
        }
    
    def _calculate_dimension_scores(self, results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """计算各维度得分"""
        dimension_scores = {}
        
        for strategy_name, result in results.items():
            metrics = result.performance_metrics
            
            # 收益维度得分
            return_score = self._calculate_return_score(metrics)
            
            # 风险维度得分
            risk_score = self._calculate_risk_score(metrics)
            
            # 稳定性维度得分
            stability_score = self._calculate_stability_score(metrics)
            
            # 效率维度得分
            efficiency_score = self._calculate_efficiency_score(metrics)
            
            # 稳健性维度得分
            robustness_score = self._calculate_robustness_score(metrics, result)
            
            dimension_scores[strategy_name] = {
                'return': return_score,
                'risk': risk_score,
                'stability': stability_score,
                'efficiency': efficiency_score,
                'robustness': robustness_score
            }
        
        # 标准化得分（0-100分）
        normalized_scores = self._normalize_scores(dimension_scores)
        
        return normalized_scores
    
    def _calculate_return_score(self, metrics: Dict[str, Any]) -> float:
        """计算收益得分"""
        annual_return = metrics.get('annual_return', 0)
        total_return = metrics.get('total_return', 0)
        
        # 基于年化收益率的得分
        return_score = min(100, max(0, annual_return * 500 + 50))  # 10%年化收益=100分
        
        # 总收益率调整
        if total_return > 0:
            return_score = min(100, return_score * (1 + total_return * 0.1))
        
        return return_score
    
    def _calculate_risk_score(self, metrics: Dict[str, Any]) -> float:
        """计算风险得分（风险越低得分越高）"""
        volatility = metrics.get('volatility', 0)
        max_drawdown = abs(metrics.get('max_drawdown', 0))
        var_95 = abs(metrics.get('var_95', 0))
        
        # 波动率得分（年化波动率15%以下为满分）
        vol_score = max(0, 100 - volatility * 500)
        
        # 最大回撤得分（5%以下为满分）
        dd_score = max(0, 100 - max_drawdown * 2000)
        
        # VaR得分
        var_score = max(0, 100 - var_95 * 5000)
        
        # 综合风险得分
        risk_score = (vol_score * 0.4 + dd_score * 0.4 + var_score * 0.2)
        
        return risk_score
    
    def _calculate_stability_score(self, metrics: Dict[str, Any]) -> float:
        """计算稳定性得分"""
        stability_ratio = metrics.get('stability', 0)
        win_rate = metrics.get('win_rate', 0)
        profit_loss_ratio = metrics.get('profit_loss_ratio', 0)
        
        # 稳定性比率得分
        stability_score = stability_ratio * 100
        
        # 胜率得分
        win_rate_score = win_rate * 100
        
        # 盈亏比得分
        pl_score = min(100, profit_loss_ratio * 50)
        
        # 综合稳定性得分
        total_score = (stability_score * 0.4 + win_rate_score * 0.3 + pl_score * 0.3)
        
        return total_score
    
    def _calculate_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """计算效率得分"""
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        sortino_ratio = metrics.get('sortino_ratio', 0)
        calmar_ratio = metrics.get('calmar_ratio', 0)
        
        # 夏普比率得分（2.0以上为满分）
        sharpe_score = min(100, max(0, sharpe_ratio * 50))
        
        # 索提诺比率得分
        sortino_score = min(100, max(0, sortino_ratio * 40))
        
        # 卡玛比率得分
        calmar_score = min(100, max(0, calmar_ratio * 20))
        
        # 综合效率得分
        efficiency_score = (sharpe_score * 0.5 + sortino_score * 0.3 + calmar_score * 0.2)
        
        return efficiency_score
    
    def _calculate_robustness_score(self, metrics: Dict[str, Any], result: Any) -> float:
        """计算稳健性得分"""
        skewness = metrics.get('skewness', 0)
        kurtosis = metrics.get('kurtosis', 0)
        tail_ratio = metrics.get('tail_ratio', 1)
        recovery_factor = metrics.get('recovery_factor', 0)
        
        # 偏度得分（接近0为最佳）
        skew_score = max(0, 100 - abs(skewness) * 50)
        
        # 峰度得分（接近正态分布为最佳）
        kurt_score = max(0, 100 - abs(kurtosis) * 20)
        
        # 尾部比率得分
        tail_score = max(0, 100 - abs(tail_ratio - 1) * 100)
        
        # 恢复因子得分
        recovery_score = min(100, recovery_factor * 10)
        
        # 综合稳健性得分
        robustness_score = (skew_score * 0.25 + kurt_score * 0.25 + 
                          tail_score * 0.25 + recovery_score * 0.25)
        
        return robustness_score
    
    def _normalize_scores(self, dimension_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """标准化得分"""
        normalized = {}
        
        # 获取所有维度
        dimensions = list(next(iter(dimension_scores.values())).keys())
        
        for dim in dimensions:
            # 获取该维度所有策略的得分
            dim_scores = [scores[dim] for scores in dimension_scores.values()]
            
            if len(set(dim_scores)) == 1:  # 所有得分相同
                for strategy in dimension_scores:
                    if strategy not in normalized:
                        normalized[strategy] = {}
                    normalized[strategy][dim] = 50.0  # 给予中等分数
            else:
                # Min-Max标准化到0-100范围
                min_score = min(dim_scores)
                max_score = max(dim_scores)
                
                for strategy, scores in dimension_scores.items():
                    if strategy not in normalized:
                        normalized[strategy] = {}
                    
                    if max_score != min_score:
                        normalized_score = (scores[dim] - min_score) / (max_score - min_score) * 100
                    else:
                        normalized_score = 50.0
                    
                    normalized[strategy][dim] = normalized_score
        
        return normalized
    
    def _calculate_composite_scores(self, dimension_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """计算综合得分"""
        composite_scores = {}
        
        for strategy, scores in dimension_scores.items():
            composite_score = sum(
                scores[dim] * self.evaluation_weights[dim] 
                for dim in scores
            )
            composite_scores[strategy] = composite_score
        
        return composite_scores
    
    def _generate_rankings(self, composite_scores: Dict[str, float], 
                         dimension_scores: Dict[str, Dict[str, float]]) -> List[StrategyRanking]:
        """生成策略排名"""
        # 按综合得分排序
        sorted_strategies = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        
        rankings = []
        for rank, (strategy_name, composite_score) in enumerate(sorted_strategies, 1):
            ranking = StrategyRanking(
                name=strategy_name,
                composite_score=composite_score,
                rank=rank,
                scores=dimension_scores[strategy_name]
            )
            rankings.append(ranking)
        
        return rankings
    
    def _analyze_rankings(self, rankings: List[StrategyRanking], 
                        dimension_scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """分析排名结果"""
        if not rankings:
            return {}
        
        top_strategy = rankings[0]
        
        # 找出各维度最佳策略
        dimension_leaders = {}
        for dim in self.evaluation_weights.keys():
            best_strategy = max(dimension_scores.items(), key=lambda x: x[1][dim])
            dimension_leaders[dim] = {
                'strategy': best_strategy[0],
                'score': best_strategy[1][dim]
            }
        
        # 分析得分分布
        all_scores = [r.composite_score for r in rankings]
        score_analysis = {
            'mean': np.mean(all_scores),
            'std': np.std(all_scores),
            'range': max(all_scores) - min(all_scores),
            'cv': np.std(all_scores) / np.mean(all_scores) if np.mean(all_scores) != 0 else 0
        }
        
        return {
            'top_performer': {
                'name': top_strategy.name,
                'score': top_strategy.composite_score,
                'strengths': self._identify_strengths(top_strategy.scores),
                'weaknesses': self._identify_weaknesses(top_strategy.scores)
            },
            'dimension_leaders': dimension_leaders,
            'score_distribution': score_analysis,
            'competitive_landscape': self._analyze_competition(rankings)
        }
    
    def _identify_strengths(self, scores: Dict[str, float]) -> List[str]:
        """识别策略优势"""
        strengths = []
        for dim, score in scores.items():
            if score >= 75:  # 高分阈值
                strengths.append(f"{dim}表现优秀 ({score:.1f}分)")
        return strengths
    
    def _identify_weaknesses(self, scores: Dict[str, float]) -> List[str]:
        """识别策略劣势"""
        weaknesses = []
        for dim, score in scores.items():
            if score <= 30:  # 低分阈值
                weaknesses.append(f"{dim}有待改进 ({score:.1f}分)")
        return weaknesses
    
    def _analyze_competition(self, rankings: List[StrategyRanking]) -> Dict[str, Any]:
        """分析竞争态势"""
        if len(rankings) < 2:
            return {}
        
        score_gaps = []
        for i in range(len(rankings) - 1):
            gap = rankings[i].composite_score - rankings[i + 1].composite_score
            score_gaps.append(gap)
        
        return {
            'close_competition': any(gap < 5 for gap in score_gaps),
            'dominant_leader': score_gaps[0] > 15 if score_gaps else False,
            'avg_score_gap': np.mean(score_gaps) if score_gaps else 0,
            'max_score_gap': max(score_gaps) if score_gaps else 0
        }
    
    def _get_methodology_description(self) -> Dict[str, Any]:
        """获取评估方法说明"""
        return {
            'evaluation_dimensions': {
                'return': '收益表现 - 基于年化收益率和总收益率',
                'risk': '风险控制 - 基于波动率、最大回撤和VaR',
                'stability': '稳定性 - 基于稳定性比率、胜率和盈亏比',
                'efficiency': '效率指标 - 基于夏普比率、索提诺比率和卡玛比率',
                'robustness': '稳健性 - 基于分布特征和恢复能力'
            },
            'dimension_weights': self.evaluation_weights,
            'scoring_method': '各维度得分标准化至0-100分，然后按权重计算综合得分',
            'ranking_basis': '按综合得分从高到低排序'
        }
