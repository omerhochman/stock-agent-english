"""
回测结果表格生成器
提供多种格式的对比表格生成和导出功能
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BacktestTableGenerator:
    """回测结果表格生成器"""
    
    def __init__(self, output_dir: str = "backtest_reports"):
        """
        初始化表格生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def generate_comprehensive_report(self, 
                                    results_summary: Dict[str, Dict[str, Any]],
                                    comparison_results: Optional[Dict[str, Any]] = None,
                                    config: Optional[Dict[str, Any]] = None,
                                    export_formats: List[str] = ['csv', 'excel', 'html']) -> Dict[str, str]:
        """
        生成综合回测报告
        
        Args:
            results_summary: 策略结果摘要
            comparison_results: 比较分析结果
            config: 回测配置信息
            export_formats: 导出格式列表
            
        Returns:
            Dict: 生成的文件路径
        """
        if not results_summary:
            logger.warning("没有结果数据，无法生成报告")
            return {}
        
        generated_files = {}
        
        try:
            # 1. 生成性能对比表格
            performance_df = self._create_performance_table(results_summary)
            
            # 2. 生成策略排名表格
            ranking_df = self._create_ranking_table(results_summary)
            
            # 3. 生成风险指标表格
            risk_df = self._create_risk_metrics_table(results_summary)
            
            # 4. 生成交易统计表格
            trading_df = self._create_trading_statistics_table(results_summary)
            
            # 5. 生成AI Agent专项分析表格
            ai_analysis_df = None
            if 'AI_Agent' in results_summary:
                ai_analysis_df = self._create_ai_agent_analysis_table(results_summary)
            
            # 6. 生成统计显著性表格
            significance_df = None
            if comparison_results:
                significance_df = self._create_statistical_significance_table(comparison_results)
            
            # 导出不同格式
            for format_type in export_formats:
                if format_type.lower() == 'csv':
                    csv_files = self._export_csv_tables(
                        performance_df, ranking_df, risk_df, trading_df, 
                        ai_analysis_df, significance_df
                    )
                    generated_files.update(csv_files)
                
                elif format_type.lower() == 'excel':
                    excel_file = self._export_excel_workbook(
                        performance_df, ranking_df, risk_df, trading_df,
                        ai_analysis_df, significance_df, config
                    )
                    if excel_file:
                        generated_files['excel_report'] = excel_file
                
                elif format_type.lower() == 'html':
                    html_file = self._export_html_report(
                        performance_df, ranking_df, risk_df, trading_df,
                        ai_analysis_df, significance_df, config
                    )
                    if html_file:
                        generated_files['html_report'] = html_file
            
            logger.info(f"成功生成 {len(generated_files)} 个报告文件")
            return generated_files
            
        except Exception as e:
            logger.error(f"生成综合报告失败: {e}")
            return {}
    
    def _create_performance_table(self, results_summary: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """创建性能对比表格"""
        table_data = []
        
        for strategy_name, metrics in results_summary.items():
            row = {
                '策略名称': strategy_name,
                '总收益率(%)': round(metrics.get('total_return', 0), 2),
                '年化收益率(%)': round(metrics.get('annual_return', 0), 2),
                '夏普比率': round(metrics.get('sharpe_ratio', 0), 3),
                '索提诺比率': round(metrics.get('sortino_ratio', 0), 3),
                '卡玛比率': round(metrics.get('calmar_ratio', 0), 3),
                '信息比率': round(metrics.get('information_ratio', 0), 3),
                '最大回撤(%)': round(abs(metrics.get('max_drawdown', 0)), 2),
                '年化波动率(%)': round(metrics.get('volatility', 0), 2),
                'VaR_95(%)': round(abs(metrics.get('var_95', 0)) * 100, 2),
                'CVaR_95(%)': round(abs(metrics.get('cvar_95', 0)) * 100, 2),
                '胜率(%)': round(metrics.get('win_rate', 0) * 100, 2),
                '盈亏比': round(metrics.get('profit_loss_ratio', 0), 2),
                '交易次数': metrics.get('trade_count', 0),
                '平均持仓天数': round(metrics.get('avg_holding_period', 0), 1),
                '换手率(%)': round(metrics.get('turnover_rate', 0) * 100, 2)
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        df = df.sort_values('总收益率(%)', ascending=False).reset_index(drop=True)
        df.insert(0, '排名', range(1, len(df) + 1))
        
        return df
    
    def _create_ranking_table(self, results_summary: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """创建策略排名表格"""
        ranking_data = []
        
        # 定义评估维度权重
        weights = {
            '收益表现': 0.30,
            '风险控制': 0.25,
            '风险调整收益': 0.25,
            '交易效率': 0.20
        }
        
        for strategy_name, metrics in results_summary.items():
            # 计算各维度得分
            scores = self._calculate_dimension_scores(metrics)
            
            # 计算综合得分
            composite_score = sum(scores[dim] * weights[dim] for dim in scores)
            
            ranking_data.append({
                '策略名称': strategy_name,
                '收益表现': round(scores['收益表现'], 1),
                '风险控制': round(scores['风险控制'], 1),
                '风险调整收益': round(scores['风险调整收益'], 1),
                '交易效率': round(scores['交易效率'], 1),
                '综合得分': round(composite_score, 1)
            })
        
        df = pd.DataFrame(ranking_data)
        df = df.sort_values('综合得分', ascending=False).reset_index(drop=True)
        df.insert(0, '排名', range(1, len(df) + 1))
        
        return df
    
    def _create_risk_metrics_table(self, results_summary: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """创建风险指标表格"""
        risk_data = []
        
        for strategy_name, metrics in results_summary.items():
            row = {
                '策略名称': strategy_name,
                '最大回撤(%)': round(abs(metrics.get('max_drawdown', 0)), 2),
                '回撤持续天数': metrics.get('max_drawdown_duration', 0),
                '年化波动率(%)': round(metrics.get('volatility', 0), 2),
                '下行波动率(%)': round(metrics.get('downside_volatility', 0), 2),
                'VaR_95(%)': round(abs(metrics.get('var_95', 0)) * 100, 2),
                'CVaR_95(%)': round(abs(metrics.get('cvar_95', 0)) * 100, 2),
                '偏度': round(metrics.get('skewness', 0), 3),
                '峰度': round(metrics.get('kurtosis', 0), 3),
                '贝塔系数': round(metrics.get('beta', 0), 3),
                '跟踪误差(%)': round(metrics.get('tracking_error', 0) * 100, 2),
                '信息比率': round(metrics.get('information_ratio', 0), 3)
            }
            risk_data.append(row)
        
        df = pd.DataFrame(risk_data)
        # 按最大回撤排序（越小越好）
        df = df.sort_values('最大回撤(%)', ascending=True).reset_index(drop=True)
        df.insert(0, '风险排名', range(1, len(df) + 1))
        
        return df
    
    def _create_trading_statistics_table(self, results_summary: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """创建交易统计表格"""
        trading_data = []
        
        for strategy_name, metrics in results_summary.items():
            row = {
                '策略名称': strategy_name,
                '总交易次数': metrics.get('trade_count', 0),
                '盈利交易次数': metrics.get('profitable_trades', 0),
                '亏损交易次数': metrics.get('losing_trades', 0),
                '胜率(%)': round(metrics.get('win_rate', 0) * 100, 2),
                '平均盈利(%)': round(metrics.get('avg_profit', 0) * 100, 2),
                '平均亏损(%)': round(metrics.get('avg_loss', 0) * 100, 2),
                '盈亏比': round(metrics.get('profit_loss_ratio', 0), 2),
                '最大单笔盈利(%)': round(metrics.get('max_profit', 0) * 100, 2),
                '最大单笔亏损(%)': round(metrics.get('max_loss', 0) * 100, 2),
                '平均持仓天数': round(metrics.get('avg_holding_period', 0), 1),
                '换手率(%)': round(metrics.get('turnover_rate', 0) * 100, 2),
                '交易成本(%)': round(metrics.get('total_costs', 0) * 100, 2)
            }
            trading_data.append(row)
        
        df = pd.DataFrame(trading_data)
        # 按胜率排序
        df = df.sort_values('胜率(%)', ascending=False).reset_index(drop=True)
        df.insert(0, '交易排名', range(1, len(df) + 1))
        
        return df
    
    def _create_ai_agent_analysis_table(self, results_summary: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """创建AI Agent专项分析表格"""
        if 'AI_Agent' not in results_summary:
            return pd.DataFrame()
        
        ai_metrics = results_summary['AI_Agent']
        other_strategies = {k: v for k, v in results_summary.items() if k != 'AI_Agent'}
        
        if not other_strategies:
            return pd.DataFrame()
        
        analysis_data = []
        
        # 关键指标对比
        key_metrics = [
            ('总收益率(%)', 'total_return', False),
            ('年化收益率(%)', 'annual_return', False),
            ('夏普比率', 'sharpe_ratio', False),
            ('最大回撤(%)', 'max_drawdown', True),
            ('年化波动率(%)', 'volatility', True),
            ('胜率(%)', 'win_rate', False),
            ('盈亏比', 'profit_loss_ratio', False)
        ]
        
        for metric_name, metric_key, lower_is_better in key_metrics:
            ai_value = ai_metrics.get(metric_key, 0)
            if metric_name in ['总收益率(%)', '年化收益率(%)', '最大回撤(%)', '年化波动率(%)']:
                ai_value = ai_value if metric_key != 'max_drawdown' else abs(ai_value)
            elif metric_name == '胜率(%)':
                ai_value = ai_value * 100
            
            # 计算基准统计
            other_values = [metrics.get(metric_key, 0) for metrics in other_strategies.values()]
            if metric_name in ['总收益率(%)', '年化收益率(%)', '最大回撤(%)', '年化波动率(%)']:
                other_values = [v if metric_key != 'max_drawdown' else abs(v) for v in other_values]
            elif metric_name == '胜率(%)':
                other_values = [v * 100 for v in other_values]
            
            avg_value = np.mean(other_values) if other_values else 0
            median_value = np.median(other_values) if other_values else 0
            best_value = min(other_values) if lower_is_better and other_values else max(other_values) if other_values else 0
            
            # 计算排名
            all_values = [ai_value] + other_values
            if lower_is_better:
                rank = sorted(all_values).index(ai_value) + 1
            else:
                rank = len(all_values) - sorted(all_values).index(ai_value)
            
            # 判断表现
            if lower_is_better:
                performance = '优于平均' if ai_value < avg_value else '低于平均'
                vs_best = '优于最佳' if ai_value < best_value else '低于最佳'
            else:
                performance = '优于平均' if ai_value > avg_value else '低于平均'
                vs_best = '优于最佳' if ai_value > best_value else '低于最佳'
            
            analysis_data.append({
                '指标': metric_name,
                'AI Agent': round(ai_value, 3),
                '基准平均': round(avg_value, 3),
                '基准中位数': round(median_value, 3),
                '基准最佳': round(best_value, 3),
                '排名': f"{rank}/{len(results_summary)}",
                '相对平均': performance,
                '相对最佳': vs_best,
                '百分位数': round((len(all_values) - rank + 1) / len(all_values) * 100, 1)
            })
        
        return pd.DataFrame(analysis_data)
    
    def _create_statistical_significance_table(self, comparison_results: Dict[str, Any]) -> pd.DataFrame:
        """创建统计显著性检验表格"""
        if 'pairwise_comparisons' not in comparison_results:
            return pd.DataFrame()
        
        comparisons = comparison_results['pairwise_comparisons']
        significance_data = []
        
        for comparison_key, result in comparisons.items():
            try:
                if 'summary' not in result:
                    continue
                
                summary = result['summary']
                strategies = comparison_key.split(' vs ')
                strategy1 = strategies[0] if len(strategies) > 0 else "未知"
                strategy2 = strategies[1] if len(strategies) > 1 else "未知"
                
                # 提取检验结果
                paired_test = result.get('paired_test', {})
                dm_test = result.get('diebold_mariano', {})
                sharpe_test = result.get('sharpe_test', {})
                
                significance_data.append({
                    '策略对比': comparison_key,
                    '策略A': strategy1,
                    '策略B': strategy2,
                    '配对t检验': '显著' if paired_test.get('significant', False) else '不显著',
                    't统计量': round(paired_test.get('statistic', 0), 3),
                    'p值(配对)': round(paired_test.get('p_value', 1), 4),
                    'DM检验': '显著' if dm_test.get('significant', False) else '不显著',
                    'DM统计量': round(dm_test.get('statistic', 0), 3),
                    'p值(DM)': round(dm_test.get('p_value', 1), 4),
                    '夏普比率检验': '显著' if sharpe_test.get('significant', False) else '不显著',
                    '夏普差异': round(sharpe_test.get('sharpe_diff', 0), 3),
                    'p值(夏普)': round(sharpe_test.get('p_value', 1), 4),
                    '统计功效': round(summary.get('statistical_power', 0), 3),
                    '结论': summary.get('overall_conclusion', '无结论')
                })
                
            except (KeyError, TypeError, IndexError) as e:
                logger.warning(f"处理比较结果 {comparison_key} 时出错: {e}")
                continue
        
        if not significance_data:
            return pd.DataFrame()
        
        return pd.DataFrame(significance_data)
    
    def _calculate_dimension_scores(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """计算各维度得分"""
        scores = {}
        
        # 收益表现得分 (0-100)
        total_return = metrics.get('total_return', 0)
        annual_return = metrics.get('annual_return', 0)
        scores['收益表现'] = min(100, max(0, (total_return + annual_return) * 50))
        
        # 风险控制得分 (0-100, 越小越好)
        max_dd = abs(metrics.get('max_drawdown', 0))
        volatility = metrics.get('volatility', 0)
        var_95 = abs(metrics.get('var_95', 0))
        risk_penalty = (max_dd + volatility + var_95) * 100
        scores['风险控制'] = max(0, 100 - risk_penalty)
        
        # 风险调整收益得分 (0-100)
        sharpe = metrics.get('sharpe_ratio', 0)
        sortino = metrics.get('sortino_ratio', 0)
        calmar = metrics.get('calmar_ratio', 0)
        scores['风险调整收益'] = min(100, max(0, (sharpe + sortino + calmar) * 20))
        
        # 交易效率得分 (0-100)
        win_rate = metrics.get('win_rate', 0)
        pl_ratio = metrics.get('profit_loss_ratio', 0)
        trade_count = metrics.get('trade_count', 0)
        efficiency_score = win_rate * 50 + min(50, pl_ratio * 25)
        if trade_count > 0:
            efficiency_score += min(20, trade_count / 10)
        scores['交易效率'] = min(100, efficiency_score)
        
        return scores
    
    def _export_csv_tables(self, *tables) -> Dict[str, str]:
        """导出CSV格式表格"""
        csv_files = {}
        table_names = [
            'performance_comparison',
            'strategy_ranking', 
            'risk_metrics',
            'trading_statistics',
            'ai_agent_analysis',
            'statistical_significance'
        ]
        
        for i, table in enumerate(tables):
            if table is not None and not table.empty:
                filename = f"{table_names[i]}_{self.timestamp}.csv"
                filepath = self.output_dir / filename
                
                try:
                    table.to_csv(filepath, index=False, encoding='utf-8-sig')
                    csv_files[table_names[i]] = str(filepath)
                    logger.info(f"CSV表格已保存: {filepath}")
                except Exception as e:
                    logger.error(f"保存CSV表格失败 {filename}: {e}")
        
        return csv_files
    
    def _export_excel_workbook(self, *tables, config=None) -> Optional[str]:
        """导出Excel工作簿"""
        try:
            filename = f"backtest_comprehensive_report_{self.timestamp}.xlsx"
            filepath = self.output_dir / filename
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                sheet_names = [
                    '性能对比', '策略排名', '风险指标', 
                    '交易统计', 'AI专项分析', '统计显著性'
                ]
                
                for i, (table, sheet_name) in enumerate(zip(tables, sheet_names)):
                    if table is not None and not table.empty:
                        table.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # 添加配置信息工作表
                if config:
                    config_df = pd.DataFrame(list(config.items()), columns=['配置项', '值'])
                    config_df.to_excel(writer, sheet_name='回测配置', index=False)
            
            logger.info(f"Excel报告已保存: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"保存Excel报告失败: {e}")
            return None
    
    def _export_html_report(self, *tables, config=None) -> Optional[str]:
        """导出HTML报告"""
        try:
            filename = f"backtest_report_{self.timestamp}.html"
            filepath = self.output_dir / filename
            
            html_content = self._generate_html_content(tables, config)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML报告已保存: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"保存HTML报告失败: {e}")
            return None
    
    def _generate_html_content(self, tables, config=None) -> str:
        """生成HTML报告内容"""
        html_parts = [
            """
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>回测分析报告</title>
                <style>
                    body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                    .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                    h1 { color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                    h2 { color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }
                    table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 12px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
                    th { background-color: #3498db; color: white; font-weight: bold; }
                    tr:nth-child(even) { background-color: #f2f2f2; }
                    tr:hover { background-color: #e8f4fd; }
                    .config-info { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
                    .timestamp { text-align: right; color: #7f8c8d; font-size: 12px; margin-top: 20px; }
                    .highlight { background-color: #f39c12; color: white; font-weight: bold; }
                    .positive { color: #27ae60; font-weight: bold; }
                    .negative { color: #e74c3c; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🚀 A股投资Agent回测分析报告</h1>
            """
        ]
        
        # 添加配置信息
        if config:
            html_parts.append('<div class="config-info">')
            html_parts.append('<h3>📋 回测配置信息</h3>')
            for key, value in config.items():
                html_parts.append(f'<p><strong>{key}:</strong> {value}</p>')
            html_parts.append('</div>')
        
        # 添加表格
        table_titles = [
            '📊 策略性能对比表',
            '🏆 策略综合排名表', 
            '⚠️ 风险指标对比表',
            '📈 交易统计表',
            '🤖 AI Agent专项分析',
            '📉 统计显著性检验'
        ]
        
        for i, (table, title) in enumerate(zip(tables, table_titles)):
            if table is not None and not table.empty:
                html_parts.append(f'<h2>{title}</h2>')
                
                # 为表格添加样式
                table_html = table.to_html(index=False, escape=False, classes='table')
                
                # 添加颜色编码
                if '排名' in table.columns:
                    # 为排名添加颜色
                    table_html = table_html.replace('<td>1</td>', '<td class="highlight">1</td>')
                
                html_parts.append(table_html)
        
        # 添加页脚
        html_parts.append(f'''
                    <div class="timestamp">
                        <p>报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                        <p>© A股投资Agent系统 - 回测分析报告</p>
                    </div>
                </div>
            </body>
            </html>
        ''')
        
        return ''.join(html_parts) 