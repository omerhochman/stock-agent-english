import argparse
import sys
from datetime import datetime, timedelta
from typing import List, Optional

from src.backtest.core import Backtester, BacktestConfig
from src.main import run_hedge_fund
from src.utils.logging_config import setup_logger

def main():
    """主函数"""
    # 设置日志
    logger = setup_logger('MainBacktester')
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建回测配置
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark_ticker=args.benchmark,
        trading_cost=args.trading_cost,
        slippage=args.slippage,
        num_of_news=args.num_of_news,
        confidence_level=args.confidence_level
    )
    
    # 处理股票代码
    tickers = None
    if args.tickers:
        tickers = [ticker.strip() for ticker in args.tickers.split(',')]
    
    logger.info("=" * 60)
    logger.info("A股投资Agent系统 - 回测分析")
    logger.info("=" * 60)
    logger.info(f"主要股票: {args.ticker}")
    if tickers and len(tickers) > 1:
        logger.info(f"多资产组合: {', '.join(tickers)}")
    logger.info(f"回测期间: {args.start_date} 至 {args.end_date}")
    logger.info(f"初始资金: {args.initial_capital:,.2f}")
    logger.info(f"基准指数: {args.benchmark}")
    
    try:
        # 创建回测器
        backtester = Backtester(
            agent_function=run_hedge_fund,
            ticker=args.ticker,
            tickers=tickers,
            config=config
        )
        
        # 1. 初始化baseline策略
        logger.info("\n" + "=" * 40)
        logger.info("初始化Baseline策略")
        logger.info("=" * 40)
        baseline_strategies = backtester.initialize_baseline_strategies()
        
        for strategy in baseline_strategies:
            logger.info(f"✓ {strategy.name} - {strategy.strategy_type}")
        
        # 2. 运行智能体回测
        if not args.baseline_only:
            logger.info("\n" + "=" * 40)
            logger.info("运行AI Agent回测")
            logger.info("=" * 40)
            agent_result = backtester.run_agent_backtest()
            logger.info(f"✓ AI Agent回测完成 - 总收益率: {agent_result.performance_metrics.get('total_return', 0)*100:.2f}%")
        
        # 3. 运行baseline回测
        logger.info("\n" + "=" * 40)
        logger.info("运行Baseline策略回测")
        logger.info("=" * 40)
        baseline_results = backtester.run_baseline_backtests()
        
        for name, result in baseline_results.items():
            total_return = result.performance_metrics.get('total_return', 0) * 100
            sharpe_ratio = result.performance_metrics.get('sharpe_ratio', 0)
            max_drawdown = result.performance_metrics.get('max_drawdown', 0) * 100
            logger.info(f"✓ {name} - 收益: {total_return:.2f}%, 夏普: {sharpe_ratio:.3f}, 回撤: {max_drawdown:.2f}%")
        
        # 4. 运行综合比较分析
        logger.info("\n" + "=" * 40)
        logger.info("统计显著性检验与比较分析")
        logger.info("=" * 40)
        comparison_results = backtester.run_comprehensive_comparison()
        
        # 显示统计检验结果
        _display_significance_results(comparison_results, logger)
        
        # 显示策略排名
        _display_strategy_ranking(comparison_results, logger)
        
        # 5. 生成可视化图表
        logger.info("\n" + "=" * 40)
        logger.info("生成可视化图表")
        logger.info("=" * 40)
        chart_paths = backtester.generate_visualization()
        
        for chart_type, path in chart_paths.items():
            logger.info(f"✓ {chart_type}: {path}")
        
        # 6. 保存结果
        if args.save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f"backtest_results_{args.ticker}_{timestamp}.pkl"
            backtester.save_results(results_file)
            logger.info(f"✓ 结果已保存: {results_file}")
        
        # 7. 生成报告
        if args.export_report:
            report_file = backtester.export_report(format='html')
            logger.info(f"✓ 报告已生成: {report_file}")
        
        # 8. 显示最终总结
        logger.info("\n" + "=" * 60)
        logger.info("回测分析完成")
        logger.info("=" * 60)
        
        # 显示最佳策略
        if comparison_results and 'strategy_ranking' in comparison_results:
            ranking = comparison_results['strategy_ranking']
            if 'by_sharpe' in ranking and ranking['by_sharpe']:
                best_strategy = ranking['by_sharpe'][0]
                logger.info(f"🏆 最佳策略: {best_strategy['strategy']}")
                logger.info(f"   夏普比率: {best_strategy['sharpe_ratio']:.3f}")
                logger.info(f"   总收益率: {best_strategy['total_return']*100:.2f}%")
        
        logger.info("感谢使用A股投资Agent系统！")
        
    except KeyboardInterrupt:
        logger.info("\n用户中断了回测过程")
        sys.exit(0)
    except Exception as e:
        logger.error(f"回测过程中发生错误: {str(e)}")
        logger.error("请检查配置参数和数据可用性")
        sys.exit(1)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="A股投资Agent系统 - 回测分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python src/main_backtester.py --ticker 000001 --start-date 2023-01-01 --end-date 2023-12-31
  python src/main_backtester.py --ticker 000001 --tickers "000001,000002,600036" --initial-capital 500000
  python src/main_backtester.py --ticker 000001 --baseline-only --save-results --export-report
        """
    )
    
    # 必需参数
    parser.add_argument(
        '--ticker', 
        type=str, 
        required=True,
        help='主要股票代码 (例如: 000001)'
    )
    
    # 可选参数
    parser.add_argument(
        '--tickers',
        type=str,
        help='多个股票代码，用逗号分隔 (例如: "000001,000002,600036")'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        help='回测开始日期 (格式: YYYY-MM-DD, 默认: 一年前)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='回测结束日期 (格式: YYYY-MM-DD, 默认: 今天)'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='初始资金 (默认: 100000)'
    )
    
    parser.add_argument(
        '--benchmark',
        type=str,
        default='000300',
        help='基准指数代码 (默认: 000300 沪深300)'
    )
    
    parser.add_argument(
        '--trading-cost',
        type=float,
        default=0.001,
        help='交易成本比例 (默认: 0.001 即0.1%%)'
    )
    
    parser.add_argument(
        '--slippage',
        type=float,
        default=0.001,
        help='滑点比例 (默认: 0.001 即0.1%%)'
    )
    
    parser.add_argument(
        '--num-of-news',
        type=int,
        default=5,
        help='每日新闻数量 (默认: 5)'
    )
    
    parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.05,
        help='统计显著性检验的置信水平 (默认: 0.05)'
    )
    
    # 运行模式
    parser.add_argument(
        '--baseline-only',
        action='store_true',
        help='仅运行baseline策略，不运行AI Agent'
    )
    
    # 输出选项
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='保存回测结果到文件'
    )
    
    parser.add_argument(
        '--export-report',
        action='store_true',
        help='导出HTML格式的详细报告'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细日志信息'
    )
    
    return parser.parse_args()

def _display_significance_results(comparison_results: dict, logger):
    """显示统计显著性检验结果"""
    if 'pairwise_comparisons' not in comparison_results:
        return
        
    comparisons = comparison_results['pairwise_comparisons']
    logger.info("\n📊 统计显著性检验结果:")
    
    # 显示两两比较结果
    significant_count = 0
    total_count = len(comparisons)
    
    for comparison_key, result in comparisons.items():
        try:
            if 'summary' in result:
                summary = result['summary']
                power = summary.get('statistical_power', 0)
                conclusion = summary.get('overall_conclusion', '无结论')
                significance = "显著" if power > 0.5 else "不显著"
                logger.info(f"  {comparison_key}: {conclusion} (功效: {power:.3f}, {significance})")
                if power > 0.5:
                    significant_count += 1
        except (KeyError, TypeError):
            logger.info(f"  {comparison_key}: 数据格式错误")
            continue
    
    # 显示统计摘要
    if total_count > 0:
        logger.info(f"\n  总计: {significant_count}/{total_count} 个比较显示显著差异")

def _display_strategy_ranking(comparison_results: dict, logger):
    """显示策略排名"""
    if 'strategy_ranking' not in comparison_results:
        return
        
    ranking = comparison_results['strategy_ranking']
    
    # 按夏普比率排名
    if 'by_sharpe' in ranking and ranking['by_sharpe']:
        logger.info("\n🏆 策略排名 (按夏普比率):")
        for i, strategy in enumerate(ranking['by_sharpe'][:5], 1):
            logger.info(f"  {i}. {strategy['strategy']}: {strategy['sharpe_ratio']:.3f}")
    
    # 按总收益率排名
    if 'by_return' in ranking and ranking['by_return']:
        logger.info("\n💰 策略排名 (按总收益率):")
        for i, strategy in enumerate(ranking['by_return'][:5], 1):
            logger.info(f"  {i}. {strategy['strategy']}: {strategy['total_return']*100:.2f}%")

if __name__ == "__main__":
    main()