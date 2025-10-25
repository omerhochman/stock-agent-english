import argparse
import sys
from datetime import datetime, timedelta
from typing import List, Optional

from src.backtest.core import Backtester, BacktestConfig
from src.main import run_hedge_fund
from src.utils.logging_config import setup_logger

def main():
    """Main function"""
    # Setup logging
    logger = setup_logger('MainBacktester')
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create backtest configuration
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
    
    # Process stock codes
    tickers = None
    if args.tickers:
        tickers = [ticker.strip() for ticker in args.tickers.split(',')]
    
    logger.info("=" * 60)
    logger.info("A-share Investment Agent System - Backtest Analysis")
    logger.info("=" * 60)
    logger.info(f"Primary stock: {args.ticker}")
    if tickers and len(tickers) > 1:
        logger.info(f"Multi-asset portfolio: {', '.join(tickers)}")
    logger.info(f"Backtest period: {args.start_date} to {args.end_date}")
    logger.info(f"Initial capital: {args.initial_capital:,.2f}")
    logger.info(f"Benchmark index: {args.benchmark}")
    
    try:
        # Create backtester
        backtester = Backtester(
            agent_function=run_hedge_fund,
            ticker=args.ticker,
            tickers=tickers,
            config=config
        )
        
        # 1. Initialize baseline strategies
        logger.info("\n" + "=" * 40)
        logger.info("Initializing Baseline Strategies")
        logger.info("=" * 40)
        baseline_strategies = backtester.initialize_baseline_strategies()
        
        for strategy in baseline_strategies:
            logger.info(f"✓ {strategy.name} - {strategy.strategy_type}")
        
        # 2. Run AI agent backtest
        if not args.baseline_only:
            logger.info("\n" + "=" * 40)
            logger.info("Running AI Agent Backtest")
            logger.info("=" * 40)
            agent_result = backtester.run_agent_backtest()
            logger.info(f"✓ AI Agent backtest completed - Total return: {agent_result.performance_metrics.get('total_return', 0)*100:.2f}%")
        
        # 3. Run baseline backtests
        logger.info("\n" + "=" * 40)
        logger.info("Running Baseline Strategy Backtests")
        logger.info("=" * 40)
        baseline_results = backtester.run_baseline_backtests()
        
        for name, result in baseline_results.items():
            total_return = result.performance_metrics.get('total_return', 0) * 100
            sharpe_ratio = result.performance_metrics.get('sharpe_ratio', 0)
            max_drawdown = result.performance_metrics.get('max_drawdown', 0) * 100
            logger.info(f"✓ {name} - Return: {total_return:.2f}%, Sharpe: {sharpe_ratio:.3f}, Drawdown: {max_drawdown:.2f}%")
        
        # 4. Run comprehensive comparison analysis
        logger.info("\n" + "=" * 40)
        logger.info("Statistical Significance Testing and Comparison Analysis")
        logger.info("=" * 40)
        comparison_results = backtester.run_comprehensive_comparison()
        
        # Display statistical test results
        _display_significance_results(comparison_results, logger)
        
        # Display strategy rankings
        _display_strategy_ranking(comparison_results, logger)
        
        # 5. Generate visualization charts
        logger.info("\n" + "=" * 40)
        logger.info("Generating Visualization Charts")
        logger.info("=" * 40)
        chart_paths = backtester.generate_visualization()
        
        for chart_type, path in chart_paths.items():
            logger.info(f"✓ {chart_type}: {path}")
        
        # 6. Save results
        if args.save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f"backtest_results_{args.ticker}_{timestamp}.pkl"
            backtester.save_results(results_file)
            logger.info(f"✓ Results saved: {results_file}")
        
        # 7. Generate report
        if args.export_report:
            report_file = backtester.export_report(format='html')
            logger.info(f"✓ Report generated: {report_file}")
        
        # 8. Display final summary
        logger.info("\n" + "=" * 60)
        logger.info("Backtest Analysis Completed")
        logger.info("=" * 60)
        
        # Display best strategy
        if comparison_results and 'strategy_ranking' in comparison_results:
            ranking = comparison_results['strategy_ranking']
            if 'by_sharpe' in ranking and ranking['by_sharpe']:
                best_strategy = ranking['by_sharpe'][0]
                logger.info(f"🏆 Best Strategy: {best_strategy['strategy']}")
                logger.info(f"   Sharpe Ratio: {best_strategy['sharpe_ratio']:.3f}")
                logger.info(f"   Total Return: {best_strategy['total_return']*100:.2f}%")
        
        logger.info("Thank you for using the A-share Investment Agent System!")
        
    except KeyboardInterrupt:
        logger.info("\nUser interrupted the backtest process")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error occurred during backtest: {str(e)}")
        logger.error("Please check configuration parameters and data availability")
        
        # Add more detailed error information
        import traceback
        logger.error("Detailed error information:")
        logger.error(traceback.format_exc())
        
        # Provide suggestions for common issues
        error_msg = str(e).lower()
        if "broadcast" in error_msg:
            logger.error("Suggestion: This might be a data dimension mismatch issue, please check data quality and length")
        elif "feature_names_in_" in error_msg:
            logger.error("Suggestion: This might be a model feature name issue, please check model training status")
        elif "insufficient data" in error_msg:
            logger.error("Suggestion: Please increase backtest time range or check data availability")
        
        sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="A-share Investment Agent System - Backtest Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python src/main_backtester.py --ticker 000001 --start-date 2023-01-01 --end-date 2023-12-31
  python src/main_backtester.py --ticker 000001 --tickers "000001,000002,600036" --initial-capital 500000
  python src/main_backtester.py --ticker 000001 --baseline-only --save-results --export-report
        """
    )
    
    # Required parameters
    parser.add_argument(
        '--ticker', 
        type=str, 
        required=True,
        help='Primary stock code (e.g.: 000001)'
    )
    
    # Optional parameters
    parser.add_argument(
        '--tickers',
        type=str,
        help='Multiple stock codes, separated by commas (e.g.: "000001,000002,600036")'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        help='Backtest start date (format: YYYY-MM-DD, default: one year ago)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='Backtest end date (format: YYYY-MM-DD, default: today)'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='Initial capital (default: 100000)'
    )
    
    parser.add_argument(
        '--benchmark',
        type=str,
        default='000300',
        help='Benchmark index code (default: 000300 CSI 300)'
    )
    
    parser.add_argument(
        '--trading-cost',
        type=float,
        default=0.001,
        help='Trading cost ratio (default: 0.001 i.e. 0.1%%)'
    )
    
    parser.add_argument(
        '--slippage',
        type=float,
        default=0.001,
        help='Slippage ratio (default: 0.001 i.e. 0.1%%)'
    )
    
    parser.add_argument(
        '--num-of-news',
        type=int,
        default=5,
        help='Daily news count (default: 5)'
    )
    
    parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.05,
        help='Confidence level for statistical significance testing (default: 0.05)'
    )
    
    # Run mode
    parser.add_argument(
        '--baseline-only',
        action='store_true',
        help='Only run baseline strategies, do not run AI Agent'
    )
    
    # Output options
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save backtest results to file'
    )
    
    parser.add_argument(
        '--export-report',
        action='store_true',
        help='Export detailed report in HTML format'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed log information'
    )
    
    return parser.parse_args()

def _display_significance_results(comparison_results: dict, logger):
    """Display statistical significance test results"""
    if 'pairwise_comparisons' not in comparison_results:
        return
        
    comparisons = comparison_results['pairwise_comparisons']
    logger.info("\n📊 Statistical Significance Test Results:")
    
    # Display pairwise comparison results
    significant_count = 0
    total_count = len(comparisons)
    
    for comparison_key, result in comparisons.items():
        try:
            if 'summary' in result:
                summary = result['summary']
                power = summary.get('statistical_power', 0)
                conclusion = summary.get('overall_conclusion', 'No conclusion')
                significance = "Significant" if power > 0.5 else "Not significant"
                logger.info(f"  {comparison_key}: {conclusion} (Power: {power:.3f}, {significance})")
                if power > 0.5:
                    significant_count += 1
        except (KeyError, TypeError):
            logger.info(f"  {comparison_key}: Data format error")
            continue
    
    # Display statistical summary
    if total_count > 0:
        logger.info(f"\n  Total: {significant_count}/{total_count} comparisons show significant differences")

def _display_strategy_ranking(comparison_results: dict, logger):
    """Display strategy rankings"""
    if 'strategy_ranking' not in comparison_results:
        return
        
    ranking = comparison_results['strategy_ranking']
    
    # Rank by Sharpe ratio
    if 'by_sharpe' in ranking and ranking['by_sharpe']:
        logger.info("\n🏆 Strategy Rankings (by Sharpe Ratio):")
        for i, strategy in enumerate(ranking['by_sharpe'][:5], 1):
            logger.info(f"  {i}. {strategy['strategy']}: {strategy['sharpe_ratio']:.3f}")
    
    # Rank by total return
    if 'by_return' in ranking and ranking['by_return']:
        logger.info("\n💰 Strategy Rankings (by Total Return):")
        for i, strategy in enumerate(ranking['by_return'][:5], 1):
            logger.info(f"  {i}. {strategy['strategy']}: {strategy['total_return']*100:.2f}%")

if __name__ == "__main__":
    main()