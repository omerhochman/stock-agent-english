import os
from colorama import Fore, Style, init

# 初始化colorama，确保ANSI颜色代码在各平台正常工作
init(autoreset=True)

class TradingDisplay:
    """交易决策美观显示类"""
    
    def __init__(self):
        """初始化表格符号和颜色"""
        # 表格绘制符号
        self.symbols = {
            'h_line': '-',     # 水平线
            'v_line': '|',     # 垂直线
            'cross': '+',      # 交叉点
            'top_left': '+',   # 左上角
            'top_right': '+',  # 右上角
            'bottom_left': '+', # 左下角
            'bottom_right': '+' # 右下角
        }
        
        # 信号颜色映射
        self.signal_colors = {
            'BULLISH': Fore.GREEN,
            'BEARISH': Fore.RED,
            'NEUTRAL': Fore.YELLOW,
            'BUY': Fore.GREEN,
            'SELL': Fore.RED,
            'HOLD': Fore.YELLOW,
            'SHORT': Fore.RED,
            'COVER': Fore.GREEN
        }
    
    def create_header(self, title, width=60):
        """创建表格标题"""
        return f"{Fore.WHITE}{Style.BRIGHT}{title}"
    
    def create_separator(self, widths):
        """创建表格分隔线"""
        parts = []
        for width in widths:
            parts.append(self.symbols['h_line'] * width)
        return self.symbols['cross'].join(parts)
    
    def create_row(self, columns, widths, colors=None):
        """创建表格行"""
        row = []
        for i, col in enumerate(columns):
            text = str(col)
            color = colors[i] if colors and i < len(colors) else ''
            padding = widths[i] - len(text)
            row.append(f"{color}{text}{' ' * padding}{Style.RESET_ALL}")
        return self.symbols['v_line'].join(row)
    
    def format_confidence(self, confidence):
        """格式化置信度为百分比"""
        if confidence is None:
            return "None%"
        elif isinstance(confidence, (int, float)):
            if confidence <= 1:
                return f"{confidence*100:.1f}%"
            else:
                return f"{confidence:.1f}%"
        return f"{confidence}%"
    
    def display_analyst_signals(self, signals):
        """显示分析师信号表格"""
        # 表头
        headers = ["Analyst", "Signal", "Confidence"]
        
        # 计算列宽度
        col_widths = [20, 10, 12]  # 默认宽度
        
        # 标准化信号数据
        formatted_signals = []
        for signal in signals:
            # 获取分析师名称
            if 'agent_name' in signal:
                analyst = signal['agent_name']
            elif 'agent' in signal:
                analyst = signal['agent']
            elif 'name' in signal:
                analyst = signal['name']
            else:
                analyst = "Unknown"
            
            # 获取信号和置信度
            signal_type = signal.get('signal', '').upper()
            confidence = self.format_confidence(signal.get('confidence', 0))
            
            formatted_signals.append({
                'analyst': analyst,
                'signal': signal_type,
                'confidence': confidence
            })
        
        # 打印表格标题
        print(self.create_header("ANALYST SIGNALS:"))
        
        # 创建顶部边框
        top_border = f"{self.symbols['top_left']}" + self.create_separator(col_widths) + f"{self.symbols['top_right']}"
        print(top_border)
        
        # 打印表头
        header_row = self.create_row(headers, col_widths, [Fore.WHITE + Style.BRIGHT] * 3)
        print(f"{self.symbols['v_line']}{header_row}{self.symbols['v_line']}")
        
        # 打印表头分隔线
        header_sep = f"{self.symbols['cross']}" + self.create_separator(col_widths) + f"{self.symbols['cross']}"
        print(header_sep)
        
        # 打印数据行
        for signal in formatted_signals:
            analyst = signal['analyst']
            signal_type = signal['signal']
            confidence = signal['confidence']
            
            # 设置颜色
            signal_color = self.signal_colors.get(signal_type, Fore.WHITE)
            
            row_data = [analyst, signal_type, confidence]
            row_colors = [Fore.CYAN, signal_color, Fore.YELLOW]
            
            data_row = self.create_row(row_data, col_widths, row_colors)
            print(f"{self.symbols['v_line']}{data_row}{self.symbols['v_line']}")
            
            # 打印行分隔线
            row_sep = f"{self.symbols['cross']}" + self.create_separator(col_widths) + f"{self.symbols['cross']}"
            print(row_sep)
    
    def display_trading_decision(self, decision):
        """显示交易决策表格"""
        # 提取决策数据
        action = decision.get('action', '').upper()
        quantity = decision.get('quantity', 0)
        confidence = self.format_confidence(decision.get('confidence', 0))
        
        # 设置列宽
        col_width = 15
        
        # 打印表格标题
        print(self.create_header("\nTRADING DECISION:"))
        
        # 创建顶部边框
        top_border = f"{self.symbols['top_left']}" + self.symbols['h_line'] * col_width + f"{self.symbols['top_right']}"
        print(top_border)
        
        # 打印行数据
        # 行1: Action
        action_color = self.signal_colors.get(action, Fore.WHITE)
        print(f"{self.symbols['v_line']}Action          {self.symbols['v_line']}")
        print(f"{self.symbols['v_line']}{action_color}{action}{' ' * (col_width - len(action))}{Style.RESET_ALL}{self.symbols['v_line']}")
        print(f"{self.symbols['cross']}" + self.symbols['h_line'] * col_width + f"{self.symbols['cross']}")
        
        # 行2: Quantity
        print(f"{self.symbols['v_line']}Quantity        {self.symbols['v_line']}")
        print(f"{self.symbols['v_line']}{action_color}{quantity}{' ' * (col_width - len(str(quantity)))}{Style.RESET_ALL}{self.symbols['v_line']}")
        print(f"{self.symbols['cross']}" + self.symbols['h_line'] * col_width + f"{self.symbols['cross']}")
        
        # 行3: Confidence
        print(f"{self.symbols['v_line']}Confidence      {self.symbols['v_line']}")
        print(f"{self.symbols['v_line']}{Fore.YELLOW}{confidence}{' ' * (col_width - len(str(confidence)))}{Style.RESET_ALL}{self.symbols['v_line']}")
        print(f"{self.symbols['bottom_left']}" + self.symbols['h_line'] * col_width + f"{self.symbols['bottom_right']}")


def print_trading_output(decision_dict):
    """
    打印美观的交易决策输出
    
    Args:
        decision_dict: 包含决策和分析师信号的字典
    """
    # 清屏（可选）
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # 创建显示器
    display = TradingDisplay()
    
    # 提取分析师信号
    analyst_signals = decision_dict.get('agent_signals', [])
    
    # 显示分析师信号
    display.display_analyst_signals(analyst_signals)
    
    # 显示交易决策
    display.display_trading_decision(decision_dict)
    
    # 如果有推理说明，显示它
    reasoning = decision_dict.get('reasoning', '')
    if reasoning:
        # 找到第一个换行符的位置来截取中文部分
        chinese_part = ""
        if "\n\n" in reasoning:
            parts = reasoning.split("\n\n")
            if len(parts) > 1:
                chinese_part = parts[-1]
        
        print(f"\n{Fore.WHITE}{Style.BRIGHT}决策理由:{Style.RESET_ALL}")
        if chinese_part:
            print(f"{Fore.CYAN}{chinese_part}{Style.RESET_ALL}")
        else:
            # 如果没有找到中文部分，显示前100个字符
            short_reasoning = reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
            print(f"{Fore.CYAN}{short_reasoning}{Style.RESET_ALL}")