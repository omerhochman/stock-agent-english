import pandas as pd
import logging

logger = logging.getLogger(__name__)

# 配置
MAX_MARKDOWN_ROWS = 50  # Markdown输出中显示的最大行数
MAX_MARKDOWN_COLS = 10  # 显示的最大列数


def format_df_to_markdown(df: pd.DataFrame, max_rows: int = MAX_MARKDOWN_ROWS, max_cols: int = MAX_MARKDOWN_COLS) -> str:
    """将Pandas DataFrame格式化为带有截断的Markdown字符串。

    参数:
        df: 要格式化的DataFrame
        max_rows: 输出中包含的最大行数
        max_cols: 输出中包含的最大列数

    返回:
        DataFrame的Markdown格式字符串表示
    """
    if df.empty:
        logger.warning("尝试将空DataFrame格式化为Markdown。")
        return "(没有可显示的数据)"

    original_rows, original_cols = df.shape
    truncated = False
    truncation_notes = []

    if original_rows > max_rows:
        df_display = pd.concat(
            [df.head(max_rows // 2), df.tail(max_rows - max_rows // 2)])
        truncation_notes.append(
            f"行数已从{original_rows}截断为{max_rows}")
        truncated = True
    else:
        df_display = df

    if original_cols > max_cols:
        # 选择首尾列进行显示
        cols_to_show = df_display.columns[:max_cols // 2].tolist() + \
            df_display.columns[-(max_cols - max_cols // 2):].tolist()
        # 确保在原始列数较小但大于max_cols时没有重复列
        cols_to_show = sorted(list(set(cols_to_show)),
                              key=list(df_display.columns).index)
        df_display = df_display[cols_to_show]
        truncation_notes.append(
            f"列数已从{original_cols}截断为{len(cols_to_show)}")
        truncated = True

    try:
        markdown_table = df_display.to_markdown(index=False)
    except Exception as e:
        logger.error(
            f"将DataFrame转换为Markdown时出错: {e}", exc_info=True)
        return "错误：无法将数据格式化为Markdown表格。"

    if truncated:
        notes = "; ".join(truncation_notes)
        logger.debug(
            f"已生成带有截断说明的Markdown表格: {notes}")
        return f"注意：数据已被截断({notes})。\n\n{markdown_table}"
    else:
        logger.debug("已生成未截断的Markdown表格。")
        return markdown_table