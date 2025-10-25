import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Configuration
MAX_MARKDOWN_ROWS = 50  # Maximum number of rows to display in Markdown output
MAX_MARKDOWN_COLS = 10  # Maximum number of columns to display


def format_df_to_markdown(df: pd.DataFrame, max_rows: int = MAX_MARKDOWN_ROWS, max_cols: int = MAX_MARKDOWN_COLS) -> str:
    """Format Pandas DataFrame to Markdown string with truncation.

    Args:
        df: DataFrame to format
        max_rows: Maximum number of rows to include in output
        max_cols: Maximum number of columns to include in output

    Returns:
        Markdown format string representation of DataFrame
    """
    if df.empty:
        logger.warning("Attempting to format empty DataFrame to Markdown.")
        return "(No data to display)"

    original_rows, original_cols = df.shape
    truncated = False
    truncation_notes = []

    if original_rows > max_rows:
        df_display = pd.concat(
            [df.head(max_rows // 2), df.tail(max_rows - max_rows // 2)])
        truncation_notes.append(
            f"Row count truncated from {original_rows} to {max_rows}")
        truncated = True
    else:
        df_display = df

    if original_cols > max_cols:
        # Select first and last columns for display
        cols_to_show = df_display.columns[:max_cols // 2].tolist() + \
            df_display.columns[-(max_cols - max_cols // 2):].tolist()
        # Ensure no duplicate columns when original column count is small but greater than max_cols
        cols_to_show = sorted(list(set(cols_to_show)),
                              key=list(df_display.columns).index)
        df_display = df_display[cols_to_show]
        truncation_notes.append(
            f"Column count truncated from {original_cols} to {len(cols_to_show)}")
        truncated = True

    try:
        markdown_table = df_display.to_markdown(index=False)
    except Exception as e:
        logger.error(
            f"Error converting DataFrame to Markdown: {e}", exc_info=True)
        return "Error: Unable to format data as Markdown table."

    if truncated:
        notes = "; ".join(truncation_notes)
        logger.debug(
            f"Generated Markdown table with truncation notes: {notes}")
        return f"Note: Data has been truncated ({notes}).\n\n{markdown_table}"
    else:
        logger.debug("Generated untruncated Markdown table.")
        return markdown_table