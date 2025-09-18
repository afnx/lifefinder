import os
import pandas as pd
from datetime import datetime

from lifefinder.utils.logger import get_logger

logger = get_logger("report")


def generate_summary(df: pd.DataFrame) -> str:
    """
    Generate a text summary of predictions.

    Args:
        df (pd.DataFrame): DataFrame with 'pl_name' and 'habitability_prob'.

    Returns:
        str: Summary report as plain text.
    """
    try:
        summary_lines = [
            "Exoplanet Habitability Prediction Report",
            "=" * 50,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total planets analyzed: {len(df)}",
            "",
        ]
        top_candidates = df.sort_values("habitability_prob", ascending=False).head(10)
        summary_lines.append("Top 10 candidates:")
        for _, row in top_candidates.iterrows():
            summary_lines.append(f" - {row['pl_name']}: {row['habitability_prob']:.3f}")
        return "\n".join(summary_lines)
    except Exception as e:
        logger.error(f"Error generating summary: {e}", exc_info=True)
        return ""


def save_report(df: pd.DataFrame, output_path: str, fmt: str = "txt") -> None:
    """
    Save prediction report to file.

    Args:
        df (pd.DataFrame): Predictions DataFrame.
        output_path (str): File path for the report.
        fmt (str): 'txt', 'csv', or 'html'.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if fmt == "txt":
            with open(output_path, "w") as f:
                f.write(generate_summary(df))
        elif fmt == "csv":
            df.to_csv(output_path, index=False)
        elif fmt == "html":
            df.to_html(output_path, index=False)
        else:
            raise ValueError(f"Unsupported report format: {fmt}")
        logger.info(f"Report saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving report: {e}", exc_info=True)
