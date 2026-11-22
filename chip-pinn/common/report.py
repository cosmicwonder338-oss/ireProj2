"""Small reporting helpers shared by stage scripts and the README generator."""
from __future__ import annotations

import numpy as np
import pandas as pd


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    """Markdown table without the optional `tabulate` dependency."""
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
