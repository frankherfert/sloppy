import pandas pd
import numpy as np


def value_counts_percentage(df, column):
    """
    Creates a DataFrame with absolute counts and percentages of total for column values.
    """
    counts = pd.DataFrame(df[column].value_counts(dropna=False))
    counts = counts.rename(columns={column:"count"})
    counts["percentage"] = 100*(counts["count"] / len(df))
    
    return counts