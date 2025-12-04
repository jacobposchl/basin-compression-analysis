"""Statistical correlation and testing utilities."""

import numpy as np
from scipy.stats import (
    pearsonr,
    spearmanr,
    ttest_ind,
    f_oneway,
    pointbiserialr
)
from typing import Tuple, Optional, List


def compute_correlations(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'pearson'
) -> Tuple[float, float]:
    """
    Compute correlation between two arrays.
    
    Args:
        x: First array
        y: Second array
        method: Correlation method ('pearson' or 'spearman')
    
    Returns:
        correlation: Correlation coefficient
        p_value: P-value
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return np.nan, np.nan
    
    if method == 'pearson':
        corr, p_val = pearsonr(x_clean, y_clean)
    elif method == 'spearman':
        corr, p_val = spearmanr(x_clean, y_clean)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return corr, p_val


def point_biserial_correlation(
    continuous: np.ndarray,
    binary: np.ndarray
) -> Tuple[float, float]:
    """
    Compute point-biserial correlation (for continuous vs binary).
    
    Args:
        continuous: Continuous variable
        binary: Binary variable (0/1 or bool)
    
    Returns:
        correlation: Correlation coefficient
        p_value: P-value
    """
    # Remove NaN values
    mask = ~(np.isnan(continuous))
    continuous_clean = continuous[mask]
    binary_clean = np.array(binary, dtype=float)[mask]
    
    if len(continuous_clean) < 2:
        return np.nan, np.nan
    
    corr, p_val = pointbiserialr(binary_clean, continuous_clean)
    return corr, p_val


def run_ttest(
    group1: np.ndarray,
    group2: np.ndarray
) -> Tuple[float, float]:
    """
    Run independent samples t-test.
    
    Args:
        group1: First group
        group2: Second group
    
    Returns:
        t_statistic: T-statistic
        p_value: P-value
    """
    # Remove NaN values
    group1_clean = group1[~np.isnan(group1)]
    group2_clean = group2[~np.isnan(group2)]
    
    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return np.nan, np.nan
    
    t_stat, p_val = ttest_ind(group1_clean, group2_clean)
    return t_stat, p_val


def run_anova(groups: List[np.ndarray]) -> Tuple[float, float]:
    """
    Run one-way ANOVA test.
    
    Args:
        groups: List of arrays, one per group
    
    Returns:
        f_statistic: F-statistic
        p_value: P-value
    """
    # Remove NaN values and filter groups with insufficient samples
    groups_clean = []
    for group in groups:
        # Convert to numpy array if needed
        if not isinstance(group, np.ndarray):
            group = np.array(group)
        
        # Remove NaN values
        group_clean = group[~np.isnan(group)]
        if len(group_clean) >= 2:
            groups_clean.append(group_clean)
    
    if len(groups_clean) < 2:
        return np.nan, np.nan
    
    f_stat, p_val = f_oneway(*groups_clean)
    return f_stat, p_val


def multiple_comparison_correction(
    p_values: np.ndarray,
    method: str = 'bonferroni'
) -> np.ndarray:
    """
    Apply multiple comparison correction to p-values.
    
    Args:
        p_values: Array of p-values
        method: Correction method ('bonferroni' or 'fdr_bh' for Benjamini-Hochberg)
    
    Returns:
        corrected_p_values: Corrected p-values
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        raise ImportError("statsmodels is required for multiple comparison correction. "
                         "Install it with: pip install statsmodels")
    
    # Remove NaN values for correction
    mask = ~np.isnan(p_values)
    p_clean = p_values[mask]
    
    if len(p_clean) == 0:
        return p_values
    
    if method == 'bonferroni':
        method_code = 'bonferroni'
    elif method == 'fdr_bh':
        method_code = 'fdr_bh'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    _, p_corrected, _, _ = multipletests(p_clean, method=method_code)
    
    # Put corrected values back
    corrected = p_values.copy()
    corrected[mask] = p_corrected
    
    return corrected

