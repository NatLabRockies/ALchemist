"""
Helper functions for visualization module.

Utilities for data preparation, validation, and computation.
"""

import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple, Union, List


def check_matplotlib() -> None:
    """
    Check if matplotlib is available for plotting.
    
    Raises:
        ImportError: If matplotlib is not installed
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def compute_z_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray
) -> np.ndarray:
    """
    Compute standardized residuals (z-scores).
    
    z = (y_true - y_pred) / y_std
    
    Args:
        y_true: Actual experimental values
        y_pred: Model predicted values
        y_std: Prediction standard deviations
    
    Returns:
        Array of z-scores (standardized residuals)
    
    Note:
        Small epsilon (1e-10) added to denominator to avoid division by zero.
    """
    return (y_true - y_pred) / (y_std + 1e-10)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    prob_levels: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute nominal vs empirical coverage for calibration curves.
    
    For each nominal probability level, computes the empirical fraction of
    observations that fall within the predicted confidence interval.
    
    Args:
        y_true: Actual experimental values
        y_pred: Model predicted values
        y_std: Prediction standard deviations
        prob_levels: Nominal coverage probabilities to evaluate.
                    Default: np.arange(0.10, 1.00, 0.05)
    
    Returns:
        Tuple of (nominal_probs, empirical_coverage)
        - nominal_probs: The requested probability levels
        - empirical_coverage: Observed coverage fractions
    
    Example:
        >>> nominal, empirical = compute_calibration_metrics(y_true, y_pred, y_std)
        >>> # nominal[i] is the expected coverage (e.g., 0.68 for ±1σ)
        >>> # empirical[i] is the observed coverage fraction
    """
    from scipy import stats
    
    if prob_levels is None:
        prob_levels = np.arange(0.10, 1.00, 0.05)
    
    empirical_coverage = []
    
    for prob in prob_levels:
        # Convert probability to sigma multiplier
        # For symmetric interval: P(|Z| < z) = prob → z = Φ^(-1)((1+prob)/2)
        sigma = stats.norm.ppf((1 + prob) / 2)
        
        # Compute empirical coverage at this sigma level
        lower_bound = y_pred - sigma * y_std
        upper_bound = y_pred + sigma * y_std
        within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
        empirical_coverage.append(np.mean(within_interval))
    
    return prob_levels, np.array(empirical_coverage)


def sort_legend_items(labels: list) -> list:
    """
    Sort legend labels for consistent ordering.
    
    Preferred order: Prediction, uncertainty bands (small to large), Experiments
    
    Args:
        labels: List of legend label strings
    
    Returns:
        List of indices for sorted order
    """
    def sort_key(lbl):
        if 'Prediction' in lbl:
            return (0, 0)
        elif 'σ' in lbl:
            # Extract sigma value for sorting bands
            import re
            match = re.search(r'±([\d.]+)σ', lbl)
            if match:
                return (1, float(match.group(1)))
            return (1, 999)
        elif 'Experiment' in lbl:
            return (2, 0)
        else:
            return (3, 0)
    
    indices = list(range(len(labels)))
    indices.sort(key=lambda i: sort_key(labels[i]))
    return indices


def generate_subplot_labels(n: int, fmt: str = "({alpha})") -> List[str]:
    """
    Generate sequential subplot labels.

    Args:
        n: Number of labels to generate.
        fmt: Format string. ``{alpha}`` is replaced by a, b, c, ...;
             ``{ALPHA}`` by A, B, C, ...; ``{num}`` by 1, 2, 3, ...

    Returns:
        List of label strings, e.g. ``['(a)', '(b)', '(c)']``.

    Examples:
        >>> generate_subplot_labels(3)
        ['(a)', '(b)', '(c)']
        >>> generate_subplot_labels(2, fmt="{ALPHA})")
        ['A)', 'B)']
        >>> generate_subplot_labels(4, fmt="({num})")
        ['(1)', '(2)', '(3)', '(4)']
    """
    labels: List[str] = []
    for i in range(n):
        label = fmt
        label = label.replace("{alpha}", chr(ord('a') + i))
        label = label.replace("{ALPHA}", chr(ord('A') + i))
        label = label.replace("{num}", str(i + 1))
        labels.append(label)
    return labels


def resolve_subplot_labels(
    subplot_labels: Optional[Union[bool, str, List[str]]],
    n_axes: int,
    fmt: str = "({alpha})",
) -> Optional[List[str]]:
    """
    Normalize a ``subplot_labels`` argument into a list of strings.

    Args:
        subplot_labels: User-supplied value.
            - ``None`` / ``False`` → no labels.
            - ``True`` → auto-generate ``(a)``, ``(b)``, ...
            - ``str`` → single label (broadcast to one-element list).
            - ``List[str]`` → used as-is (length must match *n_axes*).
        n_axes: Expected number of axes / subplots.
        fmt: Format string forwarded to :func:`generate_subplot_labels` when
             *subplot_labels* is ``True``.

    Returns:
        A list of *n_axes* label strings, or ``None`` if labels are disabled.

    Raises:
        ValueError: If a list is given whose length does not match *n_axes*.
    """
    if subplot_labels is None or subplot_labels is False:
        return None

    if subplot_labels is True:
        return generate_subplot_labels(n_axes, fmt=fmt)

    if isinstance(subplot_labels, str):
        return [subplot_labels]

    if isinstance(subplot_labels, list):
        if len(subplot_labels) != n_axes:
            raise ValueError(
                f"subplot_labels has {len(subplot_labels)} items but the figure "
                f"has {n_axes} axes. Provide exactly {n_axes} labels."
            )
        return list(subplot_labels)

    raise TypeError(
        f"subplot_labels must be bool, str, or list of str, got {type(subplot_labels).__name__}"
    )


def annotate_subplot_label(
    ax,
    label: str,
    loc: str = "upper left",
    fontsize: int = 12,
    fontweight: str = "bold",
    offset: Tuple[float, float] = (0.02, 0.98),
) -> None:
    """
    Place a panel label (e.g. ``(a)``) on a matplotlib axes.

    Works for both 2-D ``Axes`` and 3-D ``Axes3D`` objects.

    Args:
        ax: A matplotlib Axes or Axes3D instance.
        label: Text to display, e.g. ``"(a)"``.
        loc: Shorthand for anchor position.  Only ``"upper left"`` (default)
             and ``"upper right"`` are currently supported; other values fall
             back to ``"upper left"``.
        fontsize: Font size in points (default 12).
        fontweight: Font weight string (default ``"bold"``).
        offset: ``(x, y)`` in axes-fraction coordinates.  The default
                ``(0.02, 0.98)`` places the label near the top-left corner.
                Ignored when *loc* is ``"upper right"``.
    """
    if loc == "upper right":
        x, y = 0.98, 0.98
        ha = "right"
    else:
        x, y = offset
        ha = "left"

    # Axes3D uses text2D for 2-D overlay on a 3-D canvas
    text_fn = getattr(ax, "text2D", None) or ax.text
    text_fn(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        va="top",
        ha=ha,
    )


# ---------------------------------------------------------------------------
# Number-formatting helpers
# ---------------------------------------------------------------------------

def make_formatter(fmt: Any):
    """Convert a user-supplied formatter value to a ``matplotlib.ticker.Formatter``.

    Accepted input types:

    * **Python brace-style format string** (contains ``{``), e.g. ``'{:.1%}'`` or
      ``'{:.2f}'`` → ``StrMethodFormatter``
    * **printf-style format string** (no ``{``), e.g. ``'%.2f'`` →
      ``FormatStrFormatter``
    * **callable** ``func(value, pos)`` → ``FuncFormatter``
    * A ``matplotlib.ticker.Formatter`` instance → returned as-is

    Args:
        fmt: Formatter specification (see above).

    Returns:
        A ``matplotlib.ticker.Formatter`` instance.

    Raises:
        TypeError: If *fmt* is not one of the accepted types.
    """
    import matplotlib.ticker as ticker

    if isinstance(fmt, ticker.Formatter):
        return fmt
    if callable(fmt):
        return ticker.FuncFormatter(fmt)
    if isinstance(fmt, str):
        if '{' in fmt:
            return ticker.StrMethodFormatter(fmt)
        return ticker.FormatStrFormatter(fmt)
    raise TypeError(
        f"formatters values must be a format string, callable, or "
        f"matplotlib.ticker.Formatter instance, got {type(fmt).__name__!r}"
    )


def apply_axis_formatters(ax, formatters: Optional[Dict[str, Any]]) -> None:
    """Apply per-axis number formatters to a matplotlib ``Axes`` object.

    Reads the keys ``'x'``, ``'y'``, and ``'z'`` (Axes3D only) from
    *formatters* and sets the corresponding major tick formatter.

    Args:
        ax: A ``matplotlib.axes.Axes`` or ``mpl_toolkits.mplot3d.Axes3D``
            instance.
        formatters: Dict mapping axis name to a formatter value accepted by
            :func:`make_formatter`.  ``None`` or missing keys are silently
            skipped.
    """
    if not formatters:
        return
    for key, axis_obj in [
        ('x', ax.xaxis),
        ('y', ax.yaxis),
        ('z', getattr(ax, 'zaxis', None)),
    ]:
        fmt = formatters.get(key)
        if fmt is None or axis_obj is None:
            continue
        axis_obj.set_major_formatter(make_formatter(fmt))


def apply_colorbar_formatter(cbar, formatters: Optional[Dict[str, Any]]) -> None:
    """Apply a colorbar tick formatter.

    Reads the ``'cbar'`` key from *formatters* and sets the major tick
    formatter on the colorbar's long axis.

    Args:
        cbar: A ``matplotlib.colorbar.Colorbar`` instance.
        formatters: Dict that may contain a ``'cbar'`` key whose value is
            accepted by :func:`make_formatter`.  ``None`` or a missing key is
            silently skipped.
    """
    if not formatters:
        return
    fmt = formatters.get('cbar')
    if fmt is None:
        return
    formatter = make_formatter(fmt)
    # Colorbars can be oriented horizontally or vertically
    if cbar.orientation == 'horizontal':
        cbar.ax.xaxis.set_major_formatter(formatter)
    else:
        cbar.ax.yaxis.set_major_formatter(formatter)

