"""
Creator Growth Navigator - Helper Utilities

Common utility functions for data manipulation, validation, visualization,
and general operations used across the entire project.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings

# Import logging
from .logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# DATA MANIPULATION HELPERS
# =============================================================================

def safe_divide(numerator: Union[float, int, np.ndarray], 
                denominator: Union[float, int, np.ndarray], 
                default: float = 0.0) -> Union[float, np.ndarray]:
    """Safely divide two numbers, handling division by zero.
    
    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return when denominator is zero
        
    Returns:
        Result of division or default value
    """
    if isinstance(denominator, (int, float)):
        return numerator / denominator if denominator != 0 else default
    elif isinstance(denominator, np.ndarray):
        return np.where(denominator != 0, numerator / denominator, default)
    else:
        return default


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0 if new_value == 0 else float('inf')
    return ((new_value - old_value) / abs(old_value)) * 100


def rolling_aggregation(df: pd.DataFrame, 
                       window: int, 
                       agg_functions: Dict[str, str],
                       date_column: str = 'date') -> pd.DataFrame:
    """Apply rolling aggregation to dataframe columns.
    
    Args:
        df: Input dataframe
        window: Rolling window size
        agg_functions: Dictionary mapping column names to aggregation functions
        date_column: Name of date column for sorting
        
    Returns:
        Dataframe with rolling aggregations applied
    """
    if date_column in df.columns:
        df = df.sort_values(date_column)
    
    result_df = df.copy()
    
    for column, agg_func in agg_functions.items():
        if column in df.columns:
            if agg_func == 'mean':
                result_df[f'{column}_rolling_{window}d'] = df[column].rolling(window=window).mean()
            elif agg_func == 'sum':
                result_df[f'{column}_rolling_{window}d'] = df[column].rolling(window=window).sum()
            elif agg_func == 'std':
                result_df[f'{column}_rolling_{window}d'] = df[column].rolling(window=window).std()
            elif agg_func == 'var':
                result_df[f'{column}_rolling_{window}d'] = df[column].rolling(window=window).var()
            else:
                logger.warning(f"Unknown aggregation function: {agg_func}")
    
    return result_df


def interpolate_missing_values(df: pd.DataFrame, 
                              columns: List[str], 
                              method: str = 'linear') -> pd.DataFrame:
    """Interpolate missing values in specified columns.
    
    Args:
        df: Input dataframe
        columns: List of columns to interpolate
        method: Interpolation method ('linear', 'time', 'polynomial')
        
    Returns:
        Dataframe with interpolated values
    """
    result_df = df.copy()
    
    for column in columns:
        if column in df.columns:
            missing_before = result_df[column].isna().sum()
            
            if method == 'linear':
                result_df[column] = result_df[column].interpolate(method='linear')
            elif method == 'time':
                result_df[column] = result_df[column].interpolate(method='time')
            elif method == 'polynomial':
                result_df[column] = result_df[column].interpolate(method='polynomial', order=2)
            
            missing_after = result_df[column].isna().sum()
            logger.debug(f"Interpolated {column}: {missing_before - missing_after} values filled")
    
    return result_df


# =============================================================================
# DATE AND TIME HELPERS
# =============================================================================

def parse_date_string(date_string: str, 
                     format_strings: List[str] = None) -> Optional[datetime]:
    """Parse date string with multiple possible formats.
    
    Args:
        date_string: Date string to parse
        format_strings: List of date formats to try
        
    Returns:
        Parsed datetime object or None if parsing fails
    """
    if format_strings is None:
        format_strings = [
            '%Y-%m-%d',
            '%d-%m-%Y', 
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%d-%m-%Y %H:%M:%S'
        ]
    
    for fmt in format_strings:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date string: {date_string}")
    return None


def get_week_boundaries(date: datetime) -> Tuple[datetime, datetime]:
    """Get start and end of week for a given date.
    
    Args:
        date: Input date
        
    Returns:
        Tuple of (week_start, week_end)
    """
    days_since_monday = date.weekday()
    week_start = date - timedelta(days=days_since_monday)
    week_end = week_start + timedelta(days=6)
    
    return week_start, week_end


def create_date_features(df: pd.DataFrame, 
                        date_column: str = 'date') -> pd.DataFrame:
    """Create additional date-based features.
    
    Args:
        df: Input dataframe
        date_column: Name of date column
        
    Returns:
        Dataframe with additional date features
    """
    result_df = df.copy()
    
    if date_column not in df.columns:
        logger.error(f"Date column '{date_column}' not found in dataframe")
        return result_df
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(result_df[date_column]):
        result_df[date_column] = pd.to_datetime(result_df[date_column])
    
    # Extract date components
    result_df['year'] = result_df[date_column].dt.year
    result_df['month'] = result_df[date_column].dt.month
    result_df['quarter'] = result_df[date_column].dt.quarter
    result_df['week_of_year'] = result_df[date_column].dt.isocalendar().week
    result_df['day_of_week'] = result_df[date_column].dt.dayofweek
    result_df['day_of_month'] = result_df[date_column].dt.day
    result_df['is_weekend'] = result_df['day_of_week'].isin([5, 6]).astype(int)
    result_df['is_month_start'] = result_df[date_column].dt.is_month_start.astype(int)
    result_df['is_month_end'] = result_df[date_column].dt.is_month_end.astype(int)
    
    logger.debug(f"Created date features for {len(result_df)} rows")
    return result_df


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_dataframe_schema(df: pd.DataFrame, 
                             required_columns: List[str],
                             column_types: Optional[Dict[str, str]] = None) -> Tuple[bool, List[str]]:
    """Validate dataframe schema against requirements.
    
    Args:
        df: Input dataframe
        required_columns: List of required column names
        column_types: Optional dictionary mapping column names to expected types
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        issues.append(f"Missing required columns: {list(missing_columns)}")
    
    # Check column types if specified
    if column_types:
        for column, expected_type in column_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if expected_type not in actual_type:
                    issues.append(f"Column '{column}' has type '{actual_type}', expected '{expected_type}'")
    
    # Check for empty dataframe
    if df.empty:
        issues.append("Dataframe is empty")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def detect_outliers(series: pd.Series, 
                   method: str = 'zscore', 
                   threshold: float = 3.0) -> pd.Series:
    """Detect outliers in a pandas Series.
    
    Args:
        series: Input data series
        method: Method for outlier detection ('zscore', 'iqr')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean series indicating outliers
    """
    if method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    elif method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    else:
        logger.error(f"Unknown outlier detection method: {method}")
        return pd.Series([False] * len(series), index=series.index)


def validate_data_quality(df: pd.DataFrame, 
                         quality_checks: Dict[str, Any]) -> Dict[str, Any]:
    """Perform comprehensive data quality validation.
    
    Args:
        df: Input dataframe
        quality_checks: Dictionary of quality check parameters
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'issues': []
    }
    
    # Check missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    max_missing_allowed = quality_checks.get('max_missing_percentage', 5.0)
    
    for column, missing_pct in missing_percentage.items():
        if missing_pct > max_missing_allowed:
            results['issues'].append(f"Column '{column}' has {missing_pct:.1f}% missing values")
    
    # Check minimum data points
    min_rows = quality_checks.get('min_data_points', 90)
    if len(df) < min_rows:
        results['issues'].append(f"Dataset has only {len(df)} rows, minimum required: {min_rows}")
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        results['issues'].append(f"Found {duplicate_count} duplicate rows")
    
    # Check numeric ranges
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if column in quality_checks:
            min_val, max_val = quality_checks[column]
            out_of_range = ((df[column] < min_val) | (df[column] > max_val)).sum()
            if out_of_range > 0:
                results['issues'].append(f"Column '{column}' has {out_of_range} values out of range [{min_val}, {max_val}]")
    
    results['is_valid'] = len(results['issues']) == 0
    return results


# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

def calculate_confidence_interval(values: np.ndarray, 
                                 confidence_level: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for a sample.
    
    Args:
        values: Sample values
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    
    mean = np.mean(values)
    std_error = stats.sem(values)
    degrees_freedom = len(values) - 1
    
    t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_error = t_critical * std_error
    
    return mean - margin_error, mean + margin_error


def calculate_correlation_matrix(df: pd.DataFrame, 
                               columns: Optional[List[str]] = None,
                               method: str = 'pearson') -> pd.DataFrame:
    """Calculate correlation matrix for specified columns.
    
    Args:
        df: Input dataframe
        columns: List of columns to include (all numeric if None)
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Correlation matrix
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    correlation_df = df[columns].corr(method=method)
    logger.debug(f"Calculated {method} correlation matrix for {len(columns)} columns")
    
    return correlation_df


def calculate_growth_metrics(values: Union[List, np.ndarray, pd.Series]) -> Dict[str, float]:
    """Calculate various growth metrics for a time series.
    
    Args:
        values: Time series values
        
    Returns:
        Dictionary with growth metrics
    """
    values = np.array(values)
    
    if len(values) < 2:
        return {'total_growth': 0.0, 'average_growth': 0.0, 'growth_rate': 0.0}
    
    # Calculate growth metrics
    total_growth = values[-1] - values[0]
    period_changes = np.diff(values)
    average_growth = np.mean(period_changes)
    
    # Growth rate (compound annual growth rate approximation)
    periods = len(values) - 1
    growth_rate = ((values[-1] / values[0]) ** (1/periods)) - 1 if values != 0 else 0.0
    
    return {
        'total_growth': float(total_growth),
        'average_growth': float(average_growth),
        'growth_rate': float(growth_rate),
        'volatility': float(np.std(period_changes)),
        'periods': periods
    }


# =============================================================================
# FILE I/O HELPERS
# =============================================================================

def save_dataframe(df: pd.DataFrame, 
                  filepath: Union[str, Path], 
                  format: str = 'csv',
                  **kwargs) -> bool:
    """Save dataframe to file with error handling.
    
    Args:
        df: Dataframe to save
        filepath: Output file path
        format: File format ('csv', 'json', 'parquet', 'excel')
        **kwargs: Additional arguments for save function
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'csv':
            df.to_csv(filepath, index=False, **kwargs)
        elif format.lower() == 'json':
            df.to_json(filepath, orient='records', **kwargs)
        elif format.lower() == 'parquet':
            df.to_parquet(filepath, **kwargs)
        elif format.lower() == 'excel':
            df.to_excel(filepath, index=False, **kwargs)
        else:
            logger.error(f"Unsupported file format: {format}")
            return False
        
        logger.info(f"Saved dataframe to {filepath}", rows=len(df), format=format)
        return True
        
    except Exception as e:
        logger.error(f"Failed to save dataframe to {filepath}: {str(e)}")
        return False


def load_dataframe(filepath: Union[str, Path], 
                  format: str = 'auto',
                  **kwargs) -> Optional[pd.DataFrame]:
    """Load dataframe from file with error handling.
    
    Args:
        filepath: Input file path
        format: File format ('auto', 'csv', 'json', 'parquet', 'excel')
        **kwargs: Additional arguments for load function
        
    Returns:
        Loaded dataframe or None if failed
    """
    try:
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None
        
        # Auto-detect format from extension
        if format == 'auto':
            format = filepath.suffix.lower().lstrip('.')
        
        if format == 'csv':
            df = pd.read_csv(filepath, **kwargs)
        elif format == 'json':
            df = pd.read_json(filepath, **kwargs)
        elif format == 'parquet':
            df = pd.read_parquet(filepath, **kwargs)
        elif format in ['xlsx', 'excel']:
            df = pd.read_excel(filepath, **kwargs)
        else:
            logger.error(f"Unsupported file format: {format}")
            return None
        
        logger.info(f"Loaded dataframe from {filepath}", 
                   rows=len(df), columns=len(df.columns), format=format)
        return df
        
    except Exception as e:
        logger.error(f"Failed to load dataframe from {filepath}: {str(e)}")
        return None


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def format_large_numbers(value: Union[int, float], 
                        precision: int = 1) -> str:
    """Format large numbers with K, M, B suffixes.
    
    Args:
        value: Number to format
        precision: Decimal places
        
    Returns:
        Formatted string
    """
    if abs(value) >= 1_000_000_000:
        return f"{value/1_000_000_000:.{precision}f}B"
    elif abs(value) >= 1_000_000:
        return f"{value/1_000_000:.{precision}f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"


def create_color_palette(n_colors: int, 
                        palette_type: str = 'viridis') -> List[str]:
    """Create a color palette for visualizations.
    
    Args:
        n_colors: Number of colors needed
        palette_type: Type of palette ('viridis', 'plasma', 'Blues', 'Reds')
        
    Returns:
        List of hex color codes
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    try:
        cmap = plt.get_cmap(palette_type)
        colors = [mcolors.rgb2hex(cmap(i / (n_colors - 1))) for i in range(n_colors)]
        return colors
    except:
        # Fallback to default colors
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        return (default_colors * ((n_colors // len(default_colors)) + 1))[:n_colors]


def calculate_chart_dimensions(data_points: int) -> Tuple[int, int]:
    """Calculate optimal chart dimensions based on data points.
    
    Args:
        data_points: Number of data points to display
        
    Returns:
        Tuple of (width, height) in pixels
    """
    # Base dimensions
    base_width = 800
    base_height = 400
    
    # Adjust based on data points
    if data_points > 365:  # More than a year
        width = min(1200, base_width + (data_points - 365) // 10)
    elif data_points > 90:  # More than 3 months
        width = base_width
    else:  # Less than 3 months
        width = max(600, base_width - (90 - data_points) * 2)
    
    height = base_height
    
    return width, height


# =============================================================================
# UTILITY DECORATORS
# =============================================================================

def handle_exceptions(default_return=None, log_errors=True):
    """Decorator to handle exceptions gracefully.
    
    Args:
        default_return: Value to return if exception occurs
        log_errors: Whether to log errors
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Exception in {func.__name__}: {str(e)}")
                return default_return
        return wrapper
    return decorator


def validate_inputs(**validators):
    """Decorator to validate function inputs.
    
    Args:
        **validators: Dictionary mapping parameter names to validation functions
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each specified parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Validation failed for parameter '{param_name}' with value {value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# EXPORT LIST
# =============================================================================

__all__ = [
    # Data manipulation
    'safe_divide', 'calculate_percentage_change', 'rolling_aggregation', 
    'interpolate_missing_values',
    
    # Date and time
    'parse_date_string', 'get_week_boundaries', 'create_date_features',
    
    # Validation
    'validate_dataframe_schema', 'detect_outliers', 'validate_data_quality',
    
    # Statistics
    'calculate_confidence_interval', 'calculate_correlation_matrix', 
    'calculate_growth_metrics',
    
    # File I/O
    'save_dataframe', 'load_dataframe',
    
    # Visualization
    'format_large_numbers', 'create_color_palette', 'calculate_chart_dimensions',
    
    # Decorators
    'handle_exceptions', 'validate_inputs'
]
