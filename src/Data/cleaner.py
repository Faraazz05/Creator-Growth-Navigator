"""
Creator Growth Navigator - Data Cleaning Module

Comprehensive data cleaning utilities for handling missing values, outliers,
data type conversions, and quality improvements across all data sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import warnings

# Import from project modules
from src.config.config import DATA_QUALITY_THRESHOLDS, FEATURE_CONFIG
from src.utils.logger import get_data_logger
from src.utils.helpers import (
    detect_outliers, interpolate_missing_values, safe_divide,
    parse_date_string, validate_data_quality
)

logger = get_data_logger()
warnings.filterwarnings('ignore')


# =============================================================================
# CORE CLEANING FUNCTIONS
# =============================================================================

def clean_raw_data(df: pd.DataFrame,
                  fill_missing: bool = True,
                  handle_outliers: bool = True,
                  fix_data_types: bool = True,
                  validate_ranges: bool = True) -> pd.DataFrame:
    """Comprehensive cleaning of raw creator data.
    
    Args:
        df: Raw dataframe to clean
        fill_missing: Whether to handle missing values
        handle_outliers: Whether to detect and handle outliers
        fix_data_types: Whether to fix data type issues
        validate_ranges: Whether to validate value ranges
        
    Returns:
        Cleaned dataframe
    """
    logger.info("Starting comprehensive data cleaning")
    df_clean = df.copy()
    
    # Track cleaning statistics
    cleaning_stats = {
        'original_rows': len(df_clean),
        'original_columns': len(df_clean.columns),
        'missing_values_handled': 0,
        'outliers_handled': 0,
        'data_type_fixes': 0,
        'range_violations_fixed': 0
    }
    
    # 1. Fix data types
    if fix_data_types:
        df_clean, type_fixes = fix_data_types_comprehensive(df_clean)
        cleaning_stats['data_type_fixes'] = type_fixes
    
    # 2. Handle missing values
    if fill_missing:
        df_clean, missing_handled = handle_missing_values(df_clean)
        cleaning_stats['missing_values_handled'] = missing_handled
    
    # 3. Handle outliers
    if handle_outliers:
        df_clean, outliers_handled = handle_outliers_comprehensive(df_clean)
        cleaning_stats['outliers_handled'] = outliers_handled
    
    # 4. Validate and fix ranges
    if validate_ranges:
        df_clean, range_fixes = validate_and_fix_ranges(df_clean)
        cleaning_stats['range_violations_fixed'] = range_fixes
    
    # 5. Final consistency checks
    df_clean = ensure_data_consistency(df_clean)
    
    # Log cleaning results
    cleaning_stats['final_rows'] = len(df_clean)
    logger.info("Data cleaning completed", **cleaning_stats)
    
    return df_clean


def fix_data_types_comprehensive(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Fix data type issues in the dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple of (cleaned_df, number_of_fixes)
    """
    df_clean = df.copy()
    fixes_count = 0
    
    # Define expected data types for creator data
    expected_types = {
        # Integer columns
        'followers': 'int64',
        'growth': 'int64',
        'follows': 'int64',
        'unfollows': 'int64',
        'posts': 'int64',
        'stories': 'int64',
        'reels': 'int64',
        'ads_posted': 'int64',
        'hashtags_used': 'int64',
        'reach': 'int64',
        'profile_visits': 'int64',
        'post_likes': 'int64',
        'reel_plays': 'int64',
        'reel_engagement': 'int64',
        'story_reach': 'int64',
        'story_engagement': 'int64',
        'comments': 'int64',
        'saves': 'int64',
        'shares': 'int64',
        'minutes_spent': 'int64',
        
        # Float columns
        'avg_hashtag_count': 'float64',
        'engagement_rate': 'float64',
        'share_posts': 'float64',
        'share_reels': 'float64',
        'share_stories': 'float64',
        'post_consistency_variance_7d': 'float64',
        'roi_follows_per_hour': 'float64',
        
        # String columns
        'username': 'str',
        'post_time': 'str'
    }
    
    for column, expected_type in expected_types.items():
        if column in df_clean.columns:
            current_type = str(df_clean[column].dtype)
            
            try:
                if expected_type == 'int64' and 'int' not in current_type:
                    # Handle potential float to int conversion
                    df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')
                    df_clean[column] = df_clean[column].fillna(0).astype('int64')
                    fixes_count += 1
                    logger.debug(f"Fixed data type for '{column}': {current_type} -> int64")
                
                elif expected_type == 'float64' and 'float' not in current_type:
                    df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')
                    fixes_count += 1
                    logger.debug(f"Fixed data type for '{column}': {current_type} -> float64")
                
                elif expected_type == 'str' and current_type != 'object':
                    df_clean[column] = df_clean[column].astype(str)
                    fixes_count += 1
                    logger.debug(f"Fixed data type for '{column}': {current_type} -> string")
                    
            except Exception as e:
                logger.warning(f"Could not fix data type for column '{column}': {e}")
    
    # Handle date columns specifically
    if 'date' in df_clean.columns:
        try:
            df_clean['date'] = pd.to_datetime(df_clean['date'], format='%d-%m-%Y', errors='coerce')
            fixes_count += 1
            logger.debug("Fixed date column data type")
        except Exception as e:
            logger.warning(f"Could not parse date column: {e}")
    
    return df_clean, fixes_count


def handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Handle missing values using appropriate strategies for each column type.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple of (cleaned_df, number_of_missing_values_handled)
    """
    df_clean = df.copy()
    initial_missing = df_clean.isnull().sum().sum()
    
    # Define missing value strategies for different column types
    strategies = {
        # Forward fill for cumulative metrics
        'forward_fill': ['followers'],
        
        # Zero fill for counts and rates
        'zero_fill': [
            'posts', 'stories', 'reels', 'ads_posted', 'hashtags_used',
            'post_likes', 'reel_plays', 'reel_engagement', 'story_reach',
            'story_engagement', 'comments', 'saves', 'shares', 'growth',
            'follows', 'unfollows', 'reach', 'profile_visits', 'minutes_spent'
        ],
        
        # Mean imputation for rates and ratios
        'mean_fill': [
            'engagement_rate', 'avg_hashtag_count', 'roi_follows_per_hour',
            'share_posts', 'share_reels', 'share_stories'
        ],
        
        # Mode fill for categorical
        'mode_fill': ['post_time', 'username'],
        
        # Interpolation for time series
        'interpolate': ['post_consistency_variance_7d']
    }
    
    # Apply strategies
    for strategy, columns in strategies.items():
        for column in columns:
            if column in df_clean.columns and df_clean[column].isnull().any():
                
                if strategy == 'forward_fill':
                    df_clean[column] = df_clean[column].fillna(method='ffill')
                    
                elif strategy == 'zero_fill':
                    df_clean[column] = df_clean[column].fillna(0)
                    
                elif strategy == 'mean_fill':
                    mean_value = df_clean[column].mean()
                    df_clean[column] = df_clean[column].fillna(mean_value)
                    
                elif strategy == 'mode_fill':
                    mode_value = df_clean[column].mode()
                    if len(mode_value) > 0:
                        df_clean[column] = df_clean[column].fillna(mode_value.iloc[0])
                    
                elif strategy == 'interpolate':
                    df_clean[column] = df_clean[column].interpolate(method='linear')
                
                logger.debug(f"Applied {strategy} to column '{column}'")
    
    # Handle any remaining missing values with appropriate defaults
    remaining_missing = df_clean.select_dtypes(include=[np.number]).columns
    for column in remaining_missing:
        if df_clean[column].isnull().any():
            df_clean[column] = df_clean[column].fillna(0)
    
    final_missing = df_clean.isnull().sum().sum()
    handled_count = initial_missing - final_missing
    
    if handled_count > 0:
        logger.info(f"Handled {handled_count} missing values")
    
    return df_clean, handled_count


def handle_outliers_comprehensive(df: pd.DataFrame,
                                method: str = 'iqr',
                                threshold: float = 3.0,
                                action: str = 'cap') -> Tuple[pd.DataFrame, int]:
    """Detect and handle outliers in numeric columns.
    
    Args:
        df: Input dataframe
        method: Outlier detection method ('zscore', 'iqr')
        threshold: Threshold for outlier detection
        action: Action to take ('cap', 'remove', 'transform')
        
    Returns:
        Tuple of (cleaned_df, number_of_outliers_handled)
    """
    df_clean = df.copy()
    outliers_handled = 0
    
    # Columns to check for outliers (exclude ratios and percentages)
    outlier_columns = [
        'followers', 'growth', 'follows', 'unfollows', 'reach', 'profile_visits',
        'post_likes', 'reel_plays', 'reel_engagement', 'story_reach',
        'story_engagement', 'comments', 'saves', 'shares', 'hashtags_used',
        'minutes_spent', 'roi_follows_per_hour'
    ]
    
    for column in outlier_columns:
        if column in df_clean.columns:
            # Detect outliers
            outlier_mask = detect_outliers(df_clean[column], method=method, threshold=threshold)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                logger.debug(f"Found {outlier_count} outliers in '{column}' using {method} method")
                
                if action == 'cap':
                    # Cap outliers to reasonable bounds
                    if method == 'iqr':
                        Q1 = df_clean[column].quantile(0.25)
                        Q3 = df_clean[column].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                    else:  # zscore
                        mean = df_clean[column].mean()
                        std = df_clean[column].std()
                        lower_bound = mean - threshold * std
                        upper_bound = mean + threshold * std
                    
                    # Ensure non-negative values for count columns
                    if column in ['followers', 'posts', 'reels', 'stories', 'reach', 'post_likes']:
                        lower_bound = max(0, lower_bound)
                    
                    df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)
                    outliers_handled += outlier_count
                    
                elif action == 'remove':
                    # Remove outlier rows (use with caution for time series)
                    df_clean = df_clean[~outlier_mask]
                    outliers_handled += outlier_count
                    
                elif action == 'transform':
                    # Log transform for highly skewed data
                    if column in ['followers', 'reach', 'reel_plays']:
                        df_clean[column] = np.log1p(df_clean[column])
                        outliers_handled += outlier_count
    
    if outliers_handled > 0:
        logger.info(f"Handled {outliers_handled} outliers using {action} method")
    
    return df_clean, outliers_handled


def validate_and_fix_ranges(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Validate and fix value ranges for business logic consistency.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple of (cleaned_df, number_of_range_fixes)
    """
    df_clean = df.copy()
    fixes_count = 0
    
    # Define business logic constraints
    range_constraints = {
        # Engagement rate should be between 0 and 0.5 (50%)
        'engagement_rate': (0.0, 0.5),
        
        # Share ratios should sum to 1 and be between 0 and 1
        'share_posts': (0.0, 1.0),
        'share_reels': (0.0, 1.0),
        'share_stories': (0.0, 1.0),
        
        # Hashtag count per post should be reasonable
        'avg_hashtag_count': (0.0, 30.0),
        
        # ROI should be positive
        'roi_follows_per_hour': (0.0, float('inf')),
        
        # Count columns should be non-negative
        'followers': (0, float('inf')),
        'posts': (0, 20),  # Max 20 posts per day is reasonable
        'reels': (0, 10),  # Max 10 reels per day
        'stories': (0, 50), # Max 50 stories per day
    }
    
    # Apply range constraints
    for column, (min_val, max_val) in range_constraints.items():
        if column in df_clean.columns:
            # Count violations before fixing
            violations = ((df_clean[column] < min_val) | (df_clean[column] > max_val)).sum()
            
            if violations > 0:
                # Fix range violations
                df_clean[column] = df_clean[column].clip(min_val, max_val)
                fixes_count += violations
                logger.debug(f"Fixed {violations} range violations in '{column}'")
    
    # Special handling for share ratios - ensure they sum to 1
    share_columns = ['share_posts', 'share_reels', 'share_stories']
    if all(col in df_clean.columns for col in share_columns):
        # Calculate row sums
        share_sums = df_clean[share_columns].sum(axis=1)
        invalid_sums = (np.abs(share_sums - 1.0) > 0.01).sum()  # Allow small tolerance
        
        if invalid_sums > 0:
            # Normalize to sum to 1
            for idx in df_clean.index:
                row_sum = df_clean.loc[idx, share_columns].sum()
                if row_sum > 0:
                    df_clean.loc[idx, share_columns] = df_clean.loc[idx, share_columns] / row_sum
            
            fixes_count += invalid_sums
            logger.debug(f"Normalized {invalid_sums} rows where share ratios didn't sum to 1")
    
    # Fix logical inconsistencies
    logical_fixes = fix_logical_inconsistencies(df_clean)
    fixes_count += logical_fixes
    
    if fixes_count > 0:
        logger.info(f"Fixed {fixes_count} range and logic violations")
    
    return df_clean, fixes_count


def fix_logical_inconsistencies(df: pd.DataFrame) -> int:
    """Fix logical inconsistencies in the data.
    
    Args:
        df: Input dataframe
        
    Returns:
        Number of logical fixes applied
    """
    fixes_count = 0
    
    # Ensure growth = follows - unfollows (approximately)
    if all(col in df.columns for col in ['growth', 'follows', 'unfollows']):
        calculated_growth = df['follows'] - df['unfollows']
        growth_diff = np.abs(df['growth'] - calculated_growth)
        inconsistent = (growth_diff > 10).sum()  # Allow small tolerance
        
        if inconsistent > 0:
            df['growth'] = calculated_growth
            fixes_count += inconsistent
            logger.debug(f"Fixed {inconsistent} growth calculation inconsistencies")
    
    # Ensure engagement metrics are consistent with reach
    if all(col in df.columns for col in ['engagement_rate', 'post_likes', 'reach']):
        # Calculate implied engagement rate
        total_engagement = df['post_likes'] + df.get('comments', 0) + df.get('saves', 0) + df.get('shares', 0)
        implied_rate = safe_divide(total_engagement, df['reach'])
        
        # Fix engagement rate if significantly different
        rate_diff = np.abs(df['engagement_rate'] - implied_rate)
        inconsistent = (rate_diff > 0.1).sum()  # 10% tolerance
        
        if inconsistent > 0:
            df['engagement_rate'] = implied_rate
            fixes_count += inconsistent
            logger.debug(f"Fixed {inconsistent} engagement rate inconsistencies")
    
    # Ensure positive follower growth is logical
    if 'followers' in df.columns:
        # Check for impossible follower jumps
        df_sorted = df.sort_values('date') if 'date' in df.columns else df
        follower_changes = df_sorted['followers'].diff()
        extreme_changes = (np.abs(follower_changes) > 50000).sum()  # 50k follower change per day is extreme
        
        if extreme_changes > 0:
            # Smooth extreme changes
            for i in range(1, len(df_sorted)):
                prev_followers = df_sorted['followers'].iloc[i-1]
                curr_change = df_sorted['followers'].iloc[i] - prev_followers
                
                if abs(curr_change) > 50000:
                    # Cap the change to reasonable bounds
                    max_daily_change = min(10000, prev_followers * 0.1)  # 10% or 10k max
                    new_change = np.sign(curr_change) * max_daily_change
                    df_sorted['followers'].iloc[i] = prev_followers + new_change
            
            fixes_count += extreme_changes
            logger.debug(f"Smoothed {extreme_changes} extreme follower changes")
    
    return fixes_count


def ensure_data_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure final data consistency across all columns.
    
    Args:
        df: Input dataframe
        
    Returns:
        Final consistent dataframe
    """
    df_clean = df.copy()
    
    # Sort by date if available
    if 'date' in df_clean.columns:
        df_clean = df_clean.sort_values('date').reset_index(drop=True)
    
    # Ensure integer columns are actually integers
    int_columns = [
        'followers', 'growth', 'follows', 'unfollows', 'posts', 'stories', 'reels',
        'ads_posted', 'hashtags_used', 'reach', 'profile_visits', 'post_likes',
        'reel_plays', 'reel_engagement', 'story_reach', 'story_engagement',
        'comments', 'saves', 'shares', 'minutes_spent'
    ]
    
    for col in int_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].round().astype('int64')
    
    # Ensure float columns have appropriate precision
    float_columns = [
        'engagement_rate', 'avg_hashtag_count', 'roi_follows_per_hour',
        'share_posts', 'share_reels', 'share_stories', 'post_consistency_variance_7d'
    ]
    
    for col in float_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].round(6)  # 6 decimal places
    
    # Remove any duplicate rows based on date and username
    if all(col in df_clean.columns for col in ['date', 'username']):
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['date', 'username'], keep='last')
        duplicates_removed = initial_rows - len(df_clean)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
    
    return df_clean


# =============================================================================
# SPECIALIZED CLEANING FUNCTIONS
# =============================================================================

def clean_engagement_data(df: pd.DataFrame) -> pd.DataFrame:
    """Specialized cleaning for engagement-related columns.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with cleaned engagement data
    """
    df_clean = df.copy()
    
    engagement_columns = [
        'post_likes', 'reel_engagement', 'story_engagement', 
        'comments', 'saves', 'shares', 'engagement_rate'
    ]
    
    for col in engagement_columns:
        if col in df_clean.columns:
            # Remove negative engagement (impossible)
            negative_count = (df_clean[col] < 0).sum()
            if negative_count > 0:
                df_clean[col] = df_clean[col].clip(lower=0)
                logger.debug(f"Fixed {negative_count} negative values in '{col}'")
            
            # Handle extremely high engagement rates (likely data errors)
            if col == 'engagement_rate':
                high_rate_count = (df_clean[col] > 1.0).sum()  # > 100%
                if high_rate_count > 0:
                    df_clean[col] = df_clean[col].clip(upper=0.5)  # Cap at 50%
                    logger.debug(f"Capped {high_rate_count} extremely high engagement rates")
    
    return df_clean


def clean_posting_data(df: pd.DataFrame) -> pd.DataFrame:
    """Specialized cleaning for posting frequency data.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with cleaned posting data
    """
    df_clean = df.copy()
    
    posting_columns = ['posts', 'reels', 'stories', 'ads_posted']
    
    for col in posting_columns:
        if col in df_clean.columns:
            # Ensure non-negative posting counts
            df_clean[col] = df_clean[col].clip(lower=0)
            
            # Cap extremely high posting counts (likely errors)
            if col == 'posts':
                max_daily_posts = 10
            elif col == 'reels':
                max_daily_posts = 5
            elif col == 'stories':
                max_daily_posts = 20
            else:  # ads_posted
                max_daily_posts = 5
            
            high_count = (df_clean[col] > max_daily_posts).sum()
            if high_count > 0:
                df_clean[col] = df_clean[col].clip(upper=max_daily_posts)
                logger.debug(f"Capped {high_count} extremely high {col} counts")
    
    # Ensure posting time format consistency
    if 'post_time' in df_clean.columns:
        # Standardize time format to HH:MM
        def standardize_time(time_str):
            try:
                if pd.isna(time_str):
                    return '12:00'  # Default noon
                time_str = str(time_str)
                if ':' in time_str:
                    return time_str[:5]  # Keep HH:MM format
                else:
                    return '12:00'  # Default for invalid format
            except:
                return '12:00'
        
        df_clean['post_time'] = df_clean['post_time'].apply(standardize_time)
    
    return df_clean


# =============================================================================
# QUALITY VALIDATION
# =============================================================================

def validate_cleaning_quality(original_df: pd.DataFrame, 
                             cleaned_df: pd.DataFrame) -> Dict[str, Any]:
    """Validate the quality of data cleaning process.
    
    Args:
        original_df: Original dataframe before cleaning
        cleaned_df: Dataframe after cleaning
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'data_preservation': {
            'original_rows': len(original_df),
            'cleaned_rows': len(cleaned_df),
            'rows_preserved': len(cleaned_df) / len(original_df) if len(original_df) > 0 else 0,
            'columns_preserved': len(set(cleaned_df.columns) & set(original_df.columns))
        },
        'quality_improvements': {
            'missing_values_before': original_df.isnull().sum().sum(),
            'missing_values_after': cleaned_df.isnull().sum().sum(),
            'data_types_fixed': 0,  # Would need tracking from cleaning process
            'outliers_handled': 0   # Would need tracking from cleaning process
        },
        'data_consistency': {
            'negative_followers': (cleaned_df.get('followers', pd.Series()) < 0).sum(),
            'impossible_engagement_rates': (cleaned_df.get('engagement_rate', pd.Series()) > 1.0).sum(),
            'missing_critical_columns': []
        }
    }
    
    # Check for critical missing columns
    critical_columns = ['date', 'followers', 'posts', 'engagement_rate']
    missing_critical = [col for col in critical_columns if col not in cleaned_df.columns]
    validation_results['data_consistency']['missing_critical_columns'] = missing_critical
    
    # Calculate quality score (0-1)
    quality_score = 1.0
    
    # Penalize for data loss
    if validation_results['data_preservation']['rows_preserved'] < 0.95:
        quality_score -= 0.2
    
    # Penalize for remaining missing values
    missing_ratio = validation_results['quality_improvements']['missing_values_after'] / (len(cleaned_df) * len(cleaned_df.columns))
    quality_score -= missing_ratio * 0.3
    
    # Penalize for data consistency issues
    if validation_results['data_consistency']['negative_followers'] > 0:
        quality_score -= 0.1
    
    if validation_results['data_consistency']['impossible_engagement_rates'] > 0:
        quality_score -= 0.1
    
    if len(validation_results['data_consistency']['missing_critical_columns']) > 0:
        quality_score -= 0.3
    
    validation_results['overall_quality_score'] = max(0.0, quality_score)
    
    return validation_results


# =============================================================================
# EXPORT LIST
# =============================================================================

__all__ = [
    # Core cleaning functions
    'clean_raw_data',
    'fix_data_types_comprehensive',
    'handle_missing_values',
    'handle_outliers_comprehensive',
    'validate_and_fix_ranges',
    'ensure_data_consistency',
    
    # Specialized cleaning
    'clean_engagement_data',
    'clean_posting_data',
    
    # Quality validation
    'validate_cleaning_quality'
]
