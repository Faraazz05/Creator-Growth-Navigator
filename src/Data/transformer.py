"""
Creator Growth Navigator - Data Transformation Module

Feature engineering and data transformation utilities for converting raw creator data
into model-ready features with weekly aggregation, content mix analysis, and derived metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import warnings

# Import from project modules
from src.config.config import FEATURE_CONFIG, MODEL_CONFIG
from src.utils.logger import get_data_logger
from src.utils.helpers import (
    safe_divide, rolling_aggregation, calculate_percentage_change,
    get_week_boundaries, create_date_features, calculate_growth_metrics
)

logger = get_data_logger()
warnings.filterwarnings('ignore')


# =============================================================================
# CORE TRANSFORMATION PIPELINE
# =============================================================================

def transform_to_model_ready(df: pd.DataFrame,
                           target_frequency: str = 'weekly',
                           include_advanced_features: bool = True) -> pd.DataFrame:
    """Transform cleaned daily data into model-ready features.
    
    Args:
        df: Cleaned daily creator data
        target_frequency: Target aggregation frequency ('weekly', 'daily')
        include_advanced_features: Whether to include advanced engineered features
        
    Returns:
        Model-ready dataframe with engineered features
    """
    logger.info(f"Starting data transformation to {target_frequency} frequency")
    
    if target_frequency == 'weekly':
        # Weekly aggregation pipeline
        transformed_df = create_weekly_features(df)
    else:
        # Daily features (for advanced modeling)
        transformed_df = create_daily_features(df)
    
    # Add advanced features if requested
    if include_advanced_features:
        transformed_df = add_advanced_features(transformed_df)
        transformed_df = add_interaction_features(transformed_df)
        transformed_df = add_temporal_features(transformed_df)
    
    # Final feature validation and cleanup
    transformed_df = finalize_features(transformed_df)
    
    logger.log_feature_engineering(
        input_features=len(df.columns),
        output_features=len(transformed_df.columns),
        transformation=f"{target_frequency}_aggregation_with_features"
    )
    
    return transformed_df


# =============================================================================
# WEEKLY AGGREGATION FEATURES
# =============================================================================

def create_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create weekly aggregated features from daily data.
    
    Args:
        df: Daily creator data
        
    Returns:
        Weekly aggregated dataframe with features
    """
    logger.info("Creating weekly aggregated features")
    
    # Ensure data is sorted by date
    if 'date' in df.columns:
        df = df.sort_values('date').copy()
        df['date'] = pd.to_datetime(df['date'])
    else:
        raise ValueError("Date column required for weekly aggregation")
    
    # Create week grouping
    df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='d')
    df['week_end'] = df['week_start'] + pd.Timedelta(days=6)
    
    # Define aggregation functions for each metric type
    weekly_agg = {
        # Posting activity (sum for weekly totals)
        'posts': 'sum',
        'reels': 'sum', 
        'stories': 'sum',
        'ads_posted': 'sum',
        'hashtags_used': 'sum',
        'minutes_spent': 'sum',
        
        # Engagement metrics (sum for weekly totals)
        'post_likes': 'sum',
        'reel_plays': 'sum',
        'reel_engagement': 'sum',
        'story_reach': 'sum',
        'story_engagement': 'sum',
        'comments': 'sum',
        'saves': 'sum',
        'shares': 'sum',
        'reach': 'sum',
        'profile_visits': 'sum',
        
        # Growth metrics (sum for weekly changes)
        'growth': 'sum',
        'follows': 'sum',
        'unfollows': 'sum',
        
        # Rates and ratios (mean for weekly averages)
        'engagement_rate': 'mean',
        'avg_hashtag_count': 'mean',
        'roi_follows_per_hour': 'mean',
        
        # End-of-week snapshots (last value)
        'followers': 'last',
        
        # Categorical (most frequent)
        'username': 'first',
        'post_time': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc
    }
    
    # Perform weekly aggregation
    weekly_df = df.groupby('week_start').agg(weekly_agg).reset_index()
    
    # Add week_end for completeness
    weekly_df['week_end'] = weekly_df['week_start'] + pd.Timedelta(days=6)
    
    # Create primary feature: weekly posting frequency
    weekly_df['weekly_posting_frequency'] = (
        weekly_df['posts'] + 
        weekly_df['reels'] + 
        weekly_df['stories']
    )
    
    # Create target variable: weekly growth
    weekly_df['weekly_growth'] = weekly_df['growth']
    
    # Create content mix features
    weekly_df = add_content_mix_features(weekly_df)
    
    # Create consistency and quality features
    weekly_df = add_consistency_features(weekly_df, df)
    
    # Create ROI and efficiency features  
    weekly_df = add_roi_features(weekly_df)
    
    logger.info(f"Created weekly features: {len(weekly_df)} weeks, {len(weekly_df.columns)} features")
    return weekly_df


def add_content_mix_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add content mix ratio features.
    
    Args:
        df: Weekly aggregated dataframe
        
    Returns:
        Dataframe with content mix features
    """
    # Calculate total content for ratio calculations
    df['total_content'] = df['posts'] + df['reels'] + df['stories']
    
    # Content mix ratios (shares that sum to 1)
    df['share_posts'] = safe_divide(df['posts'], df['total_content'])
    df['share_reels'] = safe_divide(df['reels'], df['total_content']) 
    df['share_stories'] = safe_divide(df['stories'], df['total_content'])
    
    # Content diversity (entropy-based measure)
    def calculate_content_diversity(row):
        shares = [row['share_posts'], row['share_reels'], row['share_stories']]
        # Remove zero shares for entropy calculation
        shares = [s for s in shares if s > 0]
        if len(shares) <= 1:
            return 0.0
        # Calculate Shannon entropy
        entropy = -sum(s * np.log(s) for s in shares)
        # Normalize to 0-1 scale
        max_entropy = np.log(len(shares))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    df['content_diversity'] = df.apply(calculate_content_diversity, axis=1)
    
    # Content type focus (which type dominates)
    df['dominant_content_type'] = df[['share_posts', 'share_reels', 'share_stories']].idxmax(axis=1)
    df['dominant_content_type'] = df['dominant_content_type'].str.replace('share_', '')
    
    # Reels focus (important for Instagram algorithm)
    df['reels_focused'] = (df['share_reels'] > 0.5).astype(int)
    
    return df


def add_consistency_features(weekly_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add posting consistency features.
    
    Args:
        weekly_df: Weekly aggregated data
        daily_df: Original daily data for consistency calculations
        
    Returns:
        Weekly dataframe with consistency features
    """
    # Calculate posting consistency for each week
    consistency_features = []
    
    for _, week_row in weekly_df.iterrows():
        week_start = week_row['week_start']
        week_end = week_start + pd.Timedelta(days=6)
        
        # Get daily data for this week
        week_mask = (daily_df['date'] >= week_start) & (daily_df['date'] <= week_end)
        week_daily = daily_df[week_mask]
        
        if len(week_daily) == 0:
            consistency_features.append({
                'posting_consistency_variance': 0.0,
                'posting_consistency_days': 0,
                'max_gap_days': 7,
                'posting_regularity_score': 0.0
            })
            continue
        
        # Calculate daily posting totals
        week_daily['daily_total_posts'] = (
            week_daily['posts'] + 
            week_daily['reels'] + 
            week_daily['stories']
        )
        
        # Posting variance (lower = more consistent)
        posting_variance = week_daily['daily_total_posts'].var()
        
        # Days with posts
        posting_days = (week_daily['daily_total_posts'] > 0).sum()
        
        # Maximum gap between posts
        posting_days_indices = week_daily[week_daily['daily_total_posts'] > 0].index
        if len(posting_days_indices) > 1:
            gaps = np.diff(posting_days_indices)
            max_gap = gaps.max() if len(gaps) > 0 else 0
        else:
            max_gap = 7
        
        # Posting regularity score (0-1, higher = more regular)
        regularity_score = posting_days / 7.0 * (1 - min(max_gap / 7.0, 1.0))
        
        consistency_features.append({
            'posting_consistency_variance': posting_variance,
            'posting_consistency_days': posting_days,
            'max_gap_days': max_gap,
            'posting_regularity_score': regularity_score
        })
    
    # Add consistency features to weekly dataframe
    consistency_df = pd.DataFrame(consistency_features)
    weekly_df = pd.concat([weekly_df, consistency_df], axis=1)
    
    return weekly_df


def add_roi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ROI and efficiency features.
    
    Args:
        df: Weekly dataframe
        
    Returns:
        Dataframe with ROI features
    """
    # Time investment efficiency
    df['follows_per_minute'] = safe_divide(df['follows'], df['minutes_spent'])
    df['engagement_per_minute'] = safe_divide(
        df['post_likes'] + df['comments'] + df['saves'] + df['shares'],
        df['minutes_spent']
    )
    
    # Content efficiency
    df['follows_per_post'] = safe_divide(df['follows'], df['weekly_posting_frequency'])
    df['engagement_per_post'] = safe_divide(
        df['post_likes'] + df['comments'] + df['saves'] + df['shares'],
        df['weekly_posting_frequency']
    )
    
    # Reach efficiency
    df['reach_per_post'] = safe_divide(df['reach'], df['weekly_posting_frequency'])
    df['reach_efficiency'] = safe_divide(df['follows'], df['reach'])
    
    # Profile visit conversion
    df['profile_visit_conversion'] = safe_divide(df['follows'], df['profile_visits'])
    
    # Overall ROI score (composite metric)
    df['roi_score'] = (
        0.4 * df['roi_follows_per_hour'] / df['roi_follows_per_hour'].max() +
        0.3 * df['follows_per_post'] / df['follows_per_post'].max() +
        0.3 * df['reach_efficiency'] / df['reach_efficiency'].max()
    ).fillna(0)
    
    return df


# =============================================================================
# DAILY FEATURES (FOR ADVANCED MODELING)
# =============================================================================

def create_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create daily features with rolling statistics.
    
    Args:
        df: Daily creator data
        
    Returns:
        Daily dataframe with engineered features
    """
    logger.info("Creating daily features with rolling statistics")
    
    df_features = df.copy()
    
    # Create basic posting features
    df_features['daily_posting_frequency'] = (
        df_features['posts'] + 
        df_features['reels'] + 
        df_features['stories']
    )
    
    # Rolling features (7-day windows)
    rolling_features = {
        'daily_posting_frequency': ['mean', 'std', 'sum'],
        'engagement_rate': ['mean', 'std'],
        'growth': ['mean', 'sum'],
        'followers': ['mean'],
        'reach': ['sum', 'mean']
    }
    
    for column, agg_funcs in rolling_features.items():
        if column in df_features.columns:
            for agg_func in agg_funcs:
                if agg_func == 'mean':
                    df_features[f'{column}_7d_avg'] = df_features[column].rolling(7).mean()
                elif agg_func == 'std':
                    df_features[f'{column}_7d_std'] = df_features[column].rolling(7).std()
                elif agg_func == 'sum':
                    df_features[f'{column}_7d_sum'] = df_features[column].rolling(7).sum()
    
    # Content mix features (daily)
    df_features = add_content_mix_features(df_features)
    
    # Growth momentum features
    df_features['growth_momentum'] = df_features['growth'].rolling(3).mean()
    df_features['growth_acceleration'] = df_features['growth'].diff()
    
    # Engagement trends
    df_features['engagement_trend'] = df_features['engagement_rate'].rolling(7).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else 0
    )
    
    return df_features


# =============================================================================
# ADVANCED FEATURES
# =============================================================================

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced engineered features.
    
    Args:
        df: Base feature dataframe
        
    Returns:
        Dataframe with advanced features
    """
    logger.debug("Adding advanced engineered features")
    
    # Engagement quality features
    df['engagement_quality_score'] = calculate_engagement_quality(df)
    df['audience_quality_score'] = calculate_audience_quality(df)
    
    # Growth efficiency features  
    df['growth_efficiency'] = safe_divide(df['weekly_growth'], df['weekly_posting_frequency'])
    df['sustainable_growth_indicator'] = (
        (df['growth_efficiency'] > df['growth_efficiency'].quantile(0.7)) & 
        (df['posting_regularity_score'] > 0.5)
    ).astype(int)
    
    # Content performance tiers
    df['posting_tier'] = pd.cut(
        df['weekly_posting_frequency'],
        bins=[0, 3, 7, 14, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # Engagement performance tiers
    df['engagement_tier'] = pd.cut(
        df['engagement_rate'],
        bins=[0, 0.02, 0.05, 0.10, float('inf')],
        labels=['Low', 'Medium', 'High', 'Exceptional']
    )
    
    # Saturation indicators
    df['potential_saturation'] = detect_saturation_signals(df)
    
    return df


def calculate_engagement_quality(df: pd.DataFrame) -> pd.Series:
    """Calculate engagement quality score based on interaction types.
    
    Args:
        df: Dataframe with engagement metrics
        
    Returns:
        Series with engagement quality scores
    """
    # Weight different engagement types by value
    engagement_weights = {
        'saves': 3.0,      # Highest value - intent to revisit
        'shares': 2.5,     # High value - amplification
        'comments': 2.0,   # Medium-high value - conversation
        'post_likes': 1.0  # Base value - passive engagement
    }
    
    # Calculate weighted engagement score
    weighted_engagement = 0
    total_weights = 0
    
    for metric, weight in engagement_weights.items():
        if metric in df.columns:
            weighted_engagement += df[metric] * weight
            total_weights += weight
    
    # Normalize by reach to get quality rate
    engagement_quality = safe_divide(weighted_engagement, df.get('reach', 1))
    
    # Normalize to 0-1 scale (cap at 95th percentile)
    max_quality = engagement_quality.quantile(0.95)
    if max_quality > 0:
        engagement_quality = (engagement_quality / max_quality).clip(0, 1)
    
    return engagement_quality


def calculate_audience_quality(df: pd.DataFrame) -> pd.Series:
    """Calculate audience quality based on conversion rates.
    
    Args:
        df: Dataframe with audience metrics
        
    Returns:
        Series with audience quality scores
    """
    # Profile visit to follow conversion rate
    visit_conversion = safe_divide(df.get('follows', 0), df.get('profile_visits', 1))
    
    # Reach to profile visit rate
    reach_to_visit = safe_divide(df.get('profile_visits', 0), df.get('reach', 1))
    
    # Overall conversion funnel efficiency
    overall_conversion = safe_divide(df.get('follows', 0), df.get('reach', 1))
    
    # Composite audience quality (0-1 scale)
    audience_quality = (
        0.4 * visit_conversion + 
        0.3 * reach_to_visit + 
        0.3 * overall_conversion
    )
    
    # Normalize to 0-1 scale
    max_quality = audience_quality.quantile(0.95)
    if max_quality > 0:
        audience_quality = (audience_quality / max_quality).clip(0, 1)
    
    return audience_quality


def detect_saturation_signals(df: pd.DataFrame) -> pd.Series:
    """Detect potential saturation in posting frequency.
    
    Args:
        df: Dataframe with posting and engagement metrics
        
    Returns:
        Series with saturation flags (0 or 1)
    """
    saturation_flags = pd.Series(0, index=df.index)
    
    # High posting with declining engagement
    high_posting = df['weekly_posting_frequency'] > df['weekly_posting_frequency'].quantile(0.8)
    low_engagement = df['engagement_rate'] < df['engagement_rate'].quantile(0.3)
    
    # Declining growth efficiency
    low_efficiency = df.get('growth_efficiency', 0) < df.get('growth_efficiency', pd.Series([0])).quantile(0.3)
    
    # High time investment with low ROI
    high_time = df.get('minutes_spent', 0) > df.get('minutes_spent', pd.Series([0])).quantile(0.8)
    low_roi = df.get('roi_follows_per_hour', 0) < df.get('roi_follows_per_hour', pd.Series([0])).quantile(0.3)
    
    # Combine saturation signals
    saturation_flags = (
        (high_posting & low_engagement) |
        low_efficiency |
        (high_time & low_roi)
    ).astype(int)
    
    return saturation_flags


# =============================================================================
# INTERACTION FEATURES
# =============================================================================

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between main predictors.
    
    Args:
        df: Base feature dataframe
        
    Returns:
        Dataframe with interaction features
    """
    logger.debug("Adding interaction features")
    
    # Posting frequency × Content mix interactions
    df['posting_freq_x_reels_share'] = df['weekly_posting_frequency'] * df['share_reels']
    df['posting_freq_x_posts_share'] = df['weekly_posting_frequency'] * df['share_posts']
    df['posting_freq_x_stories_share'] = df['weekly_posting_frequency'] * df['share_stories']
    
    # Posting frequency × Consistency interactions
    df['posting_freq_x_regularity'] = df['weekly_posting_frequency'] * df['posting_regularity_score']
    df['posting_freq_x_consistency'] = df['weekly_posting_frequency'] * (1 / (1 + df['posting_consistency_variance']))
    
    # Content mix × Engagement interactions
    df['reels_share_x_engagement'] = df['share_reels'] * df['engagement_rate']
    df['posts_share_x_engagement'] = df['share_posts'] * df['engagement_rate']
    
    # ROI × Posting interactions
    df['roi_x_posting_freq'] = df['roi_follows_per_hour'] * df['weekly_posting_frequency']
    
    return df


# =============================================================================
# TEMPORAL FEATURES
# =============================================================================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal and seasonal features.
    
    Args:
        df: Dataframe with date information
        
    Returns:
        Dataframe with temporal features
    """
    logger.debug("Adding temporal features")
    
    # Use week_start for weekly data or date for daily data
    date_col = 'week_start' if 'week_start' in df.columns else 'date'
    
    if date_col in df.columns:
        df = create_date_features(df, date_col)
        
        # Week-specific features for creator content
        df['is_beginning_of_month'] = (df['day_of_month'] <= 7).astype(int)
        df['is_end_of_month'] = (df['day_of_month'] >= 24).astype(int)
        
        # Holiday/seasonal patterns (approximate)
        df['is_holiday_season'] = df['month'].isin([11, 12, 1]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_back_to_school'] = df['month'].isin([8, 9]).astype(int)
        
        # Engagement cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # Time-based lags and trends
    if 'weekly_growth' in df.columns:
        # Previous week's performance
        df['prev_week_growth'] = df['weekly_growth'].shift(1)
        df['prev_week_posting'] = df['weekly_posting_frequency'].shift(1)
        df['prev_week_engagement'] = df['engagement_rate'].shift(1)
        
        # Growth momentum (3-week rolling)
        df['growth_momentum_3w'] = df['weekly_growth'].rolling(3).mean()
        
        # Performance trend (slope over last 4 weeks)
        df['performance_trend'] = df['weekly_growth'].rolling(4).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 4 else 0
        )
    
    return df


# =============================================================================
# FEATURE FINALIZATION
# =============================================================================

def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Final feature selection and preparation for modeling.
    
    Args:
        df: Dataframe with all engineered features
        
    Returns:
        Final model-ready dataframe
    """
    logger.debug("Finalizing features for modeling")
    
    # Remove infinite and extremely large values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        # Replace infinity with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Cap extremely large values (beyond 99.9th percentile)
        if df[col].notna().sum() > 0:
            upper_cap = df[col].quantile(0.999)
            lower_cap = df[col].quantile(0.001)
            df[col] = df[col].clip(lower_cap, upper_cap)
    
    # Fill any remaining NaN values with appropriate defaults
    for col in numeric_columns:
        if df[col].isna().sum() > 0:
            if 'rate' in col.lower() or 'ratio' in col.lower() or 'share' in col.lower():
                df[col] = df[col].fillna(0.0)
            elif 'score' in col.lower():
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = df[col].fillna(df[col].median())
    
    # Round numeric features to reasonable precision
    for col in numeric_columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].round(6)
    
    # Ensure target variable exists and is properly named
    if 'weekly_growth' not in df.columns and 'growth' in df.columns:
        df['weekly_growth'] = df['growth']
    
    # Create feature importance hints for model interpretation
    df.attrs['feature_categories'] = {
        'primary_features': ['weekly_posting_frequency'],
        'content_mix': ['share_posts', 'share_reels', 'share_stories', 'content_diversity'],
        'consistency': ['posting_regularity_score', 'posting_consistency_variance'],
        'engagement': ['engagement_rate', 'engagement_quality_score'],
        'roi_metrics': ['roi_follows_per_hour', 'growth_efficiency'],
        'temporal': ['month', 'quarter', 'is_weekend'],
        'interaction': [col for col in df.columns if '_x_' in col],
        'target': ['weekly_growth']
    }
    
    logger.info(f"Finalized features: {len(df.columns)} total features ready for modeling")
    return df


def get_feature_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive summary of engineered features.
    
    Args:
        df: Feature-engineered dataframe
        
    Returns:
        Dictionary with feature summary
    """
    summary = {
        'total_features': len(df.columns),
        'total_samples': len(df),
        'feature_types': {},
        'missing_values': df.isnull().sum().sum(),
        'feature_categories': df.attrs.get('feature_categories', {}),
        'date_range': None
    }
    
    # Analyze feature types
    for dtype in df.dtypes.value_counts().index:
        summary['feature_types'][str(dtype)] = df.dtypes.value_counts()[dtype]
    
    # Date range analysis
    date_cols = ['date', 'week_start', 'week_end']
    for col in date_cols:
        if col in df.columns:
            summary['date_range'] = {
                'start': df[col].min(),
                'end': df[col].max(),
                'periods': len(df)
            }
            break
    
    # Feature statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_features'] = len(numeric_cols)
        summary['feature_statistics'] = {
            'mean_values': df[numeric_cols].mean().to_dict(),
            'std_values': df[numeric_cols].std().to_dict()
        }
    
    return summary


# =============================================================================
# EXPORT LIST
# =============================================================================

__all__ = [
    # Core transformation
    'transform_to_model_ready',
    'create_weekly_features',
    'create_daily_features',
    
    # Feature engineering components
    'add_content_mix_features',
    'add_consistency_features', 
    'add_roi_features',
    'add_advanced_features',
    'add_interaction_features',
    'add_temporal_features',
    
    # Utility functions
    'finalize_features',
    'get_feature_summary',
    'calculate_engagement_quality',
    'calculate_audience_quality',
    'detect_saturation_signals'
]
