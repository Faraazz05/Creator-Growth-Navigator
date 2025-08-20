"""
Creator Growth Navigator - Data Loading Module

Centralized data loading utilities for raw datasets, processed feature sets, 
and external benchmarks with comprehensive error handling, validation, and logging.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple, Any
import json
from datetime import datetime

# Import from project modules
from src.config.config import (
    RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
    DEFAULT_FILES, DATA_QUALITY_THRESHOLDS, get_data_path
)
from src.utils.logger import get_data_logger
from src.utils.helpers import (
    validate_dataframe_schema, validate_data_quality, 
    parse_date_string, create_date_features
)

logger = get_data_logger()


# =============================================================================
# RAW DATA LOADING
# =============================================================================

def load_raw_data(filepath: Optional[Union[str, Path]] = None,
                 validate: bool = True,
                 parse_dates: bool = True,
                 date_column: str = 'date') -> Optional[pd.DataFrame]:
    """Load raw creator data from CSV file with validation and preprocessing.
    
    Args:
        filepath: Optional path to raw data file. Uses default if None.
        validate: Whether to perform data quality validation
        parse_dates: Whether to parse date columns
        date_column: Name of the date column to parse
        
    Returns:
        Loaded and validated dataframe or None if failed
    """
    try:
        # Use default path if not provided
        if filepath is None:
            filepath = get_data_path('raw', DEFAULT_FILES['raw_data'])
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"Raw data file not found: {filepath}")
            return None
        
        logger.info(f"Loading raw data from: {filepath}")
        
        # Load CSV with error handling
        df = pd.read_csv(filepath)
        logger.log_data_load(str(filepath), len(df), len(df.columns))
        
        # Parse dates if requested
        if parse_dates and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], format='%d-%m-%Y', errors='coerce')
            date_errors = df[date_column].isna().sum()
            if date_errors > 0:
                logger.warning(f"Could not parse {date_errors} date values in column '{date_column}'")
        
        # Validate data quality
        if validate:
            validation_results = validate_raw_data_schema(df)
            if not validation_results['is_valid']:
                logger.log_data_validation(len(df), 0, validation_results)
                return None
            else:
                logger.log_data_validation(len(df), len(df), None)
        
        logger.info(f"Successfully loaded raw data: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except pd.errors.EmptyDataError:
        logger.error(f"Empty CSV file: {filepath}")
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV file {filepath}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error loading raw data from {filepath}: {str(e)}")
    
    return None


def validate_raw_data_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate raw data schema and quality.
    
    Args:
        df: Raw data dataframe to validate
        
    Returns:
        Dictionary with validation results
    """
    # Required columns for raw creator data
    required_columns = [
        'username', 'date', 'followers', 'growth', 'posts', 'reels', 'stories',
        'engagement_rate', 'reach', 'post_likes', 'comments', 'saves', 'shares'
    ]
    
    # Expected data types
    expected_types = {
        'followers': 'int',
        'growth': 'int', 
        'posts': 'int',
        'reels': 'int',
        'stories': 'int',
        'engagement_rate': 'float',
        'reach': 'int'
    }
    
    # Validate schema
    schema_valid, schema_issues = validate_dataframe_schema(df, required_columns, expected_types)
    
    # Validate data quality using configured thresholds
    quality_results = validate_data_quality(df, DATA_QUALITY_THRESHOLDS)
    
    # Combine results
    all_issues = schema_issues + quality_results['issues']
    
    return {
        'is_valid': len(all_issues) == 0,
        'total_rows': len(df),
        'issues': all_issues,
        'schema_valid': schema_valid,
        'quality_valid': quality_results['is_valid']
    }


# =============================================================================
# PROCESSED DATA LOADING
# =============================================================================

def load_processed_data(filepath: Optional[Union[str, Path]] = None,
                       feature_set: str = 'full') -> Optional[pd.DataFrame]:
    """Load processed feature-engineered data.
    
    Args:
        filepath: Optional path to processed data file
        feature_set: Which feature set to load ('full', 'core', 'weekly')
        
    Returns:
        Loaded dataframe or None if failed
    """
    try:
        # Determine filepath based on feature set
        if filepath is None:
            if feature_set == 'weekly':
                filepath = get_data_path('processed', DEFAULT_FILES['weekly_aggregated'])
            elif feature_set == 'core':
                filepath = get_data_path('processed', 'core_features.csv')
            else:
                filepath = get_data_path('processed', DEFAULT_FILES['processed_features'])
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"Processed data file not found: {filepath}")
            return None
        
        logger.info(f"Loading processed data from: {filepath}")
        
        # Load data
        df = pd.read_csv(filepath)
        logger.log_data_load(str(filepath), len(df), len(df.columns))
        
        # Parse date columns if present
        date_columns = ['date', 'week_start', 'week_end']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Validate processed data
        validation_results = validate_processed_data_schema(df, feature_set)
        if not validation_results['is_valid']:
            logger.log_data_validation(len(df), 0, validation_results)
            logger.warning("Processed data validation failed, but returning data anyway")
        
        logger.info(f"Successfully loaded processed data: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error loading processed data from {filepath}: {str(e)}")
        return None


def validate_processed_data_schema(df: pd.DataFrame, feature_set: str) -> Dict[str, Any]:
    """Validate processed data schema.
    
    Args:
        df: Processed data dataframe
        feature_set: Expected feature set type
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    
    # Core features that should always be present
    core_features = ['weekly_posting_frequency', 'weekly_growth']
    
    if feature_set == 'weekly':
        required_columns = core_features + [
            'week_start', 'week_end', 'total_posts', 'total_reels', 'total_stories',
            'avg_engagement_rate', 'total_reach'
        ]
    elif feature_set == 'core':
        required_columns = core_features + [
            'share_posts', 'share_reels', 'share_stories', 'consistency_variance'
        ]
    else:  # 'full'
        required_columns = core_features + [
            'share_posts', 'share_reels', 'share_stories', 'consistency_variance',
            'roi_follows_per_hour', 'engagement_rate_smoothed', 'posting_time_optimal'
        ]
    
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        issues.append(f"Missing required columns for {feature_set} feature set: {list(missing_columns)}")
    
    # Check for target variable
    if 'weekly_growth' not in df.columns:
        issues.append("Target variable 'weekly_growth' not found")
    
    # Check data types
    numeric_columns = [col for col in required_columns if col in df.columns and col not in ['week_start', 'week_end']]
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Column '{col}' should be numeric but is {df[col].dtype}")
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'feature_set': feature_set,
        'total_rows': len(df)
    }


# =============================================================================
# EXTERNAL DATA LOADING
# =============================================================================

def load_external_data(filepath: Optional[Union[str, Path]] = None,
                      data_type: str = 'benchmark') -> Optional[pd.DataFrame]:
    """Load external benchmark or reference data.
    
    Args:
        filepath: Optional path to external data file
        data_type: Type of external data ('benchmark', 'competitor', 'industry')
        
    Returns:
        Loaded dataframe or None if failed
    """
    try:
        if filepath is None:
            # Default external data files based on type
            filename_map = {
                'benchmark': 'instagram_benchmarks.csv',
                'competitor': 'competitor_data.csv', 
                'industry': 'industry_averages.csv'
            }
            filename = filename_map.get(data_type, 'external_data.csv')
            filepath = get_data_path('external', filename)
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"External data file not found: {filepath}")
            return None
        
        logger.info(f"Loading external data from: {filepath}")
        
        # Load data with flexible parsing
        df = pd.read_csv(filepath)
        logger.log_data_load(str(filepath), len(df), len(df.columns))
        
        logger.info(f"Successfully loaded external data: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error loading external data from {filepath}: {str(e)}")
        return None


# =============================================================================
# TRAIN/TEST DATA LOADING
# =============================================================================

def load_train_test_data(train_filepath: Optional[Union[str, Path]] = None,
                        test_filepath: Optional[Union[str, Path]] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load training and testing datasets.
    
    Args:
        train_filepath: Path to training data file
        test_filepath: Path to testing data file
        
    Returns:
        Tuple of (train_df, test_df) or (None, None) if failed
    """
    try:
        # Use default paths if not provided
        if train_filepath is None:
            train_filepath = get_data_path('processed', DEFAULT_FILES['train_data'])
        if test_filepath is None:
            test_filepath = get_data_path('processed', DEFAULT_FILES['test_data'])
        
        train_df = load_processed_data(train_filepath)
        test_df = load_processed_data(test_filepath)
        
        if train_df is not None and test_df is not None:
            logger.info(f"Loaded train/test data: {len(train_df)} train, {len(test_df)} test samples")
        
        return train_df, test_df
        
    except Exception as e:
        logger.error(f"Error loading train/test data: {str(e)}")
        return None, None


# =============================================================================
# DATA SAVING FUNCTIONS
# =============================================================================

def save_dataframe(df: pd.DataFrame,
                  filepath: Union[str, Path],
                  data_type: str = 'processed',
                  index: bool = False,
                  backup: bool = True) -> bool:
    """Save dataframe to file with comprehensive error handling.
    
    Args:
        df: Dataframe to save
        filepath: Output file path
        data_type: Type of data being saved ('raw', 'interim', 'processed', 'external')
        index: Whether to write row index
        backup: Whether to create backup of existing file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if file exists and backup is requested
        if backup and filepath.exists():
            backup_path = filepath.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            filepath.rename(backup_path)
            logger.info(f"Created backup: {backup_path}")
        
        # Save dataframe
        df.to_csv(filepath, index=index)
        logger.log_data_save(str(filepath), len(df))
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save dataframe to {filepath}: {str(e)}")
        return False


def save_train_test_split(train_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         train_filepath: Optional[Union[str, Path]] = None,
                         test_filepath: Optional[Union[str, Path]] = None) -> bool:
    """Save training and testing datasets.
    
    Args:
        train_df: Training dataframe
        test_df: Testing dataframe
        train_filepath: Path for training data file
        test_filepath: Path for testing data file
        
    Returns:
        True if both saves successful, False otherwise
    """
    try:
        # Use default paths if not provided
        if train_filepath is None:
            train_filepath = get_data_path('processed', DEFAULT_FILES['train_data'])
        if test_filepath is None:
            test_filepath = get_data_path('processed', DEFAULT_FILES['test_data'])
        
        train_success = save_dataframe(train_df, train_filepath, 'processed')
        test_success = save_dataframe(test_df, test_filepath, 'processed')
        
        if train_success and test_success:
            logger.info(f"Saved train/test split: {len(train_df)} train, {len(test_df)} test samples")
            return True
        else:
            logger.error("Failed to save train/test split")
            return False
            
    except Exception as e:
        logger.error(f"Error saving train/test split: {str(e)}")
        return False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive summary of dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with data summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'date_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
    }
    
    # Add statistics for numeric columns
    if summary['numeric_columns']:
        numeric_stats = df[summary['numeric_columns']].describe().to_dict()
        summary['numeric_statistics'] = numeric_stats
    
    return summary


def list_available_datasets() -> Dict[str, List[str]]:
    """List all available datasets in each data directory.
    
    Returns:
        Dictionary mapping data types to lists of available files
    """
    datasets = {}
    
    data_dirs = {
        'raw': RAW_DATA_DIR,
        'interim': INTERIM_DATA_DIR,
        'processed': PROCESSED_DATA_DIR,
        'external': EXTERNAL_DATA_DIR
    }
    
    for data_type, directory in data_dirs.items():
        if directory.exists():
            csv_files = [f.name for f in directory.glob('*.csv')]
            datasets[data_type] = sorted(csv_files)
        else:
            datasets[data_type] = []
    
    return datasets


def validate_data_integrity(df: pd.DataFrame,
                          reference_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Validate data integrity and consistency.
    
    Args:
        df: Dataframe to validate
        reference_df: Optional reference dataframe for comparison
        
    Returns:
        Dictionary with integrity check results
    """
    results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'issues': []
    }
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        results['issues'].append(f"Found {duplicates} duplicate rows")
    
    # Check for completely empty rows
    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows > 0:
        results['issues'].append(f"Found {empty_rows} completely empty rows")
    
    # Check for negative values in columns that should be positive
    positive_columns = ['followers', 'posts', 'reels', 'stories', 'reach', 'post_likes']
    for col in positive_columns:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                results['issues'].append(f"Found {negative_count} negative values in '{col}'")
    
    # Check date consistency if date column exists
    if 'date' in df.columns:
        try:
            df_sorted = df.sort_values('date')
            date_gaps = df_sorted['date'].diff().dt.days
            large_gaps = (date_gaps > 7).sum()  # Gaps larger than a week
            if large_gaps > 0:
                results['issues'].append(f"Found {large_gaps} date gaps larger than 7 days")
        except:
            results['issues'].append("Could not validate date consistency")
    
    # Compare with reference if provided
    if reference_df is not None:
        if len(df) != len(reference_df):
            results['issues'].append(f"Row count mismatch: {len(df)} vs {len(reference_df)} in reference")
        
        common_columns = set(df.columns) & set(reference_df.columns)
        if len(common_columns) == 0:
            results['issues'].append("No common columns with reference dataframe")
    
    results['is_valid'] = len(results['issues']) == 0
    return results


# =============================================================================
# EXPORT LIST
# =============================================================================

__all__ = [
    # Loading functions
    'load_raw_data',
    'load_processed_data', 
    'load_external_data',
    'load_train_test_data',
    
    # Validation functions
    'validate_raw_data_schema',
    'validate_processed_data_schema',
    'validate_data_integrity',
    
    # Saving functions
    'save_dataframe',
    'save_train_test_split',
    
    # Utility functions
    'get_data_summary',
    'list_available_datasets'
]
