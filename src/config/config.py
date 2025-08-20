"""
Creator Growth Navigator - Configuration Module

Central configuration for paths, model parameters, and system settings.
This module provides a single source of truth for all configurable parameters
across the entire project.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

# =============================================================================
# PROJECT STRUCTURE
# =============================================================================

# Base project directory (assumes this file is at src/config/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Core directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim" 
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Code directories
SRC_DIR = BASE_DIR / "src"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
STREAMLIT_DIR = BASE_DIR / "streamlit_app"
TESTS_DIR = BASE_DIR / "tests"

# Output directories
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

# Data generation
DATA_GEN_DIR = BASE_DIR / "datagen"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Default file names
DEFAULT_FILES = {
    "raw_data": "creator_daily_metrics_730d.csv",
    "processed_features": "model_ready_features.csv",
    "weekly_aggregated": "weekly_aggregated.csv",
    "train_data": "train_features.csv",
    "test_data": "test_features.csv",
    "model_predictions": "model_predictions.csv"
}

# Data generation parameters
DATA_GEN_PARAMS = {
    "username": "creator_alpha",
    "start_followers": 120_000,
    "n_days": 730,
    "seed": 123,
    "output_filename": DEFAULT_FILES["raw_data"]
}

# Data validation thresholds
DATA_QUALITY_THRESHOLDS = {
    "max_missing_percentage": 0.05,  # 5% missing values allowed
    "min_data_points": 90,           # Minimum 90 days of data
    "outlier_std_threshold": 3.0,    # 3-sigma outlier detection
    "engagement_rate_max": 0.50,     # Maximum reasonable engagement rate
    "followers_growth_max": 10000    # Maximum daily follower growth
}

# =============================================================================
# MODEL CONFIGURATION  
# =============================================================================

# Core model parameters
MODEL_CONFIG = {
    "model_type": "linear_regression",
    "target_variable": "weekly_growth",
    "primary_feature": "weekly_posting_frequency",
    "validation_split": 0.2,
    "test_split": 0.2,
    "time_window_days": 7,           # Weekly aggregation window
    "confidence_level": 0.95,        # For confidence intervals
    "regularization": None,          # Options: None, 'lasso', 'ridge'
    "cross_validation_folds": 5
}

# Feature engineering parameters
FEATURE_CONFIG = {
    "posting_frequency_window": 7,    # Days for rolling average
    "consistency_window": 7,          # Days for variance calculation  
    "engagement_smoothing": 3,        # Days for engagement smoothing
    "optimal_posting_hours": [8, 12, 18, 21],  # Optimal posting times
    "content_types": ["posts", "reels", "stories"],
    "roi_time_estimates": {           # Minutes per content type
        "posts": 45,
        "reels": 120, 
        "stories": 8
    }
}

# Model performance thresholds
MODEL_QUALITY_THRESHOLDS = {
    "min_r2_score": 0.60,            # Minimum R² for model acceptance
    "max_rmse_ratio": 0.15,          # Max RMSE as % of target mean
    "min_directional_accuracy": 0.75, # Minimum directional prediction accuracy
    "max_prediction_interval_width": 1000  # Max width of 95% CI
}

# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================

# Server settings
STREAMLIT_CONFIG = {
    "port": 8501,
    "address": "0.0.0.0",
    "title": "Creator Growth Navigator",
    "layout": "wide",
    "sidebar_state": "expanded"
}

# UI parameters
UI_CONFIG = {
    "max_posting_frequency": 20,     # Max posts per week in sliders
    "default_posting_frequency": 7,  # Default slider value
    "prediction_history_days": 90,   # Days to show in history charts
    "chart_height": 400,             # Default chart height
    "metrics_precision": 2,          # Decimal places for metrics
    "confidence_interval_opacity": 0.3
}

# Color scheme
COLORS = {
    "primary": "#1f77b4",           # Blue
    "secondary": "#ff7f0e",         # Orange  
    "success": "#2ca02c",           # Green
    "warning": "#d62728",           # Red
    "info": "#9467bd",              # Purple
    "background": "#f8f9fa",        # Light gray
    "text": "#333333"               # Dark gray
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "creator_growth_navigator.log"),
            "formatter": "detailed",
            "level": "DEBUG"
        }
    },
    "loggers": {
        "creator_growth_navigator": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO"
    }
}

# =============================================================================
# DEPLOYMENT CONFIGURATION
# =============================================================================

# Docker settings
DOCKER_CONFIG = {
    "port": STREAMLIT_CONFIG["port"],
    "health_check_endpoint": "/_stcore/health",
    "memory_limit": "2g",
    "cpu_limit": "1.0"
}

# Environment variables
ENV_VARIABLES = {
    "PYTHONPATH": str(SRC_DIR),
    "STREAMLIT_SERVER_PORT": str(STREAMLIT_CONFIG["port"]),
    "STREAMLIT_SERVER_ADDRESS": STREAMLIT_CONFIG["address"]
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_directories() -> None:
    """Create all necessary directories if they don't exist."""
    directories = [
        RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
        MODELS_DIR, LOGS_DIR, REPORTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_data_path(data_type: str, filename: Optional[str] = None) -> Path:
    """Get the full path for a data file.
    
    Args:
        data_type: Type of data ('raw', 'interim', 'processed', 'external')
        filename: Optional filename, uses default if not provided
        
    Returns:
        Full path to the data file
    """
    data_dirs = {
        "raw": RAW_DATA_DIR,
        "interim": INTERIM_DATA_DIR, 
        "processed": PROCESSED_DATA_DIR,
        "external": EXTERNAL_DATA_DIR
    }
    
    if data_type not in data_dirs:
        raise ValueError(f"Invalid data_type. Must be one of: {list(data_dirs.keys())}")
    
    directory = data_dirs[data_type]
    
    if filename is None:
        if data_type == "raw":
            filename = DEFAULT_FILES["raw_data"]
        elif data_type == "processed":
            filename = DEFAULT_FILES["processed_features"]
        else:
            raise ValueError(f"No default filename for data_type: {data_type}")
    
    return directory / filename

def get_config_dict() -> Dict[str, Any]:
    """Return all configuration as a dictionary for easy access."""
    return {
        "data_config": {
            "directories": {
                "base": BASE_DIR,
                "data": DATA_DIR,
                "raw": RAW_DATA_DIR,
                "interim": INTERIM_DATA_DIR,
                "processed": PROCESSED_DATA_DIR,
                "external": EXTERNAL_DATA_DIR
            },
            "files": DEFAULT_FILES,
            "quality_thresholds": DATA_QUALITY_THRESHOLDS
        },
        "model_config": MODEL_CONFIG,
        "feature_config": FEATURE_CONFIG,
        "streamlit_config": STREAMLIT_CONFIG,
        "ui_config": UI_CONFIG,
        "colors": COLORS
    }

# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_project() -> None:
    """Initialize project by creating directories and setting up logging."""
    ensure_directories()
    
    # Set up logging
    import logging.config
    logging.config.dictConfig(LOGGING_CONFIG)
    
    logger = logging.getLogger("creator_growth_navigator")
    logger.info("Project initialized successfully")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Data directory: {DATA_DIR}")

# Auto-initialize when imported
if __name__ == "__main__":
    initialize_project()
    print("✅ Creator Growth Navigator configuration loaded successfully")
    print(f"📁 Base directory: {BASE_DIR}")
    print(f"📊 Data directory: {DATA_DIR}")
    print(f"🏗️  All directories ensured")
else:
    # Ensure directories when imported as module
    ensure_directories()
