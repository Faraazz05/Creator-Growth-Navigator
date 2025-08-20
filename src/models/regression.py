"""
Creator Growth Navigator - Regression Models Module

Linear regression models for predicting follower growth based on posting frequency
and contextual features, with comprehensive diagnostics and interpretability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import joblib
from datetime import datetime
import warnings

# Import from project modules
from src.config.config import MODEL_CONFIG, FEATURE_CONFIG, MODEL_QUALITY_THRESHOLDS
from src.utils.logger import get_model_logger
from src.utils.helpers import safe_divide, calculate_confidence_interval

logger = get_model_logger()
warnings.filterwarnings('ignore')


# =============================================================================
# CORE LINEAR REGRESSION MODEL
# =============================================================================

class CreatorGrowthModel:
    """Main linear regression model for creator follower growth prediction."""
    
    def __init__(self, 
                 model_type: str = 'linear',
                 regularization: Optional[str] = None,
                 random_state: int = 42):
        """Initialize the Creator Growth Model.
        
        Args:
            model_type: Type of regression ('linear', 'ridge', 'lasso')
            regularization: Regularization parameter (None, 'ridge', 'lasso')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.regularization = regularization
        self.random_state = random_state
        
        # Initialize model based on type
        if model_type == 'ridge' or regularization == 'ridge':
            self.model = Ridge(alpha=1.0, random_state=random_state)
        elif model_type == 'lasso' or regularization == 'lasso':
            self.model = Lasso(alpha=1.0, random_state=random_state)
        else:
            self.model = LinearRegression()
        
        # Model components
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.training_stats = {}
        
        logger.info(f"Initialized {model_type} regression model")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target from your specific data structure.
        
        Args:
            df: DataFrame with your exact column structure
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Preparing features from creator data")
        
        # Create primary feature: weekly posting frequency
        if 'weekly_posting_frequency' not in df.columns:
            df['weekly_posting_frequency'] = df['posts'] + df['reels'] + df['stories']
        
        # Define feature sets based on your data structure
        core_features = [
            'weekly_posting_frequency',  # Primary predictor
        ]
        
        content_mix_features = [
            'share_posts', 'share_reels', 'share_stories'  # Already calculated in your data
        ]
        
        engagement_features = [
            'engagement_rate',  # Your calculated engagement rate
            'avg_hashtag_count'  # Your calculated hashtag average
        ]
        
        consistency_features = [
            'post_consistency_variance_7d',  # Your calculated variance
            'posted_in_optimal_window'  # Your binary timing feature
        ]
        
        roi_features = [
            'roi_follows_per_hour',  # Your calculated ROI
            'minutes_spent'  # Time investment
        ]
        
        temporal_features = [
            'month', 'quarter'  # Your seasonal features
        ]
        
        saturation_features = [
            'saturation_flag'  # Your calculated saturation indicator
        ]
        
        # Combine all features
        feature_columns = (
            core_features + content_mix_features + engagement_features + 
            consistency_features + roi_features + temporal_features + saturation_features
        )
        
        # Filter to only existing columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Prepare feature matrix
        X = df[available_features].values
        
        # Target variable (growth)
        if 'weekly_growth' in df.columns:
            y = df['weekly_growth'].values
        else:
            y = df['growth'].values  # Use daily growth if weekly not available
        
        logger.info(f"Prepared features: {len(available_features)} features, {len(y)} samples")
        
        return X, y, available_features
    
    def fit(self, df: pd.DataFrame, validate: bool = True) -> Dict[str, Any]:
        """Train the model on creator data.
        
        Args:
            df: Training dataframe with your data structure
            validate: Whether to perform validation during training
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.log_training_start(self.model_type, len(df), 0)
        training_start_time = datetime.now()
        
        # Prepare features and target
        X, y, feature_names = self.prepare_features(df)
        self.feature_names = feature_names
        
        # Handle missing values
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            logger.warning("Found missing values, handling with median imputation")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            y = np.nan_to_num(y, nan=np.nanmedian(y))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred_train = self.model.predict(X_scaled)
        train_metrics = self._calculate_metrics(y, y_pred_train)
        
        # Store training statistics
        self.training_stats = {
            'n_samples': len(X),
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'train_metrics': train_metrics,
            'model_params': self._get_model_params()
        }
        
        # Cross-validation if requested
        if validate:
            cv_results = self.cross_validate(df)
            self.training_stats['cv_results'] = cv_results
        
        # Training completion
        training_duration = (datetime.now() - training_start_time).total_seconds()
        logger.log_training_complete(self.model_type, training_duration, train_metrics)
        
        return self.training_stats
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            df: Dataframe with same structure as training data
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare features
        X, _, _ = self.prepare_features(df)
        
        # Handle missing values (use same strategy as training)
        if np.any(np.isnan(X)):
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        logger.log_prediction(len(predictions), 0.95)
        return predictions
    
    def predict_with_confidence(self, df: pd.DataFrame, 
                               confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals.
        
        Args:
            df: Dataframe with same structure as training data
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        predictions = self.predict(df)
        
        # Calculate prediction intervals using residual standard error
        if hasattr(self, 'training_stats') and 'train_metrics' in self.training_stats:
            residual_std = np.sqrt(self.training_stats['train_metrics']['mse'])
            
            # Use t-distribution for small samples
            n_train = self.training_stats['n_samples']
            df_residual = n_train - len(self.feature_names) - 1
            
            if df_residual > 0:
                t_critical = stats.t.ppf((1 + confidence_level) / 2, df_residual)
                margin_error = t_critical * residual_std
            else:
                # Fallback to normal distribution
                z_critical = stats.norm.ppf((1 + confidence_level) / 2)
                margin_error = z_critical * residual_std
            
            lower_bounds = predictions - margin_error
            upper_bounds = predictions + margin_error
        else:
            # Fallback: use 10% margin
            margin = 0.1 * np.abs(predictions)
            lower_bounds = predictions - margin
            upper_bounds = predictions + margin
        
        logger.log_prediction(len(predictions), confidence_level)
        return predictions, lower_bounds, upper_bounds
    
    def cross_validate(self, df: pd.DataFrame, cv_folds: int = 5) -> Dict[str, Any]:
        """Perform time-aware cross-validation.
        
        Args:
            df: Training dataframe
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {cv_folds}-fold time series cross-validation")
        
        # Prepare data
        X, y, _ = self.prepare_features(df)
        
        # Handle missing values
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            y = np.nan_to_num(y, nan=np.nanmedian(y))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Score the model
        cv_scores = {
            'r2_scores': cross_val_score(self.model, X_scaled, y, cv=tscv, scoring='r2'),
            'neg_mse_scores': cross_val_score(self.model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error'),
            'neg_mae_scores': cross_val_score(self.model, X_scaled, y, cv=tscv, scoring='neg_mean_absolute_error')
        }
        
        # Calculate summary statistics
        cv_results = {
            'r2_mean': cv_scores['r2_scores'].mean(),
            'r2_std': cv_scores['r2_scores'].std(),
            'mse_mean': -cv_scores['neg_mse_scores'].mean(),
            'mse_std': cv_scores['neg_mse_scores'].std(),
            'mae_mean': -cv_scores['neg_mae_scores'].mean(),
            'mae_std': cv_scores['neg_mae_scores'].std(),
            'cv_folds': cv_folds
        }
        
        logger.log_validation_results('time_series_cv', cv_results)
        return cv_results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (coefficients) with interpretation.
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Get coefficients
        if hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_
        else:
            raise ValueError("Model does not have coefficients")
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        # Sort by absolute importance
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        
        # Add interpretation
        importance_df['importance_rank'] = range(1, len(importance_df) + 1)
        importance_df['effect_direction'] = np.where(importance_df['coefficient'] > 0, 'Positive', 'Negative')
        
        # Add business interpretation for key features
        business_interpretations = {
            'weekly_posting_frequency': 'Expected follower growth per additional weekly post',
            'engagement_rate': 'Growth impact of higher engagement rates',
            'share_reels': 'Effect of having more reels in content mix',
            'share_posts': 'Effect of having more posts in content mix',
            'roi_follows_per_hour': 'Growth impact of content efficiency',
            'posted_in_optimal_window': 'Benefit of posting at optimal times',
            'saturation_flag': 'Impact of potential posting saturation'
        }
        
        importance_df['business_meaning'] = importance_df['feature'].map(
            business_interpretations
        ).fillna('Impact on follower growth')
        
        return importance_df
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary with statistics.
        
        Returns:
            Dictionary with model summary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting summary")
        
        summary = {
            'model_info': {
                'model_type': self.model_type,
                'regularization': self.regularization,
                'n_features': len(self.feature_names),
                'feature_names': self.feature_names
            },
            'training_metrics': self.training_stats.get('train_metrics', {}),
            'model_params': self.training_stats.get('model_params', {}),
            'feature_importance': self.get_feature_importance().to_dict('records')[:5]  # Top 5
        }
        
        # Add cross-validation results if available
        if 'cv_results' in self.training_stats:
            summary['cross_validation'] = self.training_stats['cv_results']
        
        # Add model quality assessment
        train_metrics = self.training_stats.get('train_metrics', {})
        quality_assessment = self._assess_model_quality(train_metrics)
        summary['quality_assessment'] = quality_assessment
        
        return summary
    
    def save_model(self, filepath: str, version: str = "1.0.0") -> bool:
        """Save the trained model to file.
        
        Args:
            filepath: Path to save the model
            version: Model version string
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_fitted:
            logger.error("Cannot save unfitted model")
            return False
        
        try:
            # Prepare model data
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'training_stats': self.training_stats,
                'version': version,
                'created_at': datetime.now().isoformat()
            }
            
            # Save using joblib
            joblib.dump(model_data, filepath)
            logger.log_model_save(filepath, version)
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    @classmethod
    def load_model(cls, filepath: str) -> 'CreatorGrowthModel':
        """Load a saved model from file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded CreatorGrowthModel instance
        """
        try:
            # Load model data
            model_data = joblib.load(filepath)
            
            # Create instance
            instance = cls(
                model_type=model_data['model_type'],
                random_state=42
            )
            
            # Restore model state
            instance.model = model_data['model']
            instance.scaler = model_data['scaler']
            instance.feature_names = model_data['feature_names']
            instance.training_stats = model_data['training_stats']
            instance.is_fitted = True
            
            logger.info(f"Loaded model version {model_data.get('version', 'unknown')}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive model metrics."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Additional metrics
        metrics['mape'] = np.mean(np.abs(safe_divide(y_true - y_pred, y_true))) * 100
        
        # Directional accuracy
        direction_correct = np.sign(y_true) == np.sign(y_pred)
        metrics['directional_accuracy'] = np.mean(direction_correct) * 100
        
        return metrics
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = {}
        
        if hasattr(self.model, 'intercept_'):
            params['intercept'] = float(self.model.intercept_)
        
        if hasattr(self.model, 'alpha'):
            params['alpha'] = float(self.model.alpha)
            
        return params
    
    def _assess_model_quality(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess model quality against thresholds."""
        assessment = {
            'overall_quality': 'Good',
            'warnings': [],
            'recommendations': []
        }
        
        # Check R² threshold
        if metrics.get('r2', 0) < MODEL_QUALITY_THRESHOLDS['min_r2_score']:
            assessment['warnings'].append(f"Low R² score: {metrics.get('r2', 0):.3f}")
            assessment['overall_quality'] = 'Poor'
            assessment['recommendations'].append("Consider adding more features or different model type")
        
        # Check directional accuracy
        if metrics.get('directional_accuracy', 0) < MODEL_QUALITY_THRESHOLDS['min_directional_accuracy'] * 100:
            assessment['warnings'].append(f"Low directional accuracy: {metrics.get('directional_accuracy', 0):.1f}%")
            assessment['recommendations'].append("Review feature engineering for better trend prediction")
        
        # Overall assessment
        if len(assessment['warnings']) == 0:
            assessment['overall_quality'] = 'Excellent'
        elif len(assessment['warnings']) == 1:
            assessment['overall_quality'] = 'Good'
        
        return assessment


# =============================================================================
# MODEL FACTORY AND UTILITIES
# =============================================================================

def create_model(model_type: str = 'linear', **kwargs) -> CreatorGrowthModel:
    """Factory function to create different model types.
    
    Args:
        model_type: Type of model to create
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized model instance
    """
    return CreatorGrowthModel(model_type=model_type, **kwargs)


def train_model_pipeline(df: pd.DataFrame, 
                        model_type: str = 'linear',
                        validate: bool = True,
                        save_path: Optional[str] = None) -> Tuple[CreatorGrowthModel, Dict[str, Any]]:
    """Complete model training pipeline.
    
    Args:
        df: Training dataframe with your data structure
        model_type: Type of model to train
        validate: Whether to perform validation
        save_path: Optional path to save the trained model
        
    Returns:
        Tuple of (trained_model, training_results)
    """
    logger.info(f"Starting model training pipeline with {model_type} regression")
    
    # Create and train model
    model = create_model(model_type)
    training_results = model.fit(df, validate=validate)
    
    # Save model if path provided
    if save_path:
        model.save_model(save_path)
    
    # Log results
    train_metrics = training_results.get('train_metrics', {})
    logger.info(f"Model training completed - R²: {train_metrics.get('r2', 0):.3f}, "
               f"RMSE: {train_metrics.get('rmse', 0):.0f}")
    
    return model, training_results


def compare_models(df: pd.DataFrame, 
                  model_types: List[str] = ['linear', 'ridge', 'lasso']) -> pd.DataFrame:
    """Compare different model types on the same data.
    
    Args:
        df: Training dataframe
        model_types: List of model types to compare
        
    Returns:
        DataFrame with comparison results
    """
    logger.info(f"Comparing {len(model_types)} model types")
    
    results = []
    
    for model_type in model_types:
        try:
            model = create_model(model_type)
            training_stats = model.fit(df, validate=True)
            
            # Extract key metrics
            train_metrics = training_stats.get('train_metrics', {})
            cv_results = training_stats.get('cv_results', {})
            
            result = {
                'model_type': model_type,
                'train_r2': train_metrics.get('r2', 0),
                'train_rmse': train_metrics.get('rmse', 0),
                'train_mae': train_metrics.get('mae', 0),
                'cv_r2_mean': cv_results.get('r2_mean', 0),
                'cv_r2_std': cv_results.get('r2_std', 0),
                'cv_rmse_mean': np.sqrt(cv_results.get('mse_mean', 0)),
                'directional_accuracy': train_metrics.get('directional_accuracy', 0)
            }
            
            results.append(result)
            logger.info(f"Completed {model_type}: R²={result['train_r2']:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
    
    comparison_df = pd.DataFrame(results)
    
    # Rank models by cross-validation R²
    if 'cv_r2_mean' in comparison_df.columns:
        comparison_df['rank'] = comparison_df['cv_r2_mean'].rank(ascending=False)
    
    return comparison_df.sort_values('cv_r2_mean', ascending=False)


# =============================================================================
# EXPORT LIST
# =============================================================================

__all__ = [
    'CreatorGrowthModel',
    'create_model',
    'train_model_pipeline', 
    'compare_models'
]
