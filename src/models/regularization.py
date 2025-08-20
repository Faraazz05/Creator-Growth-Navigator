"""
Creator Growth Navigator - Regularization Models Module

Advanced regularization techniques including Ridge and Lasso regression with
hyperparameter tuning, feature selection, and model comparison capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFECV
import joblib
from datetime import datetime
import warnings

# Import from project modules
from src.config.config import MODEL_CONFIG, FEATURE_CONFIG
from src.utils.logger import get_model_logger
from src.utils.helpers import safe_divide

logger = get_model_logger()
warnings.filterwarnings('ignore')


# =============================================================================
# REGULARIZED REGRESSION MODELS
# =============================================================================

class RegularizedRegression:
    """Advanced regularized regression with hyperparameter tuning and feature selection."""
    
    def __init__(self, 
                 model_type: str = 'ridge',
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 random_state: int = 42,
                 max_iter: int = 2000):
        """Initialize regularized regression model.
        
        Args:
            model_type: Type of regularization ('ridge', 'lasso', 'elasticnet')
            alpha: Regularization strength
            l1_ratio: ElasticNet mixing parameter (0=Ridge, 1=Lasso)
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for convergence
        """
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.max_iter = max_iter
        
        # Initialize model
        self.model = self._create_model()
        
        # Create pipeline with scaling
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', self.model)
        ])
        
        # Model state
        self.is_fitted = False
        self.feature_names = None
        self.training_stats = {}
        
        logger.info(f"Initialized {self.model_type} regression with alpha={self.alpha}")
    
    def _create_model(self):
        """Create the appropriate regression model."""
        if self.model_type == 'ridge':
            return Ridge(
                alpha=self.alpha,
                random_state=self.random_state,
                max_iter=self.max_iter
            )
        elif self.model_type == 'lasso':
            return Lasso(
                alpha=self.alpha,
                random_state=self.random_state,
                max_iter=self.max_iter,
                selection='random'  # For faster convergence
            )
        elif self.model_type == 'elasticnet':
            return ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                random_state=self.random_state,
                max_iter=self.max_iter,
                selection='random'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the regularized model on creator data.
        
        Args:
            df: Training dataframe with your data structure
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} regression model")
        training_start = datetime.now()
        
        # Prepare features using the same logic as base regression
        X, y, feature_names = self._prepare_features(df)
        self.feature_names = feature_names
        
        # Fit the pipeline
        self.pipeline.fit(X, y)
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred = self.pipeline.predict(X)
        metrics = self._calculate_metrics(y, y_pred)
        
        # Store training statistics
        self.training_stats = {
            'n_samples': len(X),
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'train_metrics': metrics,
            'model_params': self._get_model_params(),
            'regularization_info': self._get_regularization_info()
        }
        
        training_duration = (datetime.now() - training_start).total_seconds()
        logger.info(f"{self.model_type} training completed in {training_duration:.2f}s - "
                   f"R²: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:.0f}")
        
        return self.training_stats
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features using the same logic as the base model."""
        # Create primary feature
        if 'weekly_posting_frequency' not in df.columns:
            df['weekly_posting_frequency'] = df['posts'] + df['reels'] + df['stories']
        
        # Feature sets aligned with your data structure
        feature_columns = [
            'weekly_posting_frequency',  # Primary predictor
            'share_posts', 'share_reels', 'share_stories',  # Content mix
            'engagement_rate', 'avg_hashtag_count',  # Engagement
            'post_consistency_variance_7d', 'posted_in_optimal_window',  # Consistency
            'roi_follows_per_hour', 'minutes_spent',  # ROI
            'month', 'quarter',  # Temporal
            'saturation_flag'  # Saturation
        ]
        
        # Filter to existing columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features].values
        
        # Target variable
        if 'weekly_growth' in df.columns:
            y = df['weekly_growth'].values
        else:
            y = df['growth'].values
        
        # Handle missing values
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            y = np.nan_to_num(y, nan=np.nanmedian(y))
        
        return X, y, available_features
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X, _, _ = self._prepare_features(df)
        
        # Handle missing values
        if np.any(np.isnan(X)):
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        return self.pipeline.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from regularized coefficients."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        regressor = self.pipeline.named_steps['regressor']
        coefficients = regressor.coef_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Add regularization-specific insights
        if self.model_type == 'lasso':
            importance_df['selected'] = (np.abs(coefficients) > 1e-10).astype(int)
            importance_df['importance_type'] = 'Lasso Selection'
        else:
            importance_df['importance_type'] = 'Ridge Shrinkage'
        
        return importance_df
    
    def get_selected_features(self) -> List[str]:
        """Get features selected by Lasso (non-zero coefficients)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if self.model_type != 'lasso':
            return self.feature_names  # All features for Ridge/ElasticNet
        
        regressor = self.pipeline.named_steps['regressor']
        selected_mask = np.abs(regressor.coef_) > 1e-10
        
        return [self.feature_names[i] for i, selected in enumerate(selected_mask) if selected]
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # MAPE
        metrics['mape'] = np.mean(np.abs(safe_divide(y_true - y_pred, y_true))) * 100
        
        # Directional accuracy
        direction_correct = np.sign(y_true) == np.sign(y_pred)
        metrics['directional_accuracy'] = np.mean(direction_correct) * 100
        
        return metrics
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        regressor = self.pipeline.named_steps['regressor']
        params = {
            'alpha': regressor.alpha,
            'intercept': float(regressor.intercept_)
        }
        
        if hasattr(regressor, 'l1_ratio'):
            params['l1_ratio'] = regressor.l1_ratio
            
        return params
    
    def _get_regularization_info(self) -> Dict[str, Any]:
        """Get regularization-specific information."""
        regressor = self.pipeline.named_steps['regressor']
        coeffs = regressor.coef_
        
        info = {
            'n_nonzero_coef': np.sum(np.abs(coeffs) > 1e-10),
            'max_abs_coef': np.max(np.abs(coeffs)),
            'l2_norm': np.sqrt(np.sum(coeffs**2)),
            'l1_norm': np.sum(np.abs(coeffs))
        }
        
        if self.model_type == 'lasso':
            info['sparsity_ratio'] = 1 - (info['n_nonzero_coef'] / len(coeffs))
        
        return info
    
    def save(self, filepath: str) -> bool:
        """Save the trained model."""
        try:
            model_data = {
                'pipeline': self.pipeline,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'training_stats': self.training_stats,
                'created_at': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Saved {self.model_type} model to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    @classmethod
    def load(cls, filepath: str) -> 'RegularizedRegression':
        """Load a saved model."""
        try:
            model_data = joblib.load(filepath)
            
            instance = cls(model_type=model_data['model_type'])
            instance.pipeline = model_data['pipeline']
            instance.feature_names = model_data['feature_names']
            instance.training_stats = model_data['training_stats']
            instance.is_fitted = True
            
            logger.info(f"Loaded {instance.model_type} model from {filepath}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

class RegularizationTuner:
    """Hyperparameter tuning for regularized regression models."""
    
    def __init__(self, model_type: str = 'ridge', cv_folds: int = 5, random_state: int = 42):
        """Initialize the tuner.
        
        Args:
            model_type: Type of regularization to tune
            cv_folds: Number of cross-validation folds
            random_state: Random seed
        """
        self.model_type = model_type
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_model = None
        self.tuning_results = {}
        
    def tune_alpha(self, df: pd.DataFrame, 
                   alpha_range: Optional[List[float]] = None,
                   search_type: str = 'grid') -> Dict[str, Any]:
        """Tune the alpha (regularization strength) parameter.
        
        Args:
            df: Training dataframe
            alpha_range: Range of alpha values to search
            search_type: 'grid' or 'random' search
            
        Returns:
            Dictionary with tuning results
        """
        logger.info(f"Tuning alpha for {self.model_type} regression")
        
        # Prepare data
        base_model = RegularizedRegression(model_type=self.model_type)
        X, y, feature_names = base_model._prepare_features(df)
        
        # Define search space
        if alpha_range is None:
            alpha_range = np.logspace(-4, 2, 50)  # 0.0001 to 100
        
        param_grid = {'regressor__alpha': alpha_range}
        
        # Additional parameters for ElasticNet
        if self.model_type == 'elasticnet':
            param_grid['regressor__l1_ratio'] = np.linspace(0.1, 0.9, 9)
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', base_model._create_model())
        ])
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Perform search
        if search_type == 'grid':
            search = GridSearchCV(
                pipeline, param_grid, cv=tscv, 
                scoring='neg_mean_squared_error', n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                pipeline, param_grid, cv=tscv,
                scoring='neg_mean_squared_error', n_iter=50, 
                random_state=self.random_state, n_jobs=-1
            )
        
        # Fit search
        search.fit(X, y)
        
        # Extract results
        best_params = search.best_params_
        best_score = -search.best_score_  # Convert back from negative
        
        # Create best model
        if self.model_type == 'elasticnet':
            self.best_model = RegularizedRegression(
                model_type=self.model_type,
                alpha=best_params['regressor__alpha'],
                l1_ratio=best_params.get('regressor__l1_ratio', 0.5)
            )
        else:
            self.best_model = RegularizedRegression(
                model_type=self.model_type,
                alpha=best_params['regressor__alpha']
            )
        
        # Train best model
        training_stats = self.best_model.fit(df)
        
        # Store results
        self.tuning_results = {
            'best_params': best_params,
            'best_cv_mse': best_score,
            'best_cv_rmse': np.sqrt(best_score),
            'cv_scores': -search.cv_results_['mean_test_score'],
            'training_stats': training_stats,
            'search_type': search_type,
            'n_candidates': len(search.cv_results_['params'])
        }
        
        logger.info(f"Best alpha: {best_params['regressor__alpha']:.6f} "
                   f"(CV RMSE: {np.sqrt(best_score):.2f})")
        
        return self.tuning_results
    
    def compare_regularization_strengths(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compare different regularization strengths."""
        logger.info("Comparing regularization strengths")
        
        alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        results = []
        
        for alpha in alpha_values:
            try:
                model = RegularizedRegression(model_type=self.model_type, alpha=alpha)
                stats = model.fit(df)
                
                result = {
                    'alpha': alpha,
                    'train_r2': stats['train_metrics']['r2'],
                    'train_rmse': stats['train_metrics']['rmse'],
                    'train_mae': stats['train_metrics']['mae'],
                    'n_features': stats['n_features']
                }
                
                # Add regularization-specific metrics
                if self.model_type == 'lasso':
                    selected_features = model.get_selected_features()
                    result['n_selected_features'] = len(selected_features)
                    result['sparsity'] = 1 - (len(selected_features) / stats['n_features'])
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to train model with alpha={alpha}: {e}")
        
        comparison_df = pd.DataFrame(results)
        return comparison_df.sort_values('train_r2', ascending=False)


# =============================================================================
# FEATURE SELECTION UTILITIES
# =============================================================================

def recursive_feature_elimination(df: pd.DataFrame, 
                                 model_type: str = 'ridge',
                                 min_features: int = 5) -> Tuple[List[str], Dict[str, Any]]:
    """Perform recursive feature elimination with cross-validation.
    
    Args:
        df: Training dataframe
        model_type: Type of regularization model
        min_features: Minimum number of features to select
        
    Returns:
        Tuple of (selected_features, elimination_results)
    """
    logger.info(f"Performing recursive feature elimination with {model_type}")
    
    # Prepare data
    base_model = RegularizedRegression(model_type=model_type)
    X, y, feature_names = base_model._prepare_features(df)
    
    # Create estimator
    estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', base_model._create_model())
    ])
    
    # Recursive feature elimination with CV
    selector = RFECV(
        estimator=estimator,
        min_features_to_select=min_features,
        cv=TimeSeriesSplit(n_splits=5),
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Fit selector
    selector.fit(X, y)
    
    # Get selected features
    selected_features = [feature_names[i] for i, selected in enumerate(selector.support_) if selected]
    
    # Results
    results = {
        'n_features_selected': selector.n_features_,
        'selected_features': selected_features,
        'feature_ranking': dict(zip(feature_names, selector.ranking_)),
        'cv_scores': selector.cv_results_['mean_test_score'],
        'optimal_n_features': selector.n_features_
    }
    
    logger.info(f"Selected {len(selected_features)} features: {selected_features}")
    
    return selected_features, results


def compare_regularization_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Compare different regularization methods on the same data.
    
    Args:
        df: Training dataframe
        
    Returns:
        DataFrame with comparison results
    """
    logger.info("Comparing regularization methods")
    
    methods = ['ridge', 'lasso', 'elasticnet']
    results = []
    
    for method in methods:
        try:
            # Tune hyperparameters
            tuner = RegularizationTuner(model_type=method)
            tuning_results = tuner.tune_alpha(df, search_type='grid')
            
            # Extract metrics
            best_model = tuner.best_model
            train_metrics = best_model.training_stats['train_metrics']
            
            result = {
                'method': method,
                'best_alpha': tuning_results['best_params']['regressor__alpha'],
                'cv_rmse': tuning_results['best_cv_rmse'],
                'train_r2': train_metrics['r2'],
                'train_rmse': train_metrics['rmse'],
                'train_mae': train_metrics['mae'],
                'directional_accuracy': train_metrics['directional_accuracy']
            }
            
            # Add method-specific metrics
            if method == 'lasso':
                selected_features = best_model.get_selected_features()
                result['n_selected_features'] = len(selected_features)
                result['sparsity_ratio'] = 1 - (len(selected_features) / len(best_model.feature_names))
            
            if method == 'elasticnet':
                result['l1_ratio'] = tuning_results['best_params'].get('regressor__l1_ratio', 0.5)
            
            results.append(result)
            logger.info(f"Completed {method}: CV RMSE={result['cv_rmse']:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to train {method}: {e}")
    
    comparison_df = pd.DataFrame(results)
    return comparison_df.sort_values('cv_rmse', ascending=True)


# =============================================================================
# EXPORT LIST
# =============================================================================

__all__ = [
    'RegularizedRegression',
    'RegularizationTuner', 
    'recursive_feature_elimination',
    'compare_regularization_methods'
]
