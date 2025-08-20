"""
Creator Growth Navigator - Cross-Validation Module

Time-series aware cross-validation utilities for creator growth prediction models
with proper temporal splitting and comprehensive validation metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_validate
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    make_scorer
)
from sklearn.base import clone
import warnings
from datetime import datetime, timedelta

# Import from project modules
from src.config.config import MODEL_CONFIG, FEATURE_CONFIG
from src.utils.logger import get_model_logger
from src.utils.helpers import safe_divide, calculate_confidence_interval

logger = get_model_logger()
warnings.filterwarnings('ignore')


# =============================================================================
# CORE TIME SERIES CROSS-VALIDATION CLASS
# =============================================================================

class CreatorTimeSeriesCV:
    """Advanced time-series cross-validation for creator growth prediction."""
    
    def __init__(self, 
                 n_splits: int = 5,
                 test_size: Optional[int] = None,
                 gap: int = 0,
                 max_train_size: Optional[int] = None):
        """Initialize time series cross-validator.
        
        Args:
            n_splits: Number of splits for cross-validation
            test_size: Size of test set for each split (None for auto)
            gap: Gap between train and test sets
            max_train_size: Maximum training set size (None for unlimited)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size
        
        # Initialize TimeSeriesSplit
        self.cv_splitter = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            max_train_size=max_train_size
        )
        
        # Results storage
        self.cv_results = {}
        self.split_details = []
        
        logger.info(f"Initialized TimeSeriesCV with {n_splits} splits")
    
    def cross_validate(self,
                      model,
                      df: pd.DataFrame,
                      feature_columns: Optional[List[str]] = None,
                      target_column: str = 'weekly_growth',
                      scoring_metrics: Optional[Dict[str, str]] = None,
                      return_train_score: bool = True,
                      verbose: bool = True) -> Dict[str, Any]:
        """Perform comprehensive time-series cross-validation.
        
        Args:
            model: Model instance to cross-validate
            df: DataFrame with creator data
            feature_columns: List of feature columns (None for auto)
            target_column: Target column name
            scoring_metrics: Dictionary of scoring metrics
            return_train_score: Whether to return training scores
            verbose: Whether to print progress
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info("Starting time-series cross-validation")
        
        # Prepare data
        X, y, feature_names = self._prepare_data(df, feature_columns, target_column)
        
        # Default scoring metrics
        if scoring_metrics is None:
            scoring_metrics = {
                'neg_mean_squared_error': 'neg_mean_squared_error',
                'neg_mean_absolute_error': 'neg_mean_absolute_error',
                'r2': 'r2'
            }
        
        # Perform cross-validation
        cv_scores = cross_validate(
            model, X, y,
            cv=self.cv_splitter,
            scoring=scoring_metrics,
            return_train_score=return_train_score,
            return_estimator=True,
            n_jobs=1  # Sequential for time series
        )
        
        # Process results
        self.cv_results = self._process_cv_results(cv_scores, scoring_metrics)
        
        # Detailed fold analysis
        fold_details = self._analyze_fold_performance(
            model, X, y, feature_names, verbose
        )
        
        # Compile comprehensive results
        comprehensive_results = {
            'cv_summary': self.cv_results,
            'fold_details': fold_details,
            'model_stability': self._assess_model_stability(cv_scores),
            'feature_consistency': self._analyze_feature_consistency(cv_scores),
            'temporal_performance': self._analyze_temporal_performance(fold_details)
        }
        
        if verbose:
            self._print_cv_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _prepare_data(self,
                     df: pd.DataFrame,
                     feature_columns: Optional[List[str]],
                     target_column: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for cross-validation."""
        # Create primary feature if needed
        if 'weekly_posting_frequency' not in df.columns:
            df['weekly_posting_frequency'] = df['posts'] + df['reels'] + df['stories']
        
        # Auto-select features if not provided
        if feature_columns is None:
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
        y = df[target_column].values
        
        # Handle missing values
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            y = np.nan_to_num(y, nan=np.nanmedian(y))
        
        return X, y, available_features
    
    def _process_cv_results(self,
                           cv_scores: Dict[str, np.ndarray],
                           scoring_metrics: Dict[str, str]) -> Dict[str, Any]:
        """Process cross-validation results into summary statistics."""
        processed_results = {}
        
        for metric_name, metric_key in scoring_metrics.items():
            test_scores = cv_scores[f'test_{metric_name}']
            
            # Convert negative scores back to positive for MSE/MAE
            if 'neg_' in metric_name:
                test_scores = -test_scores
                display_name = metric_name.replace('neg_', '')
            else:
                display_name = metric_name
            
            processed_results[display_name] = {
                'scores': test_scores.tolist(),
                'mean': np.mean(test_scores),
                'std': np.std(test_scores),
                'min': np.min(test_scores),
                'max': np.max(test_scores),
                'cv': np.std(test_scores) / np.abs(np.mean(test_scores)) if np.mean(test_scores) != 0 else np.inf
            }
            
            # Add confidence interval
            lower, upper = calculate_confidence_interval(test_scores)
            processed_results[display_name]['ci_95'] = {'lower': lower, 'upper': upper}
            
            # Include training scores if available
            if f'train_{metric_name}' in cv_scores:
                train_scores = cv_scores[f'train_{metric_name}']
                if 'neg_' in metric_name:
                    train_scores = -train_scores
                
                processed_results[display_name]['train_scores'] = {
                    'scores': train_scores.tolist(),
                    'mean': np.mean(train_scores),
                    'std': np.std(train_scores)
                }
                
                # Calculate overfitting indicator
                processed_results[display_name]['overfitting_score'] = self._calculate_overfitting_score(
                    train_scores, test_scores, metric_name
                )
        
        return processed_results
    
    def _analyze_fold_performance(self,
                                 model,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 feature_names: List[str],
                                 verbose: bool) -> List[Dict[str, Any]]:
        """Analyze performance for each individual fold."""
        fold_details = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.cv_splitter.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Clone and fit model
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = fold_model.predict(X_train)
            y_pred_test = fold_model.predict(X_test)
            
            # Calculate detailed metrics
            fold_metrics = {
                'fold_number': fold_idx + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_period': f"samples {train_idx[0]} to {train_idx[-1]}",
                'test_period': f"samples {test_idx} to {test_idx[-1]}",
                
                # Performance metrics
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'test_mse': mean_squared_error(y_test, y_pred_test),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                
                # Business metrics
                'directional_accuracy': np.mean(np.sign(y_test) == np.sign(y_pred_test)) * 100,
                'mean_prediction': np.mean(y_pred_test),
                'mean_actual': np.mean(y_test),
                'prediction_range': np.max(y_pred_test) - np.min(y_pred_test)
            }
            
            # Feature importance (if available)
            if hasattr(fold_model, 'coef_'):
                feature_importance = dict(zip(feature_names, fold_model.coef_))
                fold_metrics['feature_importance'] = feature_importance
                fold_metrics['top_features'] = sorted(
                    feature_importance.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )[:5]
            
            fold_details.append(fold_metrics)
            
            if verbose:
                logger.info(f"Fold {fold_idx + 1}: Test R²={fold_metrics['test_r2']:.3f}, "
                           f"Test RMSE={np.sqrt(fold_metrics['test_mse']):.0f}")
        
        return fold_details
    
    def _assess_model_stability(self, cv_scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Assess model stability across folds."""
        stability_metrics = {}
        
        # Coefficient of variation for each metric
        for key, scores in cv_scores.items():
            if key.startswith('test_'):
                metric_name = key.replace('test_', '').replace('neg_', '')
                
                # Convert negative scores
                if 'neg_' in key:
                    scores = -scores
                
                cv_value = np.std(scores) / np.abs(np.mean(scores)) if np.mean(scores) != 0 else np.inf
                stability_metrics[f'{metric_name}_stability'] = {
                    'coefficient_of_variation': cv_value,
                    'stable': cv_value < 0.2,  # Less than 20% CV considered stable
                    'score_range': np.max(scores) - np.min(scores)
                }
        
        # Overall stability score
        stable_metrics = sum([
            metrics['stable'] for metrics in stability_metrics.values()
        ])
        total_metrics = len(stability_metrics)
        
        stability_metrics['overall_stability'] = {
            'stability_score': stable_metrics / total_metrics if total_metrics > 0 else 0,
            'stable_metrics_count': stable_metrics,
            'total_metrics': total_metrics,
            'assessment': 'Stable' if stable_metrics / total_metrics >= 0.8 else 'Unstable'
        }
        
        return stability_metrics
    
    def _analyze_feature_consistency(self, cv_scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze feature importance consistency across folds."""
        if 'estimator' not in cv_scores:
            return {'status': 'No estimators available for feature analysis'}
        
        estimators = cv_scores['estimator']
        feature_importances = []
        
        # Extract coefficients from each fold
        for estimator in estimators:
            if hasattr(estimator, 'coef_'):
                feature_importances.append(estimator.coef_)
        
        if not feature_importances:
            return {'status': 'No coefficients available'}
        
        feature_importances = np.array(feature_importances)
        
        # Calculate consistency metrics
        consistency_metrics = {
            'feature_importance_std': np.std(feature_importances, axis=0).tolist(),
            'feature_importance_cv': (np.std(feature_importances, axis=0) / 
                                    np.abs(np.mean(feature_importances, axis=0))).tolist(),
            'sign_consistency': np.mean(
                np.sign(feature_importances) == np.sign(np.mean(feature_importances, axis=0)),
                axis=0
            ).tolist()
        }
        
        return consistency_metrics
    
    def _analyze_temporal_performance(self, fold_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends across temporal folds."""
        fold_numbers = [fold['fold_number'] for fold in fold_details]
        test_r2_scores = [fold['test_r2'] for fold in fold_details]
        test_mae_scores = [fold['test_mae'] for fold in fold_details]
        
        # Performance trend analysis
        if len(fold_numbers) > 2:
            r2_trend_corr = np.corrcoef(fold_numbers, test_r2_scores)[0, 1]
            mae_trend_corr = np.corrcoef(fold_numbers, test_mae_scores)[0, 1]
        else:
            r2_trend_corr = 0
            mae_trend_corr = 0
        
        temporal_analysis = {
            'r2_trend_correlation': r2_trend_corr,
            'mae_trend_correlation': mae_trend_corr,
            'performance_degradation': r2_trend_corr < -0.3,  # Strong negative trend
            'performance_improvement': r2_trend_corr > 0.3,   # Strong positive trend
            'consistent_performance': abs(r2_trend_corr) < 0.2,
            
            # Performance range
            'r2_range': max(test_r2_scores) - min(test_r2_scores),
            'mae_range': max(test_mae_scores) - min(test_mae_scores),
            
            # Best/worst fold analysis
            'best_fold': fold_numbers[np.argmax(test_r2_scores)],
            'worst_fold': fold_numbers[np.argmin(test_r2_scores)],
            
            'fold_performance': [
                {
                    'fold': fold['fold_number'],
                    'test_r2': fold['test_r2'],
                    'test_mae': fold['test_mae'],
                    'directional_accuracy': fold['directional_accuracy']
                }
                for fold in fold_details
            ]
        }
        
        return temporal_analysis
    
    def _calculate_overfitting_score(self,
                                   train_scores: np.ndarray,
                                   test_scores: np.ndarray,
                                   metric_name: str) -> float:
        """Calculate overfitting indicator."""
        if 'r2' in metric_name:
            # For R², higher is better - overfitting if train >> test
            overfitting = np.mean(train_scores) - np.mean(test_scores)
        else:
            # For MSE/MAE, lower is better - overfitting if test >> train
            overfitting = np.mean(test_scores) - np.mean(train_scores)
        
        return max(0, overfitting)
    
    def _print_cv_summary(self, results: Dict[str, Any]) -> None:
        """Print formatted cross-validation summary."""
        cv_summary = results['cv_summary']
        
        print("\n" + "="*60)
        print("TIME SERIES CROSS-VALIDATION SUMMARY")
        print("="*60)
        
        # Main metrics
        for metric, data in cv_summary.items():
            if isinstance(data, dict) and 'mean' in data:
                print(f"{metric.upper():20s}: {data['mean']:8.4f} ± {data['std']:6.4f}")
                print(f"{'Range':<20s}: {data['min']:8.4f} - {data['max']:8.4f}")
                if 'overfitting_score' in data:
                    print(f"{'Overfitting':<20s}: {data['overfitting_score']:8.4f}")
                print("-" * 40)
        
        # Stability assessment
        stability = results['model_stability']['overall_stability']
        print(f"Model Stability: {stability['assessment']} "
              f"({stability['stable_metrics_count']}/{stability['total_metrics']} metrics stable)")
        
        # Temporal trends
        temporal = results['temporal_performance']
        if temporal['performance_degradation']:
            print("⚠️  Performance degradation detected over time")
        elif temporal['performance_improvement']:
            print("✅ Performance improvement detected over time")
        else:
            print("📊 Consistent performance across time periods")
        
        print("="*60 + "\n")


# =============================================================================
# SPECIALIZED CROSS-VALIDATION FUNCTIONS
# =============================================================================

def walk_forward_validation(model,
                           df: pd.DataFrame,
                           initial_train_size: int,
                           step_size: int = 1,
                           target_column: str = 'weekly_growth',
                           feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Perform walk-forward validation for time series.
    
    Args:
        model: Model to validate
        df: DataFrame with time series data
        initial_train_size: Initial training window size
        step_size: Number of steps to advance each iteration
        target_column: Target column name
        feature_columns: Feature columns (None for auto)
        
    Returns:
        Dictionary with walk-forward validation results
    """
    logger.info(f"Starting walk-forward validation with initial size {initial_train_size}")
    
    # Prepare features
    cv_instance = CreatorTimeSeriesCV()
    X, y, feature_names = cv_instance._prepare_data(df, feature_columns, target_column)
    
    results = []
    
    for i in range(initial_train_size, len(X), step_size):
        # Training set: from start to current position
        X_train = X[:i]
        y_train = y[:i]
        
        # Test set: next step_size observations
        end_idx = min(i + step_size, len(X))
        X_test = X[i:end_idx]
        y_test = y[i:end_idx]
        
        if len(X_test) == 0:
            break
        
        # Fit and predict
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)
        
        # Calculate metrics
        step_results = {
            'step': len(results) + 1,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_mse': mean_squared_error(y_test, y_pred),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_r2': r2_score(y_test, y_pred),
            'directional_accuracy': np.mean(np.sign(y_test) == np.sign(y_pred)) * 100
        }
        
        results.append(step_results)
    
    # Aggregate results
    all_metrics = {}
    for metric in ['test_mse', 'test_mae', 'test_r2', 'directional_accuracy']:
        values = [r[metric] for r in results]
        all_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    return {
        'step_results': results,
        'aggregate_metrics': all_metrics,
        'total_steps': len(results),
        'average_train_size': np.mean([r['train_size'] for r in results])
    }


def expanding_window_validation(model,
                               df: pd.DataFrame,
                               min_train_size: int,
                               test_size: int,
                               target_column: str = 'weekly_growth',
                               feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Perform expanding window validation.
    
    Args:
        model: Model to validate
        df: DataFrame with time series data
        min_train_size: Minimum training window size
        test_size: Size of each test window
        target_column: Target column name
        feature_columns: Feature columns (None for auto)
        
    Returns:
        Dictionary with expanding window validation results
    """
    logger.info(f"Starting expanding window validation")
    
    # Prepare features
    cv_instance = CreatorTimeSeriesCV()
    X, y, feature_names = cv_instance._prepare_data(df, feature_columns, target_column)
    
    results = []
    
    # Start from minimum train size and expand
    for train_end in range(min_train_size, len(X) - test_size + 1, test_size):
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        test_start = train_end
        test_end = min(train_end + test_size, len(X))
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        if len(X_test) < test_size:
            break
        
        # Fit and predict
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)
        
        # Calculate metrics
        window_results = {
            'window': len(results) + 1,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_mse': mean_squared_error(y_test, y_pred),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_r2': r2_score(y_test, y_pred),
            'directional_accuracy': np.mean(np.sign(y_test) == np.sign(y_pred)) * 100
        }
        
        results.append(window_results)
    
    # Aggregate results
    aggregate_metrics = {}
    for metric in ['test_mse', 'test_mae', 'test_r2', 'directional_accuracy']:
        values = [r[metric] for r in results]
        aggregate_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    return {
        'window_results': results,
        'aggregate_metrics': aggregate_metrics,
        'total_windows': len(results)
    }


# =============================================================================
# EXPORT LIST
# =============================================================================

__all__ = [
    'CreatorTimeSeriesCV',
    'walk_forward_validation',
    'expanding_window_validation'
]
