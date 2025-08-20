"""
Creator Growth Navigator - KPI Evaluation Module

Comprehensive KPI calculation and business metrics evaluation for creator growth
models including accuracy metrics, business impact, and ROI analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, median_absolute_error
)
from scipy import stats
import warnings
from datetime import datetime, timedelta

# Import from project modules
from src.config.config import MODEL_QUALITY_THRESHOLDS, FEATURE_CONFIG
from src.utils.logger import get_model_logger
from src.utils.helpers import safe_divide, calculate_growth_metrics, format_large_numbers

logger = get_model_logger()
warnings.filterwarnings('ignore')


# =============================================================================
# CORE KPI EVALUATOR
# =============================================================================

class CreatorKPIEvaluator:
    """Comprehensive KPI evaluation for creator growth prediction models."""
    
    def __init__(self, business_context: Optional[Dict[str, Any]] = None):
        """Initialize KPI evaluator.
        
        Args:
            business_context: Business context information (optional)
        """
        self.business_context = business_context or {}
        self.kpi_results = {}
        self.benchmark_metrics = {}
        
        logger.info("Initialized Creator KPI Evaluator")
    
    def calculate_all_kpis(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          baseline_pred: Optional[np.ndarray] = None,
                          dates: Optional[np.ndarray] = None,
                          posting_frequency: Optional[np.ndarray] = None,
                          roi_actual: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate comprehensive KPIs for creator growth prediction.
        
        Args:
            y_true: Actual growth values
            y_pred: Predicted growth values
            baseline_pred: Baseline predictions for comparison (optional)
            dates: Date array for temporal analysis (optional)
            posting_frequency: Weekly posting frequency values (optional)
            roi_actual: Actual ROI values (optional)
            
        Returns:
            Dictionary with all KPI results
        """
        logger.info("Calculating comprehensive KPIs")
        
        # 1. Statistical Accuracy Metrics
        statistical_kpis = self.calculate_statistical_kpis(y_true, y_pred)
        
        # 2. Business Impact Metrics
        business_kpis = self.calculate_business_kpis(
            y_true, y_pred, posting_frequency, roi_actual
        )
        
        # 3. Prediction Quality Metrics
        quality_kpis = self.calculate_prediction_quality_kpis(y_true, y_pred)
        
        # 4. Temporal Performance Metrics
        temporal_kpis = {}
        if dates is not None:
            temporal_kpis = self.calculate_temporal_kpis(y_true, y_pred, dates)
        
        # 5. Comparative Performance (vs baseline)
        comparative_kpis = {}
        if baseline_pred is not None:
            comparative_kpis = self.calculate_comparative_kpis(
                y_true, y_pred, baseline_pred
            )
        
        # 6. Creator-Specific KPIs
        creator_kpis = self.calculate_creator_specific_kpis(
            y_true, y_pred, posting_frequency
        )
        
        # Compile all results
        self.kpi_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'statistical_kpis': statistical_kpis,
            'business_kpis': business_kpis,
            'quality_kpis': quality_kpis,
            'temporal_kpis': temporal_kpis,
            'comparative_kpis': comparative_kpis,
            'creator_kpis': creator_kpis,
            'overall_score': self._calculate_overall_score()
        }
        
        logger.info(f"KPI calculation completed - Overall Score: {self.kpi_results['overall_score']:.2f}")
        return self.kpi_results
    
    def calculate_statistical_kpis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate standard statistical accuracy metrics."""
        logger.debug("Calculating statistical KPIs")
        
        # Core regression metrics
        kpis = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
            'median_absolute_error': median_absolute_error(y_true, y_pred)
        }
        
        # Additional statistical metrics
        residuals = y_true - y_pred
        
        # Mean Absolute Percentage Error
        kpis['mape'] = np.mean(np.abs(safe_divide(residuals, y_true))) * 100
        
        # Symmetric Mean Absolute Percentage Error
        kpis['smape'] = np.mean(np.abs(residuals) / (np.abs(y_true) + np.abs(y_pred))) * 200
        
        # Mean Bias Error (systematic error)
        kpis['mean_bias_error'] = np.mean(y_pred - y_true)
        
        # Normalized metrics
        target_std = np.std(y_true)
        if target_std > 0:
            kpis['normalized_rmse'] = kpis['rmse'] / target_std
            kpis['normalized_mae'] = kpis['mae'] / target_std
        
        # Correlation coefficient
        kpis['pearson_correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
        
        return kpis
    
    def calculate_business_kpis(self,
                               y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               posting_frequency: Optional[np.ndarray] = None,
                               roi_actual: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate business impact and ROI metrics."""
        logger.debug("Calculating business KPIs")
        
        business_kpis = {}
        
        # Directional accuracy (most important for business decisions)
        direction_correct = np.sign(y_true) == np.sign(y_pred)
        business_kpis['directional_accuracy'] = np.mean(direction_correct) * 100
        
        # Growth prediction accuracy by magnitude
        growth_ranges = [
            ('negative', y_true < 0),
            ('low_growth', (y_true >= 0) & (y_true < 100)),
            ('medium_growth', (y_true >= 100) & (y_true < 500)),
            ('high_growth', y_true >= 500)
        ]
        
        range_accuracy = {}
        for range_name, mask in growth_ranges:
            if np.any(mask):
                range_accuracy[f'{range_name}_accuracy'] = np.mean(
                    direction_correct[mask]
                ) * 100
                range_accuracy[f'{range_name}_count'] = np.sum(mask)
        
        business_kpis['growth_range_accuracy'] = range_accuracy
        
        # Economic value metrics
        if posting_frequency is not None:
            # Strategy optimization accuracy
            posting_ranges = [
                ('low_posting', posting_frequency < 3),
                ('medium_posting', (posting_frequency >= 3) & (posting_frequency < 7)),
                ('high_posting', posting_frequency >= 7)
            ]
            
            strategy_accuracy = {}
            for range_name, mask in posting_ranges:
                if np.any(mask):
                    strategy_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                    strategy_accuracy[f'{range_name}_mae'] = strategy_mae
                    strategy_accuracy[f'{range_name}_samples'] = np.sum(mask)
            
            business_kpis['strategy_accuracy'] = strategy_accuracy
        
        # ROI prediction accuracy
        if roi_actual is not None:
            roi_pred = safe_divide(y_pred, posting_frequency) if posting_frequency is not None else None
            if roi_pred is not None:
                business_kpis['roi_prediction_error'] = mean_absolute_error(roi_actual, roi_pred)
                business_kpis['roi_correlation'] = np.corrcoef(roi_actual, roi_pred)[0, 1]
        
        # Decision support metrics
        business_kpis['decision_support'] = self._calculate_decision_support_metrics(
            y_true, y_pred, posting_frequency
        )
        
        return business_kpis
    
    def calculate_prediction_quality_kpis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate prediction quality and reliability metrics."""
        logger.debug("Calculating prediction quality KPIs")
        
        residuals = y_true - y_pred
        abs_residuals = np.abs(residuals)
        
        quality_kpis = {
            # Prediction intervals and uncertainty
            'prediction_std': np.std(y_pred),
            'residual_std': np.std(residuals),
            'coefficient_of_variation': np.std(y_pred) / np.max(np.abs(np.mean(y_pred)), 1e-10),
            
            # Prediction stability
            'max_absolute_error': np.max(abs_residuals),
            'min_absolute_error': np.min(abs_residuals),
            'error_range': np.max(abs_residuals) - np.min(abs_residuals),
            
            # Percentile-based metrics
            'mae_95th_percentile': np.percentile(abs_residuals, 95),
            'mae_75th_percentile': np.percentile(abs_residuals, 75),
            'mae_25th_percentile': np.percentile(abs_residuals, 25),
            
            # Consistency metrics
            'prediction_consistency': 1 - (np.std(abs_residuals) / np.max(np.mean(abs_residuals), 1e-10))
        }
        
        # Prediction confidence analysis
        quality_kpis['confidence_metrics'] = self._calculate_confidence_metrics(y_true, y_pred)
        
        return quality_kpis
    
    def calculate_temporal_kpis(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               dates: np.ndarray) -> Dict[str, Any]:
        """Calculate temporal performance KPIs."""
        logger.debug("Calculating temporal KPIs")
        
        # Convert dates if needed
        if not isinstance(dates[0], (pd.Timestamp, datetime)):
            dates = pd.to_datetime(dates)
        
        # Create temporal dataframe
        temporal_df = pd.DataFrame({
            'date': dates,
            'actual': y_true,
            'predicted': y_pred,
            'error': np.abs(y_true - y_pred)
        }).sort_values('date')
        
        # Monthly performance
        temporal_df['month'] = temporal_df['date'].dt.month
        monthly_performance = temporal_df.groupby('month')['error'].mean()
        
        # Performance over time
        temporal_df['days_elapsed'] = (temporal_df['date'] - temporal_df['date'].min()).dt.days
        
        # Trend in performance
        if len(temporal_df) > 2:
            performance_trend_corr = np.corrcoef(
                temporal_df['days_elapsed'], temporal_df['error']
            )[0, 1]
        else:
            performance_trend_corr = 0
        
        # Seasonal consistency
        seasonal_cv = monthly_performance.std() / monthly_performance.mean()
        
        temporal_kpis = {
            'monthly_mae': monthly_performance.to_dict(),
            'seasonal_consistency': seasonal_cv,
            'performance_trend_correlation': performance_trend_corr,
            'best_month': monthly_performance.idxmin(),
            'worst_month': monthly_performance.idxmax(),
            'monthly_performance_range': monthly_performance.max() - monthly_performance.min()
        }
        
        # Long-term stability
        if len(temporal_df) >= 30:  # At least 30 data points
            # Split into first/last halves
            mid_point = len(temporal_df) // 2
            first_half_mae = temporal_df['error'].iloc[:mid_point].mean()
            last_half_mae = temporal_df['error'].iloc[mid_point:].mean()
            
            temporal_kpis['long_term_stability'] = {
                'first_half_mae': first_half_mae,
                'last_half_mae': last_half_mae,
                'performance_change': (last_half_mae - first_half_mae) / first_half_mae,
                'stable': abs(last_half_mae - first_half_mae) / first_half_mae < 0.2
            }
        
        return temporal_kpis
    
    def calculate_comparative_kpis(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray, 
                                  baseline_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate comparative performance vs baseline."""
        logger.debug("Calculating comparative KPIs")
        
        # Calculate metrics for both models
        model_mae = mean_absolute_error(y_true, y_pred)
        model_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        model_r2 = r2_score(y_true, y_pred)
        
        baseline_mae = mean_absolute_error(y_true, baseline_pred)
        baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
        baseline_r2 = r2_score(y_true, baseline_pred)
        
        # Improvement metrics
        comparative_kpis = {
            'mae_improvement': (baseline_mae - model_mae) / baseline_mae * 100,
            'rmse_improvement': (baseline_rmse - model_rmse) / baseline_rmse * 100,
            'r2_improvement': (model_r2 - baseline_r2) / max(abs(baseline_r2), 1e-10) * 100,
            
            'model_better_mae': model_mae < baseline_mae,
            'model_better_rmse': model_rmse < baseline_rmse,
            'model_better_r2': model_r2 > baseline_r2,
            
            'baseline_metrics': {
                'mae': baseline_mae,
                'rmse': baseline_rmse,
                'r2': baseline_r2
            },
            'model_metrics': {
                'mae': model_mae,
                'rmse': model_rmse,
                'r2': model_r2
            }
        }
        
        # Overall improvement score
        improvements = [
            comparative_kpis['mae_improvement'] > 0,
            comparative_kpis['rmse_improvement'] > 0,  
            comparative_kpis['r2_improvement'] > 0
        ]
        comparative_kpis['overall_improvement_score'] = sum(improvements) / len(improvements)
        
        return comparative_kpis
    
    def calculate_creator_specific_kpis(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       posting_frequency: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate creator-specific KPIs relevant to growth strategy."""
        logger.debug("Calculating creator-specific KPIs")
        
        creator_kpis = {}
        
        # Growth prediction categories
        growth_categories = self._categorize_growth_predictions(y_true, y_pred)
        creator_kpis['growth_categories'] = growth_categories
        
        # Strategy recommendation accuracy
        if posting_frequency is not None:
            strategy_kpis = self._calculate_strategy_kpis(y_true, y_pred, posting_frequency)
            creator_kpis['strategy_kpis'] = strategy_kpis
        
        # Growth momentum prediction
        if len(y_true) > 1:
            momentum_kpis = self._calculate_momentum_kpis(y_true, y_pred)
            creator_kpis['momentum_kpis'] = momentum_kpis
        
        # Threshold-based accuracy (important growth milestones)
        threshold_kpis = self._calculate_threshold_accuracy(y_true, y_pred)
        creator_kpis['threshold_accuracy'] = threshold_kpis
        
        return creator_kpis
    
    def _calculate_decision_support_metrics(self,
                                          y_true: np.ndarray,
                                          y_pred: np.ndarray,
                                          posting_frequency: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate metrics for decision support quality."""
        decision_metrics = {}
        
        # Prediction confidence for different scenarios
        residuals = np.abs(y_true - y_pred)
        
        if posting_frequency is not None:
            # Accuracy by posting strategy
            low_posting = posting_frequency < 5
            high_posting = posting_frequency >= 5
            
            if np.any(low_posting):
                decision_metrics['low_posting_mae'] = np.mean(residuals[low_posting])
            if np.any(high_posting):
                decision_metrics['high_posting_mae'] = np.mean(residuals[high_posting])
        
        # Strategic decision accuracy (growth vs decline)
        positive_growth_mask = y_true > 0
        negative_growth_mask = y_true <= 0
        
        if np.any(positive_growth_mask):
            decision_metrics['positive_growth_accuracy'] = np.mean(
                (y_pred > 0)[positive_growth_mask]
            ) * 100
        
        if np.any(negative_growth_mask):
            decision_metrics['negative_growth_accuracy'] = np.mean(
                (y_pred <= 0)[negative_growth_mask]
            ) * 100
        
        return decision_metrics
    
    def _calculate_confidence_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction confidence metrics."""
        residuals = y_true - y_pred
        
        # Prediction interval coverage (approximate)
        residual_std = np.std(residuals)
        
        # Percentage of predictions within 1, 2, 3 standard deviations
        within_1_std = np.mean(np.abs(residuals) <= residual_std) * 100
        within_2_std = np.mean(np.abs(residuals) <= 2 * residual_std) * 100
        within_3_std = np.mean(np.abs(residuals) <= 3 * residual_std) * 100
        
        return {
            'within_1_std_percentage': within_1_std,
            'within_2_std_percentage': within_2_std,
            'within_3_std_percentage': within_3_std,
            'prediction_uncertainty': residual_std,
            'confidence_68': within_1_std >= 68,  # Expected for normal distribution
            'confidence_95': within_2_std >= 95   # Expected for normal distribution
        }
    
    def _categorize_growth_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Categorize growth predictions for creator insights."""
        categories = {
            'explosive_growth': (y_true >= 1000) | (y_pred >= 1000),
            'strong_growth': ((y_true >= 500) & (y_true < 1000)) | ((y_pred >= 500) & (y_pred < 1000)),
            'steady_growth': ((y_true >= 100) & (y_true < 500)) | ((y_pred >= 100) & (y_pred < 500)),
            'slow_growth': ((y_true >= 0) & (y_true < 100)) | ((y_pred >= 0) & (y_pred < 100)),
            'decline': (y_true < 0) | (y_pred < 0)
        }
        
        category_accuracy = {}
        for category, mask in categories.items():
            if np.any(mask):
                category_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                category_accuracy[category] = {
                    'count': np.sum(mask),
                    'percentage': np.mean(mask) * 100,
                    'mae': category_mae,
                    'accuracy_score': 1 / (1 + category_mae / 100)  # Normalized accuracy
                }
        
        return category_accuracy
    
    def _calculate_strategy_kpis(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray, 
                                posting_frequency: np.ndarray) -> Dict[str, Any]:
        """Calculate strategy-related KPIs."""
        # ROI prediction accuracy
        actual_roi = safe_divide(y_true, posting_frequency)
        predicted_roi = safe_divide(y_pred, posting_frequency)
        
        roi_mae = mean_absolute_error(actual_roi, predicted_roi)
        
        # Strategy effectiveness prediction
        high_freq_mask = posting_frequency > np.median(posting_frequency)
        low_freq_mask = ~high_freq_mask
        
        strategy_kpis = {
            'roi_prediction_mae': roi_mae,
            'roi_correlation': np.corrcoef(actual_roi, predicted_roi)[0, 1],
        }
        
        if np.any(high_freq_mask):
            strategy_kpis['high_frequency_mae'] = mean_absolute_error(
                y_true[high_freq_mask], y_pred[high_freq_mask]
            )
        
        if np.any(low_freq_mask):
            strategy_kpis['low_frequency_mae'] = mean_absolute_error(
                y_true[low_freq_mask], y_pred[low_freq_mask]
            )
        
        return strategy_kpis
    
    def _calculate_momentum_kpis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate growth momentum prediction KPIs."""
        # Growth acceleration (second derivative approximation)
        if len(y_true) >= 3:
            actual_accel = np.diff(np.diff(y_true))
            pred_accel = np.diff(np.diff(y_pred))
            
            momentum_kpis = {
                'acceleration_mae': mean_absolute_error(actual_accel, pred_accel),
                'acceleration_correlation': np.corrcoef(actual_accel, pred_accel)[0, 1],
                'momentum_direction_accuracy': np.mean(
                    np.sign(actual_accel) == np.sign(pred_accel)
                ) * 100
            }
        else:
            momentum_kpis = {
                'acceleration_mae': np.nan,
                'acceleration_correlation': np.nan,
                'momentum_direction_accuracy': np.nan
            }
        
        return momentum_kpis
    
    def _calculate_threshold_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate accuracy at important growth thresholds."""
        thresholds = [0, 50, 100, 200, 500, 1000]
        threshold_accuracy = {}
        
        for threshold in thresholds:
            # Accuracy in predicting whether growth exceeds threshold
            actual_exceeds = y_true > threshold
            pred_exceeds = y_pred > threshold
            
            if np.any(actual_exceeds) or np.any(pred_exceeds):
                accuracy = np.mean(actual_exceeds == pred_exceeds) * 100
                threshold_accuracy[f'threshold_{threshold}'] = {
                    'accuracy': accuracy,
                    'actual_exceeds_count': np.sum(actual_exceeds),
                    'pred_exceeds_count': np.sum(pred_exceeds)
                }
        
        return threshold_accuracy
    
    def _calculate_overall_score(self) -> float:
        """Calculate overall KPI score (0-1)."""
        if not self.kpi_results:
            return 0.0
        
        score_components = []
        
        # Statistical performance (40%)
        if 'statistical_kpis' in self.kpi_results:
            r2 = self.kpi_results['statistical_kpis'].get('r2', 0)
            statistical_score = max(0, min(1, (r2 + 1) / 2))  # Normalize R² to 0-1
            score_components.append((statistical_score, 0.4))
        
        # Business impact (30%)
        if 'business_kpis' in self.kpi_results:
            directional_acc = self.kpi_results['business_kpis'].get('directional_accuracy', 0)
            business_score = directional_acc / 100
            score_components.append((business_score, 0.3))
        
        # Prediction quality (20%)
        if 'quality_kpis' in self.kpi_results:
            consistency = self.kpi_results['quality_kpis'].get('prediction_consistency', 0)
            quality_score = max(0, min(1, consistency))
            score_components.append((quality_score, 0.2))
        
        # Temporal stability (10%)
        if 'temporal_kpis' in self.kpi_results:
            seasonal_consistency = self.kpi_results['temporal_kpis'].get('seasonal_consistency', 1)
            temporal_score = max(0, min(1, 1 - seasonal_consistency))  # Lower CV is better
            score_components.append((temporal_score, 0.1))
        
        # Calculate weighted average
        if score_components:
            total_score = sum(score * weight for score, weight in score_components)
            total_weight = sum(weight for _, weight in score_components)
            return total_score / total_weight
        
        return 0.0
    
    def get_kpi_summary(self) -> Dict[str, Any]:
        """Get a summary of key KPIs."""
        if not self.kpi_results:
            return {'status': 'No KPIs calculated yet'}
        
        summary = {
            'overall_score': self.kpi_results['overall_score'],
            'key_metrics': {},
            'performance_grade': self._get_performance_grade(),
            'recommendations': self._generate_kpi_recommendations()
        }
        
        # Extract key metrics
        if 'statistical_kpis' in self.kpi_results:
            stats = self.kpi_results['statistical_kpis']
            summary['key_metrics'].update({
                'r2': stats.get('r2', 0),
                'rmse': stats.get('rmse', 0),
                'mae': stats.get('mae', 0)
            })
        
        if 'business_kpis' in self.kpi_results:
            business = self.kpi_results['business_kpis']
            summary['key_metrics']['directional_accuracy'] = business.get('directional_accuracy', 0)
        
        return summary
    
    def _get_performance_grade(self) -> str:
        """Get performance grade based on overall score."""
        score = self.kpi_results.get('overall_score', 0)
        
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B+'
        elif score >= 0.6:
            return 'B'
        elif score >= 0.5:
            return 'C+'
        elif score >= 0.4:
            return 'C'
        else:
            return 'D'
    
    def _generate_kpi_recommendations(self) -> List[str]:
        """Generate recommendations based on KPI results."""
        recommendations = []
        
        if not self.kpi_results:
            return ['Calculate KPIs first']
        
        # Statistical performance recommendations
        if 'statistical_kpis' in self.kpi_results:
            stats = self.kpi_results['statistical_kpis']
            if stats.get('r2', 0) < 0.7:
                recommendations.append("Consider feature engineering to improve R² score")
            if stats.get('mape', 100) > 30:
                recommendations.append("High MAPE indicates need for better feature selection")
        
        # Business impact recommendations
        if 'business_kpis' in self.kpi_results:
            business = self.kpi_results['business_kpis']
            if business.get('directional_accuracy', 0) < 75:
                recommendations.append("Improve directional accuracy for better decision support")
        
        # Temporal recommendations
        if 'temporal_kpis' in self.kpi_results:
            temporal = self.kpi_results['temporal_kpis']
            stability = temporal.get('long_term_stability', {})
            if not stability.get('stable', True):
                recommendations.append("Model showing instability over time - consider retraining")
        
        return recommendations if recommendations else ["KPI performance looks good"]


# =============================================================================
# BENCHMARK COMPARISON
# =============================================================================

def compare_against_benchmarks(kpi_results: Dict[str, Any],
                             benchmark_metrics: Dict[str, float]) -> Dict[str, Any]:
    """Compare KPI results against industry benchmarks.
    
    Args:
        kpi_results: KPI results from evaluator
        benchmark_metrics: Industry benchmark metrics
        
    Returns:
        Dictionary with benchmark comparison
    """
    logger.info("Comparing against industry benchmarks")
    
    comparison = {
        'benchmark_comparison': {},
        'performance_vs_benchmark': 'unknown',
        'benchmark_score': 0.0
    }
    
    if 'statistical_kpis' not in kpi_results:
        return comparison
    
    stats = kpi_results['statistical_kpis']
    better_count = 0
    total_comparisons = 0
    
    # Compare key metrics
    metric_comparisons = {
        'r2': ('higher_better', stats.get('r2', 0)),
        'rmse': ('lower_better', stats.get('rmse', float('inf'))),
        'mae': ('lower_better', stats.get('mae', float('inf'))),
        'mape': ('lower_better', stats.get('mape', float('inf')))
    }
    
    for metric, (direction, our_value) in metric_comparisons.items():
        if metric in benchmark_metrics:
            benchmark_value = benchmark_metrics[metric]
            
            if direction == 'higher_better':
                better = our_value > benchmark_value
                improvement = ((our_value - benchmark_value) / benchmark_value * 100) if benchmark_value != 0 else 0
            else:  # lower_better
                better = our_value < benchmark_value
                improvement = ((benchmark_value - our_value) / benchmark_value * 100) if benchmark_value != 0 else 0
            
            comparison['benchmark_comparison'][metric] = {
                'our_value': our_value,
                'benchmark_value': benchmark_value,
                'better_than_benchmark': better,
                'improvement_percentage': improvement
            }
            
            if better:
                better_count += 1
            total_comparisons += 1
    
    # Overall performance assessment
    if total_comparisons > 0:
        benchmark_score = better_count / total_comparisons
        comparison['benchmark_score'] = benchmark_score
        
        if benchmark_score >= 0.75:
            comparison['performance_vs_benchmark'] = 'excellent'
        elif benchmark_score >= 0.5:
            comparison['performance_vs_benchmark'] = 'good'
        elif benchmark_score >= 0.25:
            comparison['performance_vs_benchmark'] = 'average'
        else:
            comparison['performance_vs_benchmark'] = 'below_average'
    
    return comparison


# =============================================================================
# EXPORT LIST
# =============================================================================

__all__ = [
    'CreatorKPIEvaluator',
    'compare_against_benchmarks'
]
