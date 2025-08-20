"""
Creator Growth Navigator - Model Diagnostics Module

Comprehensive model diagnostics including residual analysis, assumption testing,
drift detection, and model health monitoring for creator growth prediction models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from datetime import datetime, timedelta

# Import from project modules
from src.config.config import MODEL_QUALITY_THRESHOLDS, DATA_QUALITY_THRESHOLDS
from src.utils.logger import get_model_logger
from src.utils.helpers import safe_divide, calculate_confidence_interval

logger = get_model_logger()
warnings.filterwarnings('ignore')


# =============================================================================
# CORE DIAGNOSTICS CLASS
# =============================================================================

class ModelDiagnostics:
    """Comprehensive model diagnostics for creator growth prediction."""
    
    def __init__(self, model=None, confidence_level: float = 0.95):
        """Initialize model diagnostics.
        
        Args:
            model: Trained model instance (optional, can be set later)
            confidence_level: Confidence level for statistical tests
        """
        self.model = model
        self.confidence_level = confidence_level
        self.diagnostic_results = {}
        self.residual_analysis = {}
        self.assumption_tests = {}
        self.drift_analysis = {}
        
        logger.info("Initialized model diagnostics")
    
    def run_full_diagnostics(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           X: Optional[np.ndarray] = None,
                           feature_names: Optional[List[str]] = None,
                           dates: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run complete diagnostic analysis.
        
        Args:
            y_true: Actual target values
            y_pred: Predicted target values
            X: Feature matrix (optional, for advanced diagnostics)
            feature_names: Feature names (optional)
            dates: Date array for temporal analysis (optional)
            
        Returns:
            Dictionary with all diagnostic results
        """
        logger.info("Running comprehensive model diagnostics")
        
        # Basic residual analysis
        self.residual_analysis = self.analyze_residuals(y_true, y_pred)
        
        # Statistical assumption tests
        self.assumption_tests = self.test_assumptions(y_true, y_pred, X)
        
        # Performance metrics
        performance_metrics = self.calculate_performance_metrics(y_true, y_pred)
        
        # Outlier analysis
        outlier_analysis = self.detect_outliers(y_true, y_pred)
        
        # Temporal analysis (if dates provided)
        temporal_analysis = {}
        if dates is not None:
            temporal_analysis = self.analyze_temporal_patterns(y_true, y_pred, dates)
        
        # Feature diagnostics (if X provided)
        feature_diagnostics = {}
        if X is not None:
            feature_diagnostics = self.analyze_feature_diagnostics(X, feature_names)
        
        # Compile all results
        self.diagnostic_results = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': performance_metrics,
            'residual_analysis': self.residual_analysis,
            'assumption_tests': self.assumption_tests,
            'outlier_analysis': outlier_analysis,
            'temporal_analysis': temporal_analysis,
            'feature_diagnostics': feature_diagnostics,
            'overall_health_score': self._calculate_health_score()
        }
        
        logger.info(f"Diagnostics completed - Health Score: {self.diagnostic_results['overall_health_score']:.2f}")
        return self.diagnostic_results
    
    def analyze_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Comprehensive residual analysis.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary with residual analysis results
        """
        logger.debug("Analyzing model residuals")
        
        residuals = y_true - y_pred
        
        # Basic statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'median': np.median(residuals),
            'q25': np.percentile(residuals, 25),
            'q75': np.percentile(residuals, 75),
            'iqr': np.percentile(residuals, 75) - np.percentile(residuals, 25)
        }
        
        # Residual patterns
        residual_patterns = {
            'mean_near_zero': abs(residual_stats['mean']) < 0.01 * np.std(y_true),
            'symmetric_distribution': abs(residual_stats['mean'] - residual_stats['median']) < 0.1 * residual_stats['std'],
            'constant_variance': self._test_homoscedasticity(y_pred, residuals),
            'no_autocorrelation': self._test_autocorrelation(residuals)
        }
        
        # Normality tests
        normality_tests = self._test_residual_normality(residuals)
        
        return {
            'residual_statistics': residual_stats,
            'residual_patterns': residual_patterns,
            'normality_tests': normality_tests,
            'residuals': residuals  # Store for plotting
        }
    
    def test_assumptions(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        X: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Test linear regression assumptions.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            X: Feature matrix (optional)
            
        Returns:
            Dictionary with assumption test results
        """
        logger.debug("Testing linear regression assumptions")
        
        residuals = y_true - y_pred
        
        assumption_results = {}
        
        # 1. Linearity assumption (via residuals vs fitted)
        linearity_test = self._test_linearity(y_pred, residuals)
        assumption_results['linearity'] = linearity_test
        
        # 2. Independence assumption (Durbin-Watson test)
        independence_test = self._test_independence(residuals)
        assumption_results['independence'] = independence_test
        
        # 3. Homoscedasticity (constant variance)
        homoscedasticity_test = self._test_homoscedasticity(y_pred, residuals)
        assumption_results['homoscedasticity'] = homoscedasticity_test
        
        # 4. Normality of residuals
        normality_test = self._test_residual_normality(residuals)
        assumption_results['normality'] = normality_test
        
        # 5. Multicollinearity (if X provided)
        if X is not None and X.shape[1] > 1:
            multicollinearity_test = self._test_multicollinearity(X)
            assumption_results['multicollinearity'] = multicollinearity_test
        
        # Overall assumption score
        passed_tests = sum([
            assumption_results['linearity']['passed'],
            assumption_results['independence']['passed'],
            assumption_results['homoscedasticity']['passed'],
            assumption_results['normality']['passed']
        ])
        total_tests = 4
        
        assumption_results['overall_score'] = passed_tests / total_tests
        assumption_results['summary'] = f"{passed_tests}/{total_tests} assumptions satisfied"
        
        return assumption_results
    
    def _test_linearity(self, y_pred: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """Test linearity assumption using residuals vs fitted plot analysis."""
        # Test if residuals show patterns when plotted against fitted values
        correlation = np.corrcoef(y_pred, residuals)[0, 1]
        
        # Test for quadratic relationship
        poly_fit = np.polyfit(y_pred, residuals, 2)
        quadratic_coeff = abs(poly_fit[0])
        
        return {
            'correlation_with_fitted': correlation,
            'quadratic_coefficient': quadratic_coeff,
            'passed': abs(correlation) < 0.1 and quadratic_coeff < 0.01,
            'interpretation': 'Linear relationship' if abs(correlation) < 0.1 else 'Non-linear patterns detected'
        }
    
    def _test_independence(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test independence using Durbin-Watson test."""
        try:
            # Durbin-Watson test for autocorrelation
            from statsmodels.stats.diagnostic import durbin_watson
            dw_statistic = durbin_watson(residuals)
            
            # Rule of thumb: DW around 2 indicates no autocorrelation
            # DW < 1.5 or DW > 2.5 suggests autocorrelation
            passed = 1.5 <= dw_statistic <= 2.5
            
            if dw_statistic < 1.5:
                interpretation = 'Positive autocorrelation detected'
            elif dw_statistic > 2.5:
                interpretation = 'Negative autocorrelation detected'
            else:
                interpretation = 'No significant autocorrelation'
                
        except ImportError:
            # Fallback: simple lag-1 autocorrelation
            if len(residuals) > 1:
                lag1_corr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
                dw_statistic = 2 * (1 - lag1_corr)  # Approximation
                passed = abs(lag1_corr) < 0.2
                interpretation = 'Independence test (simplified)'
            else:
                dw_statistic = 2.0
                passed = True
                interpretation = 'Insufficient data for test'
        
        return {
            'durbin_watson_statistic': dw_statistic,
            'passed': passed,
            'interpretation': interpretation
        }
    
    def _test_homoscedasticity(self, y_pred: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """Test homoscedasticity (constant variance) using Breusch-Pagan test."""
        try:
            # Breusch-Pagan test
            from scipy.stats import pearsonr
            
            # Test correlation between |residuals| and fitted values
            abs_residuals = np.abs(residuals)
            correlation, p_value = pearsonr(y_pred, abs_residuals)
            
            # If correlation is significant, heteroscedasticity is present
            passed = p_value > 0.05
            
            interpretation = 'Constant variance' if passed else 'Non-constant variance (heteroscedasticity)'
            
        except:
            # Fallback: simple variance ratio test
            mid_point = len(y_pred) // 2
            sorted_indices = np.argsort(y_pred)
            
            first_half_var = np.var(residuals[sorted_indices[:mid_point]])
            second_half_var = np.var(residuals[sorted_indices[mid_point:]])
            
            variance_ratio = max(first_half_var, second_half_var) / max(min(first_half_var, second_half_var), 1e-10)
            
            passed = variance_ratio < 3.0  # Rule of thumb
            correlation = np.nan
            p_value = np.nan
            interpretation = 'Variance ratio test'
        
        return {
            'correlation_abs_residuals_fitted': correlation,
            'p_value': p_value,
            'passed': passed,
            'interpretation': interpretation
        }
    
    def _test_residual_normality(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test normality of residuals using multiple tests."""
        normality_tests = {}
        
        # Shapiro-Wilk test (for smaller samples)
        if len(residuals) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            normality_tests['shapiro_wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'passed': shapiro_p > 0.05
            }
        
        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(residuals)
        normality_tests['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_p,
            'passed': jb_p > 0.05
        }
        
        # Anderson-Darling test
        ad_stat, ad_critical, ad_significance = stats.anderson(residuals, dist='norm')
        # Use 5% significance level (index 2)
        normality_tests['anderson_darling'] = {
            'statistic': ad_stat,
            'critical_value_5pct': ad_critical[2],
            'passed': ad_stat < ad_critical[1]
        }
        
        # Overall normality assessment
        tests_passed = sum([
            test_result['passed'] for test_result in normality_tests.values()
        ])
        total_tests = len(normality_tests)
        
        normality_tests['overall'] = {
            'tests_passed': tests_passed,
            'total_tests': total_tests,
            'passed': tests_passed >= total_tests // 2,  # Majority rule
            'interpretation': 'Normal distribution' if tests_passed >= total_tests // 2 else 'Non-normal distribution'
        }
        
        return normality_tests
    
    def _test_multicollinearity(self, X: np.ndarray) -> Dict[str, Any]:
        """Test for multicollinearity using correlation matrix and VIF."""
        correlation_matrix = np.corrcoef(X.T)
        
        # Find high correlations
        high_correlations = []
        n_features = X.shape[1]
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                corr = abs(correlation_matrix[i, j])
                if corr > 0.8:  # High correlation threshold
                    high_correlations.append({
                        'feature_1': i,
                        'feature_2': j,
                        'correlation': corr
                    })
        
        # Calculate condition number (matrix health)
        try:
            eigenvalues = np.linalg.eigvals(correlation_matrix)
            condition_number = np.max(eigenvalues) / np.max(np.min(eigenvalues), 1e-10)
        except:
            condition_number = np.inf
        
        multicollinearity_detected = len(high_correlations) > 0 or condition_number > 30
        
        return {
            'high_correlations': high_correlations,
            'condition_number': condition_number,
            'passed': not multicollinearity_detected,
            'interpretation': 'No multicollinearity' if not multicollinearity_detected else 'Multicollinearity detected'
        }
    
    def detect_outliers(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Detect outliers in predictions and residuals.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary with outlier analysis
        """
        logger.debug("Detecting outliers in predictions")
        
        residuals = y_true - y_pred
        
        # Standardized residuals
        std_residuals = residuals / np.std(residuals)
        
        # Different outlier detection methods
        outlier_methods = {}
        
        # 1. Z-score method (standardized residuals)
        z_score_outliers = np.abs(std_residuals) > 3
        outlier_methods['z_score'] = {
            'outlier_indices': np.where(z_score_outliers)[0],
            'n_outliers': np.sum(z_score_outliers),
            'outlier_percentage': np.mean(z_score_outliers) * 100
        }
        
        # 2. IQR method
        q25, q75 = np.percentile(residuals, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        iqr_outliers = (residuals < lower_bound) | (residuals > upper_bound)
        
        outlier_methods['iqr'] = {
            'outlier_indices': np.where(iqr_outliers)[0],
            'n_outliers': np.sum(iqr_outliers),
            'outlier_percentage': np.mean(iqr_outliers) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        # 3. Cook's distance (influence points) - simplified version
        leverage = 1 / len(y_true)  # Simplified leverage calculation
        cooks_d = (std_residuals ** 2) * leverage / (1 - leverage)
        cooks_threshold = 4 / len(y_true)  # Rule of thumb
        cooks_outliers = cooks_d > cooks_threshold
        
        outlier_methods['cooks_distance'] = {
            'outlier_indices': np.where(cooks_outliers)[0],
            'n_outliers': np.sum(cooks_outliers),
            'outlier_percentage': np.mean(cooks_outliers) * 100,
            'cooks_distances': cooks_d,
            'threshold': cooks_threshold
        }
        
        # Combined outlier assessment
        combined_outliers = z_score_outliers | iqr_outliers | cooks_outliers
        
        return {
            'outlier_methods': outlier_methods,
            'combined_outliers': {
                'outlier_indices': np.where(combined_outliers)[0],
                'n_outliers': np.sum(combined_outliers),
                'outlier_percentage': np.mean(combined_outliers) * 100
            },
            'outlier_summary': {
                'total_outliers': np.sum(combined_outliers),
                'outlier_rate': np.mean(combined_outliers),
                'severe_outliers': np.sum(np.abs(std_residuals) > 4),
                'max_absolute_residual': np.max(np.abs(residuals))
            }
        }
    
    def analyze_temporal_patterns(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 dates: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal patterns in model performance.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Date array
            
        Returns:
            Dictionary with temporal analysis
        """
        logger.debug("Analyzing temporal patterns")
        
        # Convert dates to datetime if needed
        if not isinstance(dates[0], (pd.Timestamp, datetime)):
            dates = pd.to_datetime(dates)
        
        # Create temporal dataframe
        temporal_df = pd.DataFrame({
            'date': dates,
            'actual': y_true,
            'predicted': y_pred,
            'residual': y_true - y_pred,
            'abs_residual': np.abs(y_true - y_pred)
        }).sort_values('date')
        
        # Time-based performance metrics
        temporal_df['month'] = temporal_df['date'].dt.month
        temporal_df['quarter'] = temporal_df['date'].dt.quarter
        temporal_df['week_of_year'] = temporal_df['date'].dt.isocalendar().week
        
        # Monthly performance analysis
        monthly_performance = temporal_df.groupby('month').agg({
            'residual': ['mean', 'std'],
            'abs_residual': 'mean'
        }).round(3)
        
        # Trend analysis
        days_elapsed = (temporal_df['date'] - temporal_df['date'].min()).dt.days
        
        # Test for trends in residuals over time
        if len(days_elapsed) > 2:
            residual_trend_corr = np.corrcoef(days_elapsed, temporal_df['residual'])[0, 1]
            abs_residual_trend_corr = np.corrcoef(days_elapsed, temporal_df['abs_residual'])[0, 1]
        else:
            residual_trend_corr = 0
            abs_residual_trend_corr = 0
        
        # Performance degradation detection
        if len(temporal_df) >= 10:
            # Compare first 20% vs last 20% performance
            split_point_1 = len(temporal_df) // 5
            split_point_2 = 4 * len(temporal_df) // 5
            
            early_mae = temporal_df['abs_residual'].iloc[:split_point_1].mean()
            late_mae = temporal_df['abs_residual'].iloc[split_point_2:].mean()
            
            performance_change = (late_mae - early_mae) / early_mae if early_mae > 0 else 0
        else:
            performance_change = 0
        
        return {
            'monthly_performance': monthly_performance.to_dict(),
            'trend_analysis': {
                'residual_trend_correlation': residual_trend_corr,
                'abs_residual_trend_correlation': abs_residual_trend_corr,
                'performance_degradation': performance_change > 0.2,
                'performance_change_percentage': performance_change * 100
            },
            'temporal_stability': {
                'cv_of_monthly_mae': monthly_performance[('abs_residual', 'mean')].std() / monthly_performance[('abs_residual', 'mean')].mean(),
                'max_monthly_mae': monthly_performance[('abs_residual', 'mean')].max(),
                'min_monthly_mae': monthly_performance[('abs_residual', 'mean')].min()
            }
        }
    
    def analyze_feature_diagnostics(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze feature-related diagnostics.
        
        Args:
            X: Feature matrix
            feature_names: Feature names
            
        Returns:
            Dictionary with feature diagnostics
        """
        logger.debug("Analyzing feature diagnostics")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Feature statistics
        feature_stats = {}
        for i, name in enumerate(feature_names):
            feature_stats[name] = {
                'mean': np.mean(X[:, i]),
                'std': np.std(X[:, i]),
                'min': np.min(X[:, i]),
                'max': np.max(X[:, i]),
                'zero_percentage': np.mean(X[:, i] == 0) * 100,
                'missing_percentage': np.mean(np.isnan(X[:, i])) * 100 if np.any(np.isnan(X[:, i])) else 0
            }
        
        # Feature scaling diagnostics
        feature_scales = [np.std(X[:, i]) for i in range(X.shape[1])]
        scale_ratios = np.max(feature_scales) / np.max(np.min(feature_scales), 1e-10)
        
        return {
            'feature_statistics': feature_stats,
            'scaling_diagnostics': {
                'max_scale_ratio': scale_ratios,
                'scaling_needed': scale_ratios > 10,
                'feature_scales': dict(zip(feature_names, feature_scales))
            },
            'feature_quality': {
                'n_constant_features': sum([stats['std'] < 1e-10 for stats in feature_stats.values()]),
                'n_high_zero_features': sum([stats['zero_percentage'] > 90 for stats in feature_stats.values()]),
                'n_features_with_missing': sum([stats['missing_percentage'] > 0 for stats in feature_stats.values()])
            }
        }
    
    def calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
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
        
        # Mean bias error
        metrics['mean_bias_error'] = np.mean(y_pred - y_true)
        
        return metrics
    
    def _calculate_health_score(self) -> float:
        """Calculate overall model health score (0-1)."""
        if not self.diagnostic_results:
            return 0.0
        
        score = 1.0
        
        # Performance component (40%)
        performance = self.diagnostic_results.get('performance_metrics', {})
        r2 = performance.get('r2', 0)
        if r2 < MODEL_QUALITY_THRESHOLDS.get('min_r2_score', 0.6):
            score -= 0.2
        
        # Assumptions component (30%)
        assumptions = self.diagnostic_results.get('assumption_tests', {})
        assumption_score = assumptions.get('overall_score', 0.5)
        score -= 0.3 * (1 - assumption_score)
        
        # Outlier component (20%)
        outliers = self.diagnostic_results.get('outlier_analysis', {})
        outlier_rate = outliers.get('combined_outliers', {}).get('outlier_percentage', 0) / 100
        if outlier_rate > 0.1:  # More than 10% outliers
            score -= 0.2 * min(outlier_rate, 0.5)
        
        # Temporal stability component (10%)
        temporal = self.diagnostic_results.get('temporal_analysis', {})
        if temporal and temporal.get('trend_analysis', {}).get('performance_degradation', False):
            score -= 0.1
        
        return max(0.0, score)
    
    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """Get a summary of diagnostic results."""
        if not self.diagnostic_results:
            return {'status': 'No diagnostics run yet'}
        
        return {
            'overall_health_score': self.diagnostic_results['overall_health_score'],
            'performance_summary': {
                'r2': self.diagnostic_results['performance_metrics']['r2'],
                'rmse': self.diagnostic_results['performance_metrics']['rmse'],
                'mae': self.diagnostic_results['performance_metrics']['mae']
            },
            'assumption_summary': {
                'assumptions_passed': self.diagnostic_results['assumption_tests']['summary'],
                'overall_score': self.diagnostic_results['assumption_tests']['overall_score']
            },
            'outlier_summary': {
                'outlier_percentage': self.diagnostic_results['outlier_analysis']['combined_outliers']['outlier_percentage'],
                'total_outliers': self.diagnostic_results['outlier_analysis']['combined_outliers']['n_outliers']
            },
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on diagnostic results."""
        recommendations = []
        
        if not self.diagnostic_results:
            return ['Run diagnostics first']
        
        # Performance recommendations
        performance = self.diagnostic_results['performance_metrics']
        if performance['r2'] < 0.7:
            recommendations.append("Consider adding more features or feature engineering")
        
        # Assumption recommendations
        assumptions = self.diagnostic_results['assumption_tests']
        if not assumptions['normality']['overall']['passed']:
            recommendations.append("Consider transforming target variable for normality")
        if not assumptions['homoscedasticity']['passed']:
            recommendations.append("Consider using robust regression or weighted least squares")
        if 'multicollinearity' in assumptions and not assumptions['multicollinearity']['passed']:
            recommendations.append("Remove highly correlated features or use regularization")
        
        # Outlier recommendations
        outliers = self.diagnostic_results['outlier_analysis']
        if outliers['combined_outliers']['outlier_percentage'] > 10:
            recommendations.append("Investigate and potentially remove outliers")
        
        # Temporal recommendations
        temporal = self.diagnostic_results.get('temporal_analysis', {})
        if temporal and temporal.get('trend_analysis', {}).get('performance_degradation', False):
            recommendations.append("Model performance is degrading over time - consider retraining")
        
        return recommendations if recommendations else ["Model diagnostics look good"]


# =============================================================================
# DRIFT DETECTION
# =============================================================================

def detect_model_drift(baseline_metrics: Dict[str, float],
                      current_metrics: Dict[str, float],
                      drift_threshold: float = 0.1) -> Dict[str, Any]:
    """Detect model performance drift.
    
    Args:
        baseline_metrics: Performance metrics from baseline/training
        current_metrics: Current performance metrics
        drift_threshold: Threshold for significant drift
        
    Returns:
        Dictionary with drift analysis
    """
    logger.info("Detecting model performance drift")
    
    drift_analysis = {
        'drift_detected': False,
        'significant_changes': [],
        'metric_changes': {},
        'overall_drift_score': 0.0
    }
    
    # Compare key metrics
    key_metrics = ['r2', 'rmse', 'mae', 'directional_accuracy']
    
    total_drift = 0.0
    
    for metric in key_metrics:
        if metric in baseline_metrics and metric in current_metrics:
            baseline_val = baseline_metrics[metric]
            current_val = current_metrics[metric]
            
            # Calculate relative change
            if baseline_val != 0:
                relative_change = (current_val - baseline_val) / abs(baseline_val)
            else:
                relative_change = 0.0
            
            # For R² and directional accuracy, negative change is bad
            # For RMSE and MAE, positive change is bad
            if metric in ['r2', 'directional_accuracy']:
                drift_score = -relative_change if relative_change < 0 else 0
            else:  # rmse, mae
                drift_score = relative_change if relative_change > 0 else 0
            
            drift_analysis['metric_changes'][metric] = {
                'baseline': baseline_val,
                'current': current_val,
                'relative_change': relative_change,
                'drift_score': drift_score,
                'significant': abs(relative_change) > drift_threshold
            }
            
            total_drift += drift_score
            
            if abs(relative_change) > drift_threshold:
                drift_analysis['significant_changes'].append({
                    'metric': metric,
                    'change': relative_change,
                    'severity': 'high' if abs(relative_change) > 2 * drift_threshold else 'medium'
                })
    
    drift_analysis['overall_drift_score'] = total_drift / len(key_metrics)
    drift_analysis['drift_detected'] = drift_analysis['overall_drift_score'] > drift_threshold
    
    if drift_analysis['drift_detected']:
        logger.warning(f"Model drift detected - Overall drift score: {drift_analysis['overall_drift_score']:.3f}")
    
    return drift_analysis


# =============================================================================
# EXPORT LIST
# =============================================================================

__all__ = [
    'ModelDiagnostics',
    'detect_model_drift'
]
