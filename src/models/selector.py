"""
Creator Growth Navigator - Feature Selection Module

Advanced feature selection utilities including univariate selection, recursive
feature elimination, and wrapper methods optimized for creator growth data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression, 
    RFE, RFECV, VarianceThreshold
)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from datetime import datetime

# Import from project modules
from src.config.config import MODEL_CONFIG, FEATURE_CONFIG
from src.utils.logger import get_model_logger
from src.utils.helpers import safe_divide

logger = get_model_logger()
warnings.filterwarnings('ignore')


# =============================================================================
# COMPREHENSIVE FEATURE SELECTOR
# =============================================================================

class CreatorFeatureSelector:
    """Advanced feature selection for creator growth prediction."""
    
    def __init__(self, 
                 method: str = 'univariate',
                 k_features: int = 10,
                 estimator=None,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """Initialize feature selector.
        
        Args:
            method: Selection method ('univariate', 'recursive', 'wrapper', 'filter')
            k_features: Number of features to select (for fixed methods)
            estimator: Base estimator for recursive/wrapper methods
            cv_folds: Cross-validation folds for time series
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.k_features = k_features
        self.estimator = estimator if estimator else LinearRegression()
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Selection components
        self.selector = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_names = None
        self.selection_results = {}
        
        logger.info(f"Initialized feature selector with method: {method}")
    
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fit feature selector on creator data.
        
        Args:
            df: Training dataframe with your data structure
            
        Returns:
            Dictionary with selection results
        """
        logger.info(f"Fitting feature selector using {self.method} method")
        
        # Prepare features and target
        X, y, feature_names = self._prepare_features(df)
        self.feature_names = feature_names
        
        # Apply appropriate selection method
        if self.method == 'univariate':
            return self._fit_univariate(X, y)
        elif self.method == 'recursive':
            return self._fit_recursive(X, y)
        elif self.method == 'wrapper':
            return self._fit_wrapper(X, y)
        elif self.method == 'filter':
            return self._fit_filter(X, y)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features using same logic as regression models."""
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
    
    def _fit_univariate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit univariate feature selection."""
        logger.debug("Applying univariate feature selection")
        
        # Use f_regression for continuous target
        self.selector = SelectKBest(
            score_func=f_regression,
            k=min(self.k_features, X.shape[1])
        )
        
        self.selector.fit(X, y)
        
        # Get selected features
        selected_indices = self.selector.get_support(indices=True)
        self.selected_features = [self.feature_names[i] for i in selected_indices]
        
        # Get feature scores
        scores = self.selector.scores_
        p_values = self.selector.pvalues_
        
        # Create results
        feature_scores = pd.DataFrame({
            'feature': self.feature_names,
            'score': scores,
            'p_value': p_values,
            'selected': self.selector.get_support()
        }).sort_values('score', ascending=False)
        
        self.selection_results = {
            'method': 'univariate',
            'n_features_selected': len(self.selected_features),
            'selected_features': self.selected_features,
            'feature_scores': feature_scores,
            'selection_criterion': 'f_regression_score'
        }
        
        logger.info(f"Univariate selection: {len(self.selected_features)} features selected")
        return self.selection_results
    
    def _fit_recursive(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit recursive feature elimination."""
        logger.debug("Applying recursive feature elimination")
        
        # Use simple RFE (not RFECV to avoid pipeline issues)
        self.selector = RFE(
            estimator=self.estimator,
            n_features_to_select=min(self.k_features, X.shape[1]),
            step=1
        )
        
        # Scale features for RFE
        X_scaled = self.scaler.fit_transform(X)
        self.selector.fit(X_scaled, y)
        
        # Get selected features
        selected_indices = self.selector.get_support(indices=True)
        self.selected_features = [self.feature_names[i] for i in selected_indices]
        
        # Get feature ranking
        feature_ranking = pd.DataFrame({
            'feature': self.feature_names,
            'ranking': self.selector.ranking_,
            'selected': self.selector.get_support()
        }).sort_values('ranking')
        
        self.selection_results = {
            'method': 'recursive',
            'n_features_selected': len(self.selected_features),
            'selected_features': self.selected_features,
            'feature_ranking': feature_ranking,
            'selection_criterion': 'recursive_elimination'
        }
        
        logger.info(f"Recursive elimination: {len(self.selected_features)} features selected")
        return self.selection_results
    
    def _fit_wrapper(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit wrapper-based selection using forward/backward search."""
        logger.debug("Applying wrapper-based feature selection")
        
        # Implement simple forward selection
        selected_features_idx = []
        remaining_features_idx = list(range(X.shape[1]))
        best_score = float('-inf')
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Forward selection
        for _ in range(min(self.k_features, X.shape[1])):
            best_feature = None
            temp_best_score = best_score
            
            for feature_idx in remaining_features_idx:
                # Test adding this feature
                test_features = selected_features_idx + [feature_idx]
                
                # Cross-validate performance
                tscv = TimeSeriesSplit(n_splits=3)  # Reduced folds for speed
                scores = []
                
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train_fold = X_scaled[train_idx][:, test_features]
                    X_val_fold = X_scaled[val_idx][:, test_features]
                    y_train_fold = y[train_idx]
                    y_val_fold = y[val_idx]
                    
                    # Fit and evaluate
                    temp_model = LinearRegression()
                    temp_model.fit(X_train_fold, y_train_fold)
                    pred = temp_model.predict(X_val_fold)
                    score = r2_score(y_val_fold, pred)
                    scores.append(score)
                
                avg_score = np.mean(scores)
                
                if avg_score > temp_best_score:
                    temp_best_score = avg_score
                    best_feature = feature_idx
            
            if best_feature is not None:
                selected_features_idx.append(best_feature)
                remaining_features_idx.remove(best_feature)
                best_score = temp_best_score
            else:
                break  # No improvement found
        
        # Get selected feature names
        self.selected_features = [self.feature_names[i] for i in selected_features_idx]
        
        # Create feature importance ranking
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'selected': [i in selected_features_idx for i in range(len(self.feature_names))],
            'selection_order': [selected_features_idx.index(i) + 1 if i in selected_features_idx else 0 
                               for i in range(len(self.feature_names))]
        }).sort_values('selection_order')
        
        self.selection_results = {
            'method': 'wrapper',
            'n_features_selected': len(self.selected_features),
            'selected_features': self.selected_features,
            'feature_importance': feature_importance,
            'best_cv_score': best_score,
            'selection_criterion': 'forward_selection_cv'
        }
        
        logger.info(f"Wrapper selection: {len(self.selected_features)} features selected")
        return self.selection_results
    
    def _fit_filter(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit filter-based selection using variance and correlation."""
        logger.debug("Applying filter-based feature selection")
        
        # Step 1: Remove low variance features
        variance_selector = VarianceThreshold(threshold=0.01)
        X_var_filtered = variance_selector.fit_transform(X)
        var_selected_idx = variance_selector.get_support(indices=True)
        
        # Step 2: Remove highly correlated features
        if X_var_filtered.shape[1] > 1:
            corr_matrix = np.corrcoef(X_var_filtered.T)
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    if abs(corr_matrix[i, j]) > 0.95:
                        high_corr_pairs.append((i, j, abs(corr_matrix[i, j])))
            
            # Remove one from each highly correlated pair
            to_remove = set()
            for i, j, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
                if i not in to_remove and j not in to_remove:
                    to_remove.add(j)  # Remove second feature
            
            final_selected_idx = [idx for k, idx in enumerate(var_selected_idx) if k not in to_remove]
        else:
            final_selected_idx = var_selected_idx.tolist()
        
        # Select top k features by correlation with target
        if len(final_selected_idx) > self.k_features:
            target_corrs = []
            for idx in final_selected_idx:
                corr = abs(np.corrcoef(X[:, idx], y)[0, 1])
                target_corrs.append((idx, corr))
            
            target_corrs.sort(key=lambda x: x[1], reverse=True)
            final_selected_idx = [idx for idx, _ in target_corrs[:self.k_features]]
        
        # Get selected feature names
        self.selected_features = [self.feature_names[i] for i in final_selected_idx]
        
        # Create feature analysis
        feature_analysis = []
        for i, feature_name in enumerate(self.feature_names):
            analysis = {
                'feature': feature_name,
                'variance': np.var(X[:, i]),
                'target_correlation': abs(np.corrcoef(X[:, i], y)[0, 1]),
                'selected': i in final_selected_idx
            }
            feature_analysis.append(analysis)
        
        feature_analysis_df = pd.DataFrame(feature_analysis).sort_values('target_correlation', ascending=False)
        
        self.selection_results = {
            'method': 'filter',
            'n_features_selected': len(self.selected_features),
            'selected_features': self.selected_features,
            'feature_analysis': feature_analysis_df,
            'selection_criterion': 'variance_correlation_filter'
        }
        
        logger.info(f"Filter selection: {len(self.selected_features)} features selected")
        return self.selection_results
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe to selected features only.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with only selected features
        """
        if self.selected_features is None:
            raise ValueError("Selector must be fitted before transform")
        
        # Create primary feature if needed
        if 'weekly_posting_frequency' not in df.columns:
            df['weekly_posting_frequency'] = df['posts'] + df['reels'] + df['stories']
        
        # Return selected features
        available_features = [f for f in self.selected_features if f in df.columns]
        return df[available_features]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance/ranking from selection results."""
        if self.selection_results is None:
            raise ValueError("Selector must be fitted first")
        
        method = self.selection_results['method']
        
        if method == 'univariate':
            return self.selection_results['feature_scores']
        elif method == 'recursive':
            return self.selection_results['feature_ranking']
        elif method == 'wrapper':
            return self.selection_results['feature_importance']
        elif method == 'filter':
            return self.selection_results['feature_analysis']
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive selection summary."""
        if self.selection_results is None:
            raise ValueError("Selector must be fitted first")
        
        return {
            'selection_method': self.method,
            'total_features_available': len(self.feature_names),
            'features_selected': len(self.selected_features),
            'selection_ratio': len(self.selected_features) / len(self.feature_names),
            'selected_features': self.selected_features,
            'top_features': self.selected_features[:5] if len(self.selected_features) >= 5 else self.selected_features
        }


# =============================================================================
# COMPARISON AND UTILITIES
# =============================================================================

def compare_selection_methods(df: pd.DataFrame, 
                             methods: List[str] = ['univariate', 'recursive', 'filter'],
                             k_features: int = 8) -> pd.DataFrame:
    """Compare different feature selection methods.
    
    Args:
        df: Training dataframe
        methods: List of selection methods to compare
        k_features: Number of features to select for each method
        
    Returns:
        DataFrame with comparison results
    """
    logger.info(f"Comparing {len(methods)} feature selection methods")
    
    results = []
    
    for method in methods:
        try:
            selector = CreatorFeatureSelector(method=method, k_features=k_features)
            selection_results = selector.fit(df)
            
            result = {
                'method': method,
                'n_features_selected': selection_results['n_features_selected'],
                'selected_features': ', '.join(selection_results['selected_features'][:3]) + '...',
                'selection_criterion': selection_results['selection_criterion']
            }
            
            # Add method-specific metrics
            if method == 'wrapper' and 'best_cv_score' in selection_results:
                result['cv_score'] = selection_results['best_cv_score']
            
            results.append(result)
            logger.info(f"Completed {method}: {result['n_features_selected']} features")
            
        except Exception as e:
            logger.error(f"Failed {method} selection: {e}")
    
    return pd.DataFrame(results)


def feature_stability_analysis(df: pd.DataFrame, 
                              method: str = 'univariate',
                              n_bootstrap: int = 10,
                              k_features: int = 8) -> Dict[str, Any]:
    """Analyze stability of feature selection across bootstrap samples.
    
    Args:
        df: Training dataframe
        method: Feature selection method
        n_bootstrap: Number of bootstrap iterations
        k_features: Number of features to select
        
    Returns:
        Dictionary with stability analysis results
    """
    logger.info(f"Analyzing feature selection stability with {n_bootstrap} bootstrap samples")
    
    feature_selection_counts = {}
    all_features = []
    
    # Get all possible features
    selector = CreatorFeatureSelector(method=method, k_features=k_features)
    X, y, feature_names = selector._prepare_features(df)
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        n_samples = len(df)
        bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_df = df.iloc[bootstrap_idx].reset_index(drop=True)
        
        try:
            # Apply selection
            bootstrap_selector = CreatorFeatureSelector(method=method, k_features=k_features)
            results = bootstrap_selector.fit(bootstrap_df)
            selected_features = results['selected_features']
            
            # Count selections
            for feature in selected_features:
                feature_selection_counts[feature] = feature_selection_counts.get(feature, 0) + 1
            
            all_features.extend(selected_features)
            
        except Exception as e:
            logger.warning(f"Bootstrap iteration {i} failed: {e}")
    
    # Calculate stability metrics
    stability_results = []
    for feature in feature_names:
        count = feature_selection_counts.get(feature, 0)
        stability_results.append({
            'feature': feature,
            'selection_frequency': count / n_bootstrap,
            'selection_count': count
        })
    
    stability_df = pd.DataFrame(stability_results).sort_values('selection_frequency', ascending=False)
    
    # Overall stability score (Jaccard similarity across bootstrap samples)
    stability_score = len(set(all_features)) / len(all_features) if all_features else 0
    
    return {
        'method': method,
        'n_bootstrap': n_bootstrap,
        'stability_score': stability_score,
        'feature_stability': stability_df,
        'most_stable_features': stability_df.head(k_features)['feature'].tolist()
    }


# =============================================================================
# EXPORT LIST
# =============================================================================

__all__ = [
    'CreatorFeatureSelector',
    'compare_selection_methods', 
    'feature_stability_analysis'
]
