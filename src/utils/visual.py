"""
Creator Growth Navigator - Visualization Utilities

Centralized visualization functions for charts, plots, and visual analytics
used across notebooks, Streamlit app, and reporting components.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

# Import configuration and utilities
from ..config.config import COLORS, UI_CONFIG
from .logger import get_logger
from .helpers import format_large_numbers, create_color_palette

logger = get_logger(__name__)

# Set default style
plt.style.use('seaborn-v0_8')
warnings.filterwarnings('ignore')

# =============================================================================
# PLOTLY CONFIGURATION
# =============================================================================

# Default Plotly layout
DEFAULT_LAYOUT = {
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'font': {'family': "Arial, sans-serif", 'size': 12, 'color': COLORS['text']},
    'title': {'x': 0.5, 'xanchor': 'center'},
    'margin': {'l': 50, 'r': 50, 't': 60, 'b': 50},
    'showlegend': True,
    'legend': {'orientation': "h", 'yanchor': "bottom", 'y': 1.02, 'xanchor': "right", 'x': 1}
}

# Default color sequence
DEFAULT_COLORS = [COLORS['primary'], COLORS['secondary'], COLORS['success'], 
                 COLORS['warning'], COLORS['info']]

# =============================================================================
# TIME SERIES VISUALIZATIONS
# =============================================================================

def plot_follower_growth_history(df: pd.DataFrame,
                                date_col: str = 'date',
                                followers_col: str = 'followers',
                                predictions_col: Optional[str] = None,
                                confidence_cols: Optional[Tuple[str, str]] = None,
                                title: str = "Follower Growth History") -> go.Figure:
    """Create an interactive follower growth chart with actual and predicted values.
    
    Args:
        df: DataFrame with time series data
        date_col: Date column name
        followers_col: Actual followers column name
        predictions_col: Predicted followers column name (optional)
        confidence_cols: Tuple of (lower_bound, upper_bound) column names
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Actual followers line
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[followers_col],
        mode='lines+markers',
        name='Actual Followers',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=4),
        hovertemplate='<b>%{y:,.0f}</b> followers<br>%{x}<extra></extra>'
    ))
    
    # Predicted followers line
    if predictions_col and predictions_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[predictions_col],
            mode='lines',
            name='Predicted Followers',
            line=dict(color=COLORS['secondary'], width=2, dash='dash'),
            hovertemplate='<b>%{y:,.0f}</b> predicted<br>%{x}<extra></extra>'
        ))
    
    # Confidence interval
    if confidence_cols and all(col in df.columns for col in confidence_cols):
        lower_col, upper_col = confidence_cols
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[upper_col],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[lower_col],
            mode='lines',
            line=dict(width=0),
            name='95% Confidence Interval',
            fill='tonexty',
            fillcolor=f'rgba{(*hex_to_rgb(COLORS["primary"]), 0.2)}',
            hovertemplate='CI: %{y:,.0f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        **DEFAULT_LAYOUT,
        title=title,
        xaxis_title="Date",
        yaxis_title="Followers",
        height=UI_CONFIG['chart_height'],
        hovermode='x unified'
    )
    
    # Format y-axis labels
    fig.update_yaxis(tickformat=',')
    
    logger.debug(f"Created follower growth chart with {len(df)} data points")
    return fig


def plot_posting_frequency_trend(df: pd.DataFrame,
                                date_col: str = 'date',
                                posts_col: str = 'posts',
                                reels_col: str = 'reels',
                                stories_col: str = 'stories',
                                title: str = "Posting Frequency Trends") -> go.Figure:
    """Create a stacked area chart showing posting frequency by content type.
    
    Args:
        df: DataFrame with posting data
        date_col: Date column name
        posts_col: Posts count column
        reels_col: Reels count column
        stories_col: Stories count column
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add traces for each content type
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[posts_col],
        mode='lines',
        name='Posts',
        stackgroup='one',
        line=dict(color=COLORS['primary'], width=0),
        fillcolor=COLORS['primary'],
        hovertemplate='Posts: <b>%{y}</b><br>%{x}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[reels_col],
        mode='lines',
        name='Reels',
        stackgroup='one',
        line=dict(color=COLORS['secondary'], width=0),
        fillcolor=COLORS['secondary'],
        hovertemplate='Reels: <b>%{y}</b><br>%{x}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[stories_col],
        mode='lines',
        name='Stories',
        stackgroup='one',
        line=dict(color=COLORS['success'], width=0),
        fillcolor=COLORS['success'],
        hovertemplate='Stories: <b>%{y}</b><br>%{x}<extra></extra>'
    ))
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        title=title,
        xaxis_title="Date",
        yaxis_title="Number of Posts",
        height=UI_CONFIG['chart_height'],
        hovermode='x unified'
    )
    
    return fig


def plot_weekly_aggregation_comparison(df: pd.DataFrame,
                                     date_col: str = 'date',
                                     weekly_freq_col: str = 'weekly_posting_frequency',
                                     weekly_growth_col: str = 'weekly_growth',
                                     title: str = "Weekly Posting vs Growth") -> go.Figure:
    """Create dual-axis chart comparing posting frequency and growth.
    
    Args:
        df: DataFrame with weekly aggregated data
        date_col: Date column name
        weekly_freq_col: Weekly posting frequency column
        weekly_growth_col: Weekly growth column
        title: Chart title
        
    Returns:
        Plotly figure object with dual y-axes
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Posting frequency (left axis)
    fig.add_trace(
        go.Bar(
            x=df[date_col],
            y=df[weekly_freq_col],
            name="Weekly Posts",
            marker_color=COLORS['primary'],
            opacity=0.7,
            hovertemplate='Posts: <b>%{y}</b><br>%{x}<extra></extra>'
        ),
        secondary_y=False,
    )
    
    # Weekly growth (right axis)
    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[weekly_growth_col],
            mode='lines+markers',
            name="Weekly Growth",
            line=dict(color=COLORS['secondary'], width=3),
            marker=dict(size=6),
            hovertemplate='Growth: <b>%{y:,.0f}</b><br>%{x}<extra></extra>'
        ),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_layout(
        **DEFAULT_LAYOUT,
        title=title,
        height=UI_CONFIG['chart_height'],
        hovermode='x unified'
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Weekly Posts", secondary_y=False)
    fig.update_yaxes(title_text="Follower Growth", secondary_y=True)
    
    return fig


# =============================================================================
# ENGAGEMENT ANALYTICS
# =============================================================================

def plot_engagement_breakdown(df: pd.DataFrame,
                             engagement_cols: List[str],
                             labels: Optional[List[str]] = None,
                             title: str = "Engagement Breakdown") -> go.Figure:
    """Create a pie chart showing engagement distribution.
    
    Args:
        df: DataFrame with engagement data
        engagement_cols: List of engagement column names
        labels: Custom labels for engagement types
        title: Chart title
        
    Returns:
        Plotly pie chart figure
    """
    if labels is None:
        labels = [col.replace('_', ' ').title() for col in engagement_cols]
    
    # Calculate totals for each engagement type
    values = [df[col].sum() for col in engagement_cols]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker_colors=DEFAULT_COLORS[:len(values)],
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>%{value:,.0f} total<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        title=title,
        height=UI_CONFIG['chart_height'],
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
    )
    
    return fig


def plot_engagement_rate_trend(df: pd.DataFrame,
                              date_col: str = 'date',
                              engagement_rate_col: str = 'engagement_rate',
                              rolling_window: int = 7,
                              title: str = "Engagement Rate Trend") -> go.Figure:
    """Create engagement rate trend with rolling average.
    
    Args:
        df: DataFrame with engagement rate data
        date_col: Date column name
        engagement_rate_col: Engagement rate column
        rolling_window: Window size for rolling average
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Calculate rolling average
    df_plot = df.copy()
    df_plot[f'engagement_rate_rolling_{rolling_window}d'] = df_plot[engagement_rate_col].rolling(rolling_window).mean()
    
    fig = go.Figure()
    
    # Daily engagement rate (light line)
    fig.add_trace(go.Scatter(
        x=df_plot[date_col],
        y=df_plot[engagement_rate_col] * 100,  # Convert to percentage
        mode='lines',
        name='Daily Rate',
        line=dict(color=COLORS['primary'], width=1, dash='dot'),
        opacity=0.5,
        hovertemplate='Daily: <b>%{y:.2f}%</b><br>%{x}<extra></extra>'
    ))
    
    # Rolling average (bold line)
    fig.add_trace(go.Scatter(
        x=df_plot[date_col],
        y=df_plot[f'engagement_rate_rolling_{rolling_window}d'] * 100,
        mode='lines',
        name=f'{rolling_window}-Day Average',
        line=dict(color=COLORS['secondary'], width=3),
        hovertemplate=f'{rolling_window}d avg: <b>%{{y:.2f}}%</b><br>%{{x}}<extra></extra>'
    ))
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        title=title,
        xaxis_title="Date",
        yaxis_title="Engagement Rate (%)",
        height=UI_CONFIG['chart_height'],
        hovermode='x unified'
    )
    
    # Add percentage formatting
    fig.update_yaxis(ticksuffix='%')
    
    return fig


# =============================================================================
# MODEL DIAGNOSTICS
# =============================================================================

def plot_residuals_analysis(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           title: str = "Residual Analysis") -> go.Figure:
    """Create residual analysis plots for model diagnostics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        title: Chart title
        
    Returns:
        Plotly figure with subplots
    """
    residuals = y_true - y_pred
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Residuals vs Fitted', 'Q-Q Plot', 'Residuals Histogram', 'Residuals vs Time'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    # 1. Residuals vs Fitted Values
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color=COLORS['primary'], opacity=0.6),
            hovertemplate='Fitted: %{x:,.0f}<br>Residual: %{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # 2. Q-Q Plot (Normal probability plot)
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode='markers',
            name='Q-Q Points',
            marker=dict(color=COLORS['secondary'], opacity=0.6),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Q-Q reference line
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=slope * osm + intercept,
            mode='lines',
            name='Normal Line',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Residuals Histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            name='Residuals Distribution',
            marker_color=COLORS['success'],
            opacity=0.7,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Residuals vs Time (assuming sequential order)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(residuals))),
            y=residuals,
            mode='lines+markers',
            name='Residuals Over Time',
            line=dict(color=COLORS['warning']),
            marker=dict(size=4),
            showlegend=False,
            hovertemplate='Index: %{x}<br>Residual: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Add horizontal line at y=0 for time plot
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        showlegend=False,
        **{k: v for k, v in DEFAULT_LAYOUT.items() if k != 'height'}
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
    fig.update_xaxes(title_text="Residuals", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Observation Index", row=2, col=2)
    fig.update_yaxes(title_text="Residuals", row=2, col=2)
    
    return fig


def plot_feature_importance(feature_names: List[str],
                           importance_values: np.ndarray,
                           title: str = "Feature Importance") -> go.Figure:
    """Create horizontal bar chart of feature importance.
    
    Args:
        feature_names: List of feature names
        importance_values: Array of importance values (coefficients)
        title: Chart title
        
    Returns:
        Plotly horizontal bar chart
    """
    # Sort by absolute importance
    abs_importance = np.abs(importance_values)
    sorted_indices = np.argsort(abs_importance)
    
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_values = importance_values[sorted_indices]
    
    # Color bars based on positive/negative
    colors = [COLORS['success'] if val >= 0 else COLORS['warning'] for val in sorted_values]
    
    fig = go.Figure([go.Bar(
        x=sorted_values,
        y=sorted_features,
        orientation='h',
        marker_color=colors,
        hovertemplate='<b>%{y}</b><br>Coefficient: %{x:.4f}<extra></extra>'
    )])
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        title=title,
        xaxis_title="Coefficient Value",
        yaxis_title="Features",
        height=max(400, len(feature_names) * 25),
        showlegend=False
    )
    
    # Add vertical line at x=0
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
    
    return fig


def plot_prediction_intervals(dates: pd.Series,
                             predictions: np.ndarray,
                             lower_bounds: np.ndarray,
                             upper_bounds: np.ndarray,
                             actual: Optional[np.ndarray] = None,
                             title: str = "Prediction Intervals") -> go.Figure:
    """Create chart showing predictions with confidence intervals.
    
    Args:
        dates: Date series
        predictions: Point predictions
        lower_bounds: Lower confidence bounds
        upper_bounds: Upper confidence bounds
        actual: Actual values (optional)
        title: Chart title
        
    Returns:
        Plotly figure with confidence intervals
    """
    fig = go.Figure()
    
    # Confidence interval (fill between)
    fig.add_trace(go.Scatter(
        x=dates,
        y=upper_bounds,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=lower_bounds,
        mode='lines',
        line=dict(width=0),
        name='95% Confidence Interval',
        fill='tonexty',
        fillcolor=f'rgba{(*hex_to_rgb(COLORS["primary"]), UI_CONFIG["confidence_interval_opacity"])}',
        hovertemplate='CI: %{y:,.0f}<extra></extra>'
    ))
    
    # Point predictions
    fig.add_trace(go.Scatter(
        x=dates,
        y=predictions,
        mode='lines+markers',
        name='Predictions',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=4),
        hovertemplate='Prediction: <b>%{y:,.0f}</b><br>%{x}<extra></extra>'
    ))
    
    # Actual values if provided
    if actual is not None:
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            mode='markers',
            name='Actual',
            marker=dict(color=COLORS['secondary'], size=6),
            hovertemplate='Actual: <b>%{y:,.0f}</b><br>%{x}<extra></extra>'
        ))
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        title=title,
        xaxis_title="Date",
        yaxis_title="Followers",
        height=UI_CONFIG['chart_height'],
        hovermode='x unified'
    )
    
    return fig


# =============================================================================
# BUSINESS INTELLIGENCE CHARTS
# =============================================================================

def plot_roi_analysis(df: pd.DataFrame,
                     roi_col: str = 'roi_follows_per_hour',
                     posting_freq_col: str = 'weekly_posting_frequency',
                     title: str = "ROI Analysis: Follows per Hour vs Posting Frequency") -> go.Figure:
    """Create scatter plot of ROI vs posting frequency.
    
    Args:
        df: DataFrame with ROI data
        roi_col: ROI column name
        posting_freq_col: Posting frequency column
        title: Chart title
        
    Returns:
        Plotly scatter plot with trend line
    """
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=df[posting_freq_col],
        y=df[roi_col],
        mode='markers',
        name='Weekly Performance',
        marker=dict(
            color=COLORS['primary'],
            size=8,
            opacity=0.7
        ),
        hovertemplate='Posts: <b>%{x}</b><br>ROI: <b>%{y:.1f}</b> follows/hour<extra></extra>'
    ))
    
    # Add trend line
    try:
        z = np.polyfit(df[posting_freq_col].dropna(), df[roi_col].dropna(), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df[posting_freq_col].min(), df[posting_freq_col].max(), 100)
        y_trend = p(x_trend)
        
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            name='Trend Line',
            line=dict(color=COLORS['secondary'], width=2, dash='dash'),
            hoverinfo='skip'
        ))
    except:
        logger.warning("Could not generate trend line for ROI analysis")
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        title=title,
        xaxis_title="Weekly Posting Frequency",
        yaxis_title="ROI (Follows per Hour)",
        height=UI_CONFIG['chart_height']
    )
    
    return fig


def plot_content_mix_performance(df: pd.DataFrame,
                                post_share_col: str = 'share_posts',
                                reel_share_col: str = 'share_reels',
                                story_share_col: str = 'share_stories',
                                growth_col: str = 'weekly_growth',
                                title: str = "Content Mix vs Performance") -> go.Figure:
    """Create ternary plot showing content mix performance.
    
    Args:
        df: DataFrame with content mix data
        post_share_col: Posts share column
        reel_share_col: Reels share column
        story_share_col: Stories share column
        growth_col: Growth metric column
        title: Chart title
        
    Returns:
        Plotly ternary plot
    """
    fig = go.Figure(go.Scatterternary({
        'mode': 'markers',
        'a': df[post_share_col] * 100,  # Convert to percentages
        'b': df[reel_share_col] * 100,
        'c': df[story_share_col] * 100,
        'marker': {
            'color': df[growth_col],
            'colorscale': 'Viridis',
            'size': 8,
            'colorbar': {
                'title': 'Weekly Growth'
            }
        },
        'text': df.index,
        'hovertemplate': 'Posts: %{a:.1f}%<br>Reels: %{b:.1f}%<br>Stories: %{c:.1f}%<br>Growth: %{marker.color:,.0f}<extra></extra>'
    }))
    
    fig.update_layout(
        title=title,
        height=600,
        ternary={
            'sum': 100,
            'aaxis': {'title': 'Posts %'},
            'baxis': {'title': 'Reels %'},
            'caxis': {'title': 'Stories %'}
        },
        **{k: v for k, v in DEFAULT_LAYOUT.items() if k not in ['height']}
    )
    
    return fig


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple.
    
    Args:
        hex_color: Hex color string (e.g., '#1f77b4')
        
    Returns:
        RGB tuple
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def create_metric_card_data(value: Union[int, float],
                           title: str,
                           comparison_value: Optional[Union[int, float]] = None,
                           format_as_number: bool = True,
                           format_as_percentage: bool = False) -> Dict[str, Any]:
    """Prepare data for metric cards in Streamlit.
    
    Args:
        value: Main metric value
        title: Metric title
        comparison_value: Previous period value for comparison
        format_as_number: Whether to format as large number
        format_as_percentage: Whether to format as percentage
        
    Returns:
        Dictionary with formatted metric data
    """
    formatted_value = value
    
    if format_as_percentage:
        formatted_value = f"{value:.1f}%"
    elif format_as_number:
        formatted_value = format_large_numbers(value)
    
    delta = None
    delta_color = "normal"
    
    if comparison_value is not None:
        delta = value - comparison_value
        delta_color = "normal" if delta == 0 else ("inverse" if delta < 0 else "normal")
        
        if format_as_percentage:
            delta = f"{delta:.1f}pp"
        elif format_as_number:
            delta = format_large_numbers(delta)
    
    return {
        "title": title,
        "value": formatted_value,
        "delta": delta,
        "delta_color": delta_color
    }


def save_chart_as_image(fig: go.Figure,
                       filepath: str,
                       width: int = 1200,
                       height: int = 600,
                       format: str = 'png') -> bool:
    """Save Plotly figure as image file.
    
    Args:
        fig: Plotly figure
        filepath: Output file path
        width: Image width in pixels
        height: Image height in pixels
        format: Image format ('png', 'jpg', 'pdf', 'svg')
        
    Returns:
        True if successful, False otherwise
    """
    try:
        fig.write_image(filepath, width=width, height=height, format=format)
        logger.info(f"Chart saved as image: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save chart as image: {e}")
        return False


# =============================================================================
# EXPORT LIST
# =============================================================================

__all__ = [
    # Time series visualizations
    'plot_follower_growth_history',
    'plot_posting_frequency_trend', 
    'plot_weekly_aggregation_comparison',
    
    # Engagement analytics
    'plot_engagement_breakdown',
    'plot_engagement_rate_trend',
    
    # Model diagnostics
    'plot_residuals_analysis',
    'plot_feature_importance',
    'plot_prediction_intervals',
    
    # Business intelligence
    'plot_roi_analysis',
    'plot_content_mix_performance',
    
    # Utilities
    'hex_to_rgb',
    'create_metric_card_data',
    'save_chart_as_image',
    
    # Constants
    'DEFAULT_LAYOUT',
    'DEFAULT_COLORS'
]
