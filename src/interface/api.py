"""
Creator Growth Navigator - API Interface Module

FastAPI-based REST API for serving creator growth prediction models with
comprehensive validation, error handling, and monitoring capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import joblib
import json
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, root_validator
import uvicorn

# Import from project modules  
from src.config.config import MODEL_CONFIG, STREAMLIT_CONFIG, get_data_path
from src.utils.logger import get_model_logger
from src.models.regression import CreatorGrowthModel
from src.evaluation.kpi import CreatorKPIEvaluator

logger = get_model_logger()

# Global model instance
model_instance: Optional[CreatorGrowthModel] = None
model_metadata: Dict[str, Any] = {}


# =============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE
# =============================================================================

class PredictionRequest(BaseModel):
    """Request model for growth predictions."""
    
    # Primary features (aligned with your data structure)
    weekly_posting_frequency: float = Field(..., ge=0, le=50, description="Total posts + reels + stories per week")
    
    # Content mix features (optional, with defaults)
    share_posts: Optional[float] = Field(0.33, ge=0, le=1, description="Share of total content that are posts")
    share_reels: Optional[float] = Field(0.33, ge=0, le=1, description="Share of total content that are reels")
    share_stories: Optional[float] = Field(0.34, ge=0, le=1, description="Share of total content that are stories")
    
    # Engagement features
    engagement_rate: Optional[float] = Field(0.05, ge=0, le=1, description="Average engagement rate")
    avg_hashtag_count: Optional[float] = Field(10.0, ge=0, le=30, description="Average hashtags per post")
    
    # Consistency features
    post_consistency_variance_7d: Optional[float] = Field(0.0, ge=0, description="7-day posting variance")
    posted_in_optimal_window: Optional[int] = Field(0, ge=0, le=1, description="Binary flag for optimal timing")
    
    # ROI features
    roi_follows_per_hour: Optional[float] = Field(1.0, ge=0, description="Follows gained per hour of work")
    minutes_spent: Optional[int] = Field(120, ge=0, le=2000, description="Minutes spent creating content")
    
    # Temporal features
    month: Optional[int] = Field(1, ge=1, le=12, description="Month of the year")
    quarter: Optional[int] = Field(1, ge=1, le=4, description="Quarter of the year")
    
    # Quality features
    saturation_flag: Optional[int] = Field(0, ge=0, le=1, description="Posting saturation indicator")
    
    @validator('share_posts', 'share_reels', 'share_stories', pre=True, always=True)
    def validate_content_shares(cls, v, values):
        """Ensure content shares sum to approximately 1."""
        return v
    
    @root_validator
    def validate_content_mix_sum(cls, values):
        """Validate that content shares sum to 1."""
        shares = [
            values.get('share_posts', 0),
            values.get('share_reels', 0), 
            values.get('share_stories', 0)
        ]
        total = sum(shares)
        if abs(total - 1.0) > 0.01:  # Allow small tolerance
            # Normalize to sum to 1
            normalized_shares = [s / total for s in shares]
            values['share_posts'] = normalized_shares[0]
            values['share_reels'] = normalized_shares[12]
            values['share_stories'] = normalized_shares[13]
        
        return values


class PredictionResponse(BaseModel):
    """Response model for growth predictions."""
    
    predicted_growth: float = Field(..., description="Predicted weekly follower growth")
    confidence_interval_lower: Optional[float] = Field(None, description="Lower bound of 95% confidence interval")
    confidence_interval_upper: Optional[float] = Field(None, description="Upper bound of 95% confidence interval")
    
    # Prediction metadata
    prediction_timestamp: str = Field(..., description="When prediction was made")
    model_version: Optional[str] = Field(None, description="Model version used")
    confidence_level: float = Field(0.95, description="Confidence level for intervals")
    
    # Business insights
    growth_category: Optional[str] = Field(None, description="Growth category (low/medium/high)")
    recommendation: Optional[str] = Field(None, description="Strategy recommendation")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    predictions: List[PredictionRequest] = Field(..., min_items=1, max_items=100)
    include_metadata: bool = Field(True, description="Include prediction metadata")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[PredictionResponse]
    batch_summary: Dict[str, Any]
    processing_time_seconds: float


class ModelStatusResponse(BaseModel):
    """Response model for model status."""
    
    model_loaded: bool
    model_version: Optional[str] = None
    model_type: Optional[str] = None
    training_date: Optional[str] = None
    feature_count: Optional[int] = None
    performance_metrics: Optional[Dict[str, float]] = None


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Check timestamp")
    model_ready: bool = Field(..., description="Whether model is ready for predictions")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime")


# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="Creator Growth Navigator API",
    description="REST API for predicting Instagram follower growth based on posting frequency and content strategy",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store startup time
startup_time = datetime.now()


# =============================================================================
# STARTUP AND SHUTDOWN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model and initialize services on startup."""
    global model_instance, model_metadata
    
    logger.info("Starting Creator Growth Navigator API")
    
    try:
        # Attempt to load saved model
        model_path = Path("models/creator_growth_model.pkl")
        
        if model_path.exists():
            model_instance = CreatorGrowthModel.load_model(str(model_path))
            
            # Extract metadata
            model_metadata = {
                "model_loaded": True,
                "model_type": model_instance.model_type,
                "feature_count": len(model_instance.feature_names) if model_instance.feature_names else 0,
                "training_stats": model_instance.training_stats,
                "load_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Model loaded successfully: {model_instance.model_type}")
        else:
            logger.warning(f"Model file not found at {model_path}")
            model_metadata = {"model_loaded": False, "error": "Model file not found"}
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_metadata = {"model_loaded": False, "error": str(e)}


@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Creator Growth Navigator API")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Creator Growth Navigator API",
        "version": "1.0.0",
        "description": "Predict Instagram follower growth based on posting strategy",
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    current_time = datetime.now()
    uptime = (current_time - startup_time).total_seconds()
    
    return HealthCheckResponse(
        status="healthy" if model_instance else "degraded",
        timestamp=current_time.isoformat(),
        model_ready=model_instance is not None,
        uptime_seconds=uptime
    )


@app.get("/model/status", response_model=ModelStatusResponse)
async def model_status():
    """Get detailed model status information."""
    if not model_instance:
        return ModelStatusResponse(
            model_loaded=False,
            model_version=None,
            model_type=None
        )
    
    # Extract performance metrics if available
    performance_metrics = None
    if hasattr(model_instance, 'training_stats'):
        train_metrics = model_instance.training_stats.get('train_metrics', {})
        if train_metrics:
            performance_metrics = {
                'r2': train_metrics.get('r2'),
                'rmse': train_metrics.get('rmse'),
                'mae': train_metrics.get('mae'),
                'directional_accuracy': train_metrics.get('directional_accuracy')
            }
    
    return ModelStatusResponse(
        model_loaded=True,
        model_version=model_metadata.get('version', '1.0.0'),
        model_type=model_instance.model_type,
        training_date=model_metadata.get('load_timestamp'),
        feature_count=len(model_instance.feature_names) if model_instance.feature_names else 0,
        performance_metrics=performance_metrics
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_growth(request: PredictionRequest):
    """Make a single growth prediction."""
    if not model_instance:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check service status."
        )
    
    try:
        # Prepare input dataframe
        input_data = {
            'weekly_posting_frequency': request.weekly_posting_frequency,
            'share_posts': request.share_posts,
            'share_reels': request.share_reels,
            'share_stories': request.share_stories,
            'engagement_rate': request.engagement_rate,
            'avg_hashtag_count': request.avg_hashtag_count,
            'post_consistency_variance_7d': request.post_consistency_variance_7d,
            'posted_in_optimal_window': request.posted_in_optimal_window,
            'roi_follows_per_hour': request.roi_follows_per_hour,
            'minutes_spent': request.minutes_spent,
            'month': request.month,
            'quarter': request.quarter,
            'saturation_flag': request.saturation_flag
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Make prediction with confidence intervals
        predictions, lower_bounds, upper_bounds = model_instance.predict_with_confidence(input_df)
        
        predicted_growth = float(predictions[0])
        confidence_lower = float(lower_bounds) if lower_bounds is not None else None
        confidence_upper = float(upper_bounds) if upper_bounds is not None else None
        
        # Determine growth category
        growth_category = _categorize_growth(predicted_growth)
        
        # Generate recommendation
        recommendation = _generate_recommendation(request, predicted_growth)
        
        return PredictionResponse(
            predicted_growth=predicted_growth,
            confidence_interval_lower=confidence_lower,
            confidence_interval_upper=confidence_upper,
            prediction_timestamp=datetime.now().isoformat(),
            model_version=model_metadata.get('version', '1.0.0'),
            growth_category=growth_category,
            recommendation=recommendation
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_growth_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    if not model_instance:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check service status."
        )
    
    start_time = datetime.now()
    
    try:
        # Prepare batch input
        batch_data = []
        for pred_request in request.predictions:
            input_data = {
                'weekly_posting_frequency': pred_request.weekly_posting_frequency,
                'share_posts': pred_request.share_posts,
                'share_reels': pred_request.share_reels,
                'share_stories': pred_request.share_stories,
                'engagement_rate': pred_request.engagement_rate,
                'avg_hashtag_count': pred_request.avg_hashtag_count,
                'post_consistency_variance_7d': pred_request.post_consistency_variance_7d,
                'posted_in_optimal_window': pred_request.posted_in_optimal_window,
                'roi_follows_per_hour': pred_request.roi_follows_per_hour,
                'minutes_spent': pred_request.minutes_spent,
                'month': pred_request.month,
                'quarter': pred_request.quarter,
                'saturation_flag': pred_request.saturation_flag
            }
            batch_data.append(input_data)
        
        batch_df = pd.DataFrame(batch_data)
        
        # Make batch predictions
        predictions, lower_bounds, upper_bounds = model_instance.predict_with_confidence(batch_df)
        
        # Process results
        response_predictions = []
        for i, (pred_request, prediction) in enumerate(zip(request.predictions, predictions)):
            confidence_lower = float(lower_bounds[i]) if lower_bounds is not None else None
            confidence_upper = float(upper_bounds[i]) if upper_bounds is not None else None
            
            pred_response = PredictionResponse(
                predicted_growth=float(prediction),
                confidence_interval_lower=confidence_lower,
                confidence_interval_upper=confidence_upper,
                prediction_timestamp=datetime.now().isoformat(),
                model_version=model_metadata.get('version', '1.0.0'),
                growth_category=_categorize_growth(float(prediction)) if request.include_metadata else None,
                recommendation=_generate_recommendation(pred_request, float(prediction)) if request.include_metadata else None
            )
            response_predictions.append(pred_response)
        
        # Calculate batch summary
        processing_time = (datetime.now() - start_time).total_seconds()
        
        batch_summary = {
            'total_predictions': len(predictions),
            'average_predicted_growth': float(np.mean(predictions)),
            'min_predicted_growth': float(np.min(predictions)),
            'max_predicted_growth': float(np.max(predictions)),
            'processing_time_per_prediction_ms': (processing_time * 1000) / len(predictions)
        }
        
        return BatchPredictionResponse(
            predictions=response_predictions,
            batch_summary=batch_summary,
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/features", response_model=Dict[str, List[str]])
async def get_model_features():
    """Get list of model features."""
    if not model_instance or not model_instance.feature_names:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded or feature information unavailable"
        )
    
    return {
        "features": model_instance.feature_names,
        "feature_count": len(model_instance.feature_names)
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _categorize_growth(predicted_growth: float) -> str:
    """Categorize predicted growth into business-friendly labels."""
    if predicted_growth >= 1000:
        return "explosive"
    elif predicted_growth >= 500:
        return "high" 
    elif predicted_growth >= 100:
        return "moderate"
    elif predicted_growth >= 0:
        return "low"
    else:
        return "decline"


def _generate_recommendation(request: PredictionRequest, predicted_growth: float) -> str:
    """Generate strategic recommendation based on input and prediction."""
    recommendations = []
    
    # Posting frequency recommendations
    if request.weekly_posting_frequency < 5:
        recommendations.append("Consider increasing posting frequency")
    elif request.weekly_posting_frequency > 15:
        recommendations.append("Monitor for posting saturation")
    
    # Content mix recommendations
    if request.share_reels < 0.4:
        recommendations.append("Increase reels content for better reach")
    
    # Engagement recommendations
    if request.engagement_rate < 0.03:
        recommendations.append("Focus on improving engagement quality")
    
    # Growth-based recommendations
    if predicted_growth < 0:
        recommendations.append("Review content strategy immediately")
    elif predicted_growth < 100:
        recommendations.append("Optimize posting timing and consistency")
    
    return "; ".join(recommendations) if recommendations else "Continue current strategy"


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with logging."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc} - {request.url}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )


# =============================================================================
# EXPORT LIST
# =============================================================================

__all__ = [
    'app',
    'PredictionRequest',
    'PredictionResponse', 
    'BatchPredictionRequest',
    'BatchPredictionResponse',
    'ModelStatusResponse',
    'HealthCheckResponse'
]
