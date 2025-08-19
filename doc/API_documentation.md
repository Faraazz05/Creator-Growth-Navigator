# Creator Growth Navigator - API Documentation

## Overview
The Creator Growth Navigator provides a programmatic interface for batch predictions, model training, and data processing workflows.

## Installation & Setup

from src.models.regression import LinearGrowthModel
from src.data.loader import DataLoader
from src.evaluation.kpi import ModelEvaluator


## Core Classes

### DataLoader
Load and preprocess creator data for modeling.

loader = DataLoader()

#### **Load raw data**
raw_data = loader.load_raw_data('data/raw/creator_daily_metrics.csv')

#### **Load processed features**
features = loader.load_processed_data('data/processed/model_ready_features.csv')

#### **Validate data quality**
validation_report = loader.validate_data(raw_data)


**Methods:**
- `load_raw_data(filepath)`: Load raw creator metrics
- `load_processed_data(filepath)`: Load feature-engineered data
- `validate_data(df)`: Data quality validation
- `save_processed_data(df, filepath)`: Save processed datasets

### LinearGrowthModel
Core linear regression model for follower growth prediction.

model = LinearGrowthModel()

#### **Train model**
model.fit(X_train, y_train)

#### **Make predictions**
predictions = model.predict(X_test)

#### **Get confidence intervals**
predictions, intervals = model.predict_with_confidence(X_test, confidence=0.95)

#### **Feature importance**
importance = model.get_feature_importance()


**Methods:**
- `fit(X, y)`: Train the linear regression model
- `predict(X)`: Generate point predictions
- `predict_with_confidence(X, confidence)`: Predictions with uncertainty
- `get_feature_importance()`: Coefficient interpretation
- `get_model_summary()`: Detailed model statistics

### ModelEvaluator
Comprehensive model performance assessment.

evaluator = ModelEvaluator()

Calculate metrics
metrics = evaluator.calculate_metrics(y_true, y_pred)

Time series validation
cv_scores = evaluator.time_series_cross_validation(model, X, y)

Residual analysis
residual_analysis = evaluator.analyze_residuals(y_true, y_pred)



**Methods:**
- `calculate_metrics(y_true, y_pred)`: RMSE, MAE, R², MAPE
- `time_series_cross_validation(model, X, y)`: Temporal validation
- `analyze_residuals(y_true, y_pred)`: Diagnostic analysis
- `detect_drift(current_metrics, baseline_metrics)`: Model drift detection

## Data Structures

### Input Data Format
Required columns for daily metrics
required_columns = [
'date', # YYYY-MM-DD format
'followers', # Current follower count
'posts', # Number of posts that day
'stories', # Number of stories that day
'reels', # Number of reels that day
'engagement_rate', # Daily engagement rate
'reach' # Content reach
]



### Prediction Output
Prediction response structure
{
'prediction': 1250, # Predicted follower growth
'confidence_interval': {
'lower': 980, # Lower bound (95% CI)
'upper': 1520 # Upper bound (95% CI)
},
'model_version': '1.0.0',
'prediction_date': '2025-08-19',
'features_used': [
'weekly_posting_frequency',
'content_mix_ratio',
'posting_consistency'
]
}



## Batch Processing

### Bulk Predictions
from src.interface.api import BatchPredictor

predictor = BatchPredictor()

Process multiple scenarios
scenarios = [
{'weekly_posts': 5, 'content_mix': 'balanced'},
{'weekly_posts': 7, 'content_mix': 'reel_heavy'},
{'weekly_posts': 3, 'content_mix': 'post_focused'}
]

results = predictor.predict_scenarios(scenarios)



### Automated Retraining
from src.models.regression import ModelTrainer

trainer = ModelTrainer()

Automated training pipeline
training_results = trainer.retrain_model(
new_data_path='data/raw/latest_metrics.csv',
validation_split=0.2,
save_model=True
)



## Configuration

### Model Parameters
Default configuration
config = {
'model_type': 'linear_regression',
'feature_selection': 'manual',
'validation_method': 'time_series_split',
'confidence_level': 0.95,
'outlier_detection': True,
'regularization': None
}


### Data Processing Options
Processing configuration
processing_config = {
'aggregation_window': '7D', # Weekly aggregation
'outlier_threshold': 3.0, # Standard deviations
'missing_value_strategy': 'interpolate',
'feature_scaling': 'standard'
}



## Error Handling

### Common Exceptions
Custom exceptions
from src.utils.exceptions import (
DataValidationError,
ModelNotTrainedError,
InsufficientDataError
)

try:
predictions = model.predict(X_new)
except ModelNotTrainedError:
print("Model must be trained before making predictions")
except DataValidationError as e:
print(f"Data validation failed: {e}")



## Rate Limits & Performance

### Batch Size Recommendations
- **Single prediction**: < 1ms
- **Batch predictions (100 scenarios)**: < 100ms  
- **Model training**: 1-5 seconds (depending on data size)
- **Cross-validation**: 10-30 seconds

### Memory Usage
- **Model object**: ~1MB
- **Feature data (1 year)**: ~500KB
- **Prediction cache**: Configurable (default 1000 predictions)

## Integration Examples

### Streamlit Integration
Custom Streamlit component
@st.cache_data
def load_model():
return LinearGrowthModel.load('models/latest_model.pkl')

model = load_model()
prediction = model.predict(user_input)



### API Endpoint Integration
Flask/FastAPI endpoint
@app.post("/predict")
def predict_growth(request: PredictionRequest):
prediction = model.predict(request.features)
return PredictionResponse(prediction=prediction)



## Versioning
- **API Version**: 1.0.0
- **Model Version**: Tracked automatically
- **Data Schema Version**: Validated on load

## Support
- **GitHub Issues**: Technical support and bug reports
- **Documentation**: Comprehensive guides and examples
- **Community**: Best practices and use cases