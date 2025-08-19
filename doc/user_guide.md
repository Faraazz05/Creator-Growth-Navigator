# Creator Growth Navigator - User Guide

## Overview
Creator Growth Navigator is a data-driven tool that predicts Instagram follower growth based on posting frequency while providing rich insights into content strategy and engagement patterns.

------------------

## Getting Started

### Installation
```bash
git clone https://github.com/your-username/creator-growth-navigator
cd creator-growth-navigator
pi
```
----------------

### Quick Start
```bash
streamlit run streamlit_app/Home.py
```
----------------

## Using the Application

### 1. Home Page
- **Overview**: Project introduction and key assumptions
- **Navigation**: Access to all analysis modules
- **Quick Stats**: Summary of your model performance

### 2. Predictor
- **Input Controls**: Adjust posting frequency and content mix
- **Predictions**: Real-time follower growth predictions
- **Confidence Intervals**: Uncertainty quantification around predictions
- **Scenario Planning**: Compare different posting strategies

### 3. Growth History
- **Time Series**: Visualize historical growth patterns
- **Actual vs Predicted**: Model performance over time
- **Trend Analysis**: Identify seasonal patterns and growth phases

### 4. Decision Support
- **ROI Calculator**: Time investment vs growth analysis
- **Counterfactuals**: "What if" scenario analysis
- **Recommendations**: Actionable insights based on your data

### 5. Model Health
- **Performance Metrics**: R², RMSE, MAE tracking
- **Diagnostics**: Residual analysis and assumption checking
- **Drift Detection**: Alerts when model needs retraining
- **Feature Importance**: Understanding which factors drive growth

--------------------

## Key Features

### Core Prediction Engine
- **Simple Linear Regression**: Interpretable relationship between posting frequency and growth
- **Weekly Aggregation**: Focus on sustainable posting patterns
- **Confidence Intervals**: Quantified prediction uncertainty

### Advanced Analytics
- **Content Mix Analysis**: Post/story/reel ratio optimization
- **Timing Analysis**: Optimal posting window identification
- **Consistency Metrics**: Posting regularity impact assessment
- **Engagement Weighting**: Quality over quantity insights

### Business Intelligence
- **ROI Metrics**: Follows per hour of content creation
- **Saturation Warnings**: Identify diminishing returns
- **Competitor Benchmarking**: Industry comparison features

---------------------

## Data Requirements

### Input Format
- **Daily metrics**: Posts, stories, reels, engagement data
- **Time series**: Minimum 3 months of historical data
- **Follower counts**: Daily follower numbers

### Supported File Types
- CSV files with daily creator metrics
- JSON exports from social media management tools

-------------------

## Best Practices

### Data Quality
1. **Consistent tracking**: Ensure daily data collection
2. **Complete records**: Minimize missing values
3. **Accurate timestamps**: Maintain proper date formatting

### Model Usage
1. **Regular updates**: Retrain monthly with new data
2. **Validation**: Monitor prediction accuracy over time
3. **Context awareness**: Consider external factors affecting growth

### Strategic Application
1. **Sustainable growth**: Focus on long-term posting consistency
2. **Quality content**: Balance frequency with engagement quality
3. **Audience understanding**: Tailor strategies to your specific audience

--------------------

## Troubleshooting

### Common Issues
- **Data import errors**: Check CSV format and column names
- **Model performance**: Ensure sufficient historical data
- **Prediction accuracy**: Consider external factors and model limitations

### Support
- Check the API documentation for technical details
- Review model documentation for algorithmic insights
- Submit issues on GitHub for bug reports

-------------------

## Limitations

### Model Scope
- **Single platform**: Designed specifically for Instagram
- **Posting frequency focus**: Primary lever is weekly posting cadence
- **Historical patterns**: Predictions based on past behavior patterns

### External Factors
- Algorithm changes on Instagram
- Viral content effects
- Seasonal and cultural events
- Competitor actions

-------------------

## Updates and Maintenance

### Regular Tasks
- **Monthly retraining**: Update model with recent data
- **Performance monitoring**: Track prediction accuracy
- **Feature validation**: Ensure data quality standards

### Version Updates
- Check CHANGELOG.md for recent improvements
- Follow semantic versioning for compatibility
- Review breaking changes before upgrading