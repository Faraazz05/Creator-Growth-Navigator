# Creator Growth Navigator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Creator Growth Navigator is a data-driven tool designed to model and predict Instagram follower growth based on a single lever—average weekly posting frequency—while incorporating rich features like content type breakdown, engagement weighting, posting time analysis, and ROI insights.

Built with interpretability, robust diagnostics, and actionability in mind, it includes a Streamlit interface for scenario planning and model health visualization.

-----

## Features

- Simple linear regression with advanced feature engineering
- Time-aware validation and predictive confidence intervals
- Counterfactual decision support: How posting cadence affects growth
- Model diagnostics, drift detection, and saturation warnings
- Data generated/sourced for realistic creator behaviors over 2 years

## Data

- Raw data placed in `data/raw/`
- Processed and feature-engineered datasets in `data/processed/`
- Synthetic daily creator data modeling 2 years of activity

## Installation

pip install -r requirements.txt


## Usage

Launch the Streamlit app:

streamlit run streamlit_app/Home.py


## License

This project is licensed under the MIT License.

---
