# Scientific Instrument Maintenance and Financial Analyzer

This project analyzes maintenance and financial data for scientific instruments to optimize service agreements and predict maintenance costs. It uses machine learning to identify patterns in instrument usage, service history, and financial metrics to predict costs and optimize service delivery.

## Features

- Comprehensive financial analysis of service contracts
- Predictive modeling for repair costs
- Service efficiency metrics
- Preventative maintenance compliance tracking
- Cost and revenue analysis
- Feature importance analysis
- Model interpretation using SHAP values
- Detailed financial reporting

## Required Data Format

The model expects a CSV file with the following columns:

### Client and Contract Information
- `client_id`: Unique identifier for each client
- `contract_id`: Identifier for service agreement
- `contract_start_date`: When the service agreement began
- `contract_end_date`: When the service agreement ends
- `contract_revenue`: Revenue from service agreement

### Instrument Information
- `instrument_id`: Unique identifier for each instrument
- `instrument_type`: Type of scientific instrument (e.g., mass spec, gas chromatograph)
- `instrument_age`: Age in years

### Service Metrics
- `phone_consultations`: Number of phone calls for expert consultation
- `onsite_visits`: Number of service calls requiring onsite visit
- `days_onsite`: Total days spent on site
- `repair_hours`: Technician hours for covered repairs
- `pm_visits_scheduled`: Number of scheduled preventative maintenance visits
- `pm_visits_completed`: Number of completed preventative maintenance visits
- `client_errors`: Number of issues caused by client misuse

### Financial Metrics
- `parts_cost`: Cost of parts used in repairs
- `total_repair_cost`: Total cost of repairs (target variable)

Additional columns can be included and will be automatically processed.

## Derived Metrics

The analyzer automatically calculates several useful metrics:
- Contract duration
- PM compliance rate
- Cost per day onsite
- Revenue per visit
- Service intensity score
- Estimated profit margins

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data in CSV format with the required columns
2. Update the file path in `maintenance_analysis.py`
3. Run the analysis:
```bash
python maintenance_analysis.py
```

## Output

The script generates:
- Model performance metrics (RMSE and R² score)
- Feature importance visualization (`feature_importance.png`)
- SHAP value analysis (`shap_summary.png`)
- Detailed financial analysis report (`financial_analysis.txt`)
- Predictions for future maintenance costs

## Model Details

The project implements two types of models:
1. Random Forest Regressor (default)
   - Optimized for complex relationships between features
   - Handles both numerical and categorical data
   - Provides feature importance rankings

2. XGBoost Regressor
   - High performance gradient boosting
   - Excellent for capturing non-linear relationships
   - Advanced hyperparameter tuning

## Financial Analysis

The system provides detailed financial insights including:
- Average contract revenue
- Average repair costs
- Profit margins
- PM compliance rates
- Service call statistics
- Parts cost analysis

## Interpretation

- Feature importance plot shows which factors most strongly influence repair costs
- SHAP values provide detailed insights into how each feature affects individual predictions
- Financial analysis report provides business-level insights
- Service metrics help identify efficiency opportunities 