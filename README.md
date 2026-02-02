# HackML: Fraud Detection with Data Validation

A machine learning project for detecting fraudulent transactions using advanced data validation and model training pipelines.

## Project Overview

This project implements a comprehensive fraud detection system that includes:

- **Data Validation Layer**: Automated validation, cleaning, and quality checks for financial transaction data
- **Model Training**: Baseline machine learning models for fraud classification
- **Reporting**: Detailed validation reports and model performance metrics
- **Testing**: Comprehensive test suite for data validation components

## Project Structure

```
hackML/
├── README.md                    # This file - main project documentation
├── README_DATA_VALIDATION.md    # Data validation specific documentation
├── validation_report.md         # Sample validation report output
├── fraud/                       # Directory for fraud datasets (train.csv, test.csv)
├── scripts/
│   ├── train_model.py          # Model training script
│   ├── validate_data.py        # Data validation script
│   └── learning.md             # Learning notes on type hinting/validation
├── src/
│   ├── __init__.py
│   └── data_validation/
│       ├── __init__.py
│       ├── clean.py            # Data cleaning utilities
│       ├── report.py           # Report generation
│       ├── schema.py           # Validation schema definitions
│       └── validators.py       # Validation logic
└── tests/
    ├── conftest.py             # Test configuration
    └── test_data_validation.py # Data validation tests
```

## Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dshak1/hackML.git
cd hackML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

Place your fraud detection datasets in the `fraud/` directory:
- `fraud/train.csv` - Training data with target column
- `fraud/test.csv` - Test data without target column

Expected columns in train.csv:
- `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `urgency_level`, `id`

Expected columns in test.csv:
- Same as train.csv but without `urgency_level`

### 2. Validate Data

Run data validation to check data quality and generate reports:

```bash
python scripts/validate_data.py \
  --train fraud/train.csv \
  --test fraud/test.csv \
  --out_dir runs \
  --mode warn
```

For strict validation (fails on any issues):
```bash
python scripts/validate_data.py \
  --train fraud/train.csv \
  --test fraud/test.csv \
  --out_dir runs \
  --mode strict
```

To also generate cleaned data and validation features:
```bash
python scripts/validate_data.py \
  --train fraud/train.csv \
  --test fraud/test.csv \
  --out_dir runs \
  --mode warn \
  --write_cleaned
```

### 3. Train Model

Train a baseline fraud detection model:

```bash
python scripts/train_model.py \
  --train fraud/train.csv \
  --test fraud/test.csv \
  --out_dir runs
```

Optional parameters:
- `--sample_size`: Stratified sample size (default: 500000, 0 for full dataset)
- `--min_per_class`: Minimum samples per class (default: 1000)
- `--random_state`: Random seed (default: 42)

## Data Validation Details

The data validation layer performs comprehensive checks:

### Validation Checks
- **Missing Values**: Detects NaN/null values
- **Infinite Values**: Identifies inf/-inf values
- **Duplicates**: Checks for duplicate rows and IDs
- **Domain Validation**: Validates transaction types, amounts, account names
- **Cross-field Consistency**: Verifies balance calculations
- **Outlier Detection**: Identifies amount outliers and spikes
- **Data Drift**: Compares train/test distributions

### Output Files
- `runs/validation_report.md` - Human-readable summary
- `runs/validation_report.json` - Machine-readable detailed results
- `runs/cleaned_train.csv` - Cleaned training data (if --write_cleaned)
- `runs/cleaned_test.csv` - Cleaned test data (if --write_cleaned)
- `runs/validation_features.csv` - Additional validation flags for modeling

### Interpreting Results
- **Failed Checks**: Critical issues requiring immediate attention
- **Warnings**: Potential issues that may affect model performance
- **Model Impact**: Links validation issues to expected F1 degradation

## Model Training Details

The training script supports:

### Models
- Random Forest Classifier (default)
- Logistic Regression (configurable)

### Features
- Original transaction features
- Validation flags (when available)
- One-hot encoded categorical variables
- Imputed missing values

### Evaluation
- Macro F1 score
- Class-wise performance metrics
- Feature importance analysis

## Development

### Running Tests

Run the test suite:

```bash
python -m pytest tests/
```

### Code Quality

- Use type hints throughout the codebase
- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes

### Adding New Validations

1. Define validation logic in `src/data_validation/validators.py`
2. Add schema rules in `src/data_validation/schema.py`
3. Update tests in `tests/test_data_validation.py`
4. Update documentation in `README_DATA_VALIDATION.md`

### Extending the Model

1. Modify `scripts/train_model.py` to add new models
2. Update feature engineering in the pipeline
3. Add new evaluation metrics
4. Update model serialization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the full test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues:
- Check the validation report for data-related issues
- Review the test output for code issues
- Open an issue on GitHub for bugs or feature requests</content>
<parameter name="filePath">/home/hmarthens/Desktop/hackathon/hackML/README.md
