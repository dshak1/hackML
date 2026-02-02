# Data Validation Report

## Dataset Summary
- Train shape: (6244474, 11)
- Test shape: (118146, 10)
- Train columns: ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'urgency_level', 'id']
- Test columns: ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'id']

## Inferred Schema
- Allowed types: ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
- Target values: [0, 1, 2, 3]
- Thresholds: {'step_min_reasonable': 1, 'step_max_reasonable': 743, 'amount_outlier_quantile': 0.999, 'amount_spike_top_n': 10, 'amount_spike_min_count': 100, 'balance_error_threshold': 1.0, 'balance_error_high_threshold': 100.0, 'name_regex': '^[A-Z]\\d+$', 'name_prefixes': ['C', 'M'], 'drift_sample_size': 200000, 'check_sample_size': 200000, 'random_state': 42}

## Validation Plan (Evidence-Based)
- Missing/NaN checks (train_missing_values: None missing).
- Missing/NaN checks (test_missing_values: None missing).
- Inf checks (train_inf_values: None inf values).
- Inf checks (test_inf_values: None inf values).
- Duplicate checks (train_duplicate_rows: None issues).
- Duplicate checks (train_duplicate_ids: None issues).
- Domain checks (type_invalid_categories: None flagged).
- Domain checks (amount_negative: None flagged).
- Domain checks (amount_outliers: 6245 flagged).
- Cross-field consistency (balance_error_orig: sample_size=200000).
- Cross-field consistency (balance_error_dest: sample_size=200000).
- Drift checks: numeric KS-stat + categorical PSI.
- Target class distribution report for imbalance tracking.

## Failed Checks
- None

## Warnings
- amount_outliers (count=6245, severity=medium)
- amount_spikes (count=3096, severity=low)
- balance_error_orig (count=113055, severity=medium)
- balance_error_dest (count=120586, severity=medium)

## Drift Summary
- Top numeric drift: [{'column': 'step', 'ks_stat': 1.0, 'p_value': 0.0, 'train_mean': 235.813795, 'test_mean': 663.601975521812}, {'column': 'id', 'ks_stat': 1.0, 'p_value': 0.0, 'train_mean': 3127642.31218, 'test_mean': 6303547.5}, {'column': 'oldbalanceOrg', 'ks_stat': 0.0707605715809253, 'p_value': 2e-323, 'train_mean': 846396.1891219, 'test_mean': 642540.1975272967}, {'column': 'newbalanceDest', 'ks_stat': 0.036626696037106665, 'p_value': 5.727303112743013e-87, 'train_mean': 1230166.1892462, 'test_mean': 1241662.1045514026}, {'column': 'newbalanceOrig', 'ks_stat': 0.03633089584073945, 'p_value': 1.41306961870253e-85, 'train_mean': 868639.9738402499, 'test_mean': 644301.0251016539}]
- Top categorical drift: [{'column': 'nameDest', 'psi': 2.5016303548031504, 'train_top': {'C1286084959': 1.8095999759147046e-05, 'C985934102': 1.74554333959914e-05, 'C665576141': 1.681486703283575e-05, 'C248609774': 1.6174300669680103e-05, 'C2083562754': 1.6174300669680103e-05}, 'test_top': {'C260674558': 4.2320518680276946e-05, 'C156150928': 3.385641494422156e-05, 'C45737324': 3.385641494422156e-05, 'C60560601': 3.385641494422156e-05, 'C1422454320': 3.385641494422156e-05}}, {'column': 'nameOrig', 'psi': 2.3959336042592794, 'train_top': {'C1976208114': 4.804247723667358e-07, 'C400299098': 4.804247723667358e-07, 'C2098525306': 4.804247723667358e-07, 'C724452879': 4.804247723667358e-07, 'C363736674': 4.804247723667358e-07}, 'test_top': {'C369448390': 1.692820747211078e-05, 'C807221466': 1.692820747211078e-05, 'C587825574': 8.46410373605539e-06, 'C50604238': 8.46410373605539e-06, 'C2054010658': 8.46410373605539e-06}}, {'column': 'type', 'psi': 0.0076481587452934, 'train_top': {'CASH_OUT': 0.35236018277920605, 'PAYMENT': 0.3379429556436619, 'CASH_IN': 0.2197208283676095, 'TRANSFER': 0.08349190019847949, 'DEBIT': 0.006484133011043044}, 'test_top': {'PAYMENT': 0.34888189189646707, 'CASH_OUT': 0.31483080256631624, 'CASH_IN': 0.23058757808135696, 'TRANSFER': 0.09772654173649552, 'DEBIT': 0.007973185719364177}}]

## Class Distribution (Train)
- Counts: {0: 6237903, 3: 2244, 1: 2176, 2: 2151}
- Percentages: {0: 0.998947709606926, 3: 0.00035935772973031836, 1: 0.00034846810155667236, 2: 0.00034446456178694953}

## Model Impact
- Issue: Amount spikes can bias decision boundaries. Impact: Macro F1 can suffer if rare classes correlate with repeated amounts. Action: Add `amount_spike_flag` for values: [10000000.0].
- Issue: Extreme amount outliers compress model scale. Impact: Minority classes with large amounts can be underfit. Action: Use `log1p_amount` and `amount_outlier_flag` (threshold=8914135.280491272).
- Issue: balance_error_orig shows balance inconsistencies (sample count=113055). Impact: Balance errors are strong fraud signals but can add noise. Action: Use `balance_error_*` features and threshold flags.
- Issue: balance_error_dest shows balance inconsistencies (sample count=120586). Impact: Balance errors are strong fraud signals but can add noise. Action: Use `balance_error_*` features and threshold flags.
- Issue: Name prefixes encode account types (customer vs merchant). Impact: Ignoring prefixes can hide high-risk merchant patterns. Action: Add `is_merchant_dest` and prefix features.
- Issue: Target class imbalance is severe. Impact: Macro F1 will drop if rare classes are ignored. Action: Use class weights or focal loss; stratified CV by `urgency_level`.
- Issue: Numeric drift in step (KS=1.000). Impact: Train/test mismatch can lower recall on rare classes. Action: Consider robust scaling or monotonic binning for drifting features.
- Issue: Categorical drift in nameDest (PSI=2.502). Impact: Category frequency shifts can bias calibration. Action: Use target encoding with smoothing or frequency-aware encoding.
