# Auto-generated results

_Data: SYNTHETIC DEMO (not real results)_

- 40,000 transactions, 224 fraud (0.560%). Split 60/20/20 (train/validation/test), stratified.
- The baseline that never flags fraud scores **99.44% accuracy** but catches **0%** of fraud, so accuracy is misleading here.
- Best model (by validation PR-AUC): **Logistic Regression**. On the test set it reaches precision 75.0% and recall 53.3%, catches 81.5% of fraud value, with 1.00 false alerts per 1,000 transactions.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Fraud value caught | False alerts / 1,000 |
|---|---|---|---|---|---|---|---|
| Baseline: never flags fraud | 0.006 | 0.500 | 0.000 | 0.000 | 0.000 | 0.0% | 0.00 |
| Logistic Regression | 0.646 | 0.963 | 0.750 | 0.533 | 0.623 | 81.5% | 1.00 |
| Random Forest | 0.580 | 0.937 | 0.769 | 0.444 | 0.563 | 17.1% | 0.75 |
| Gradient Boosting | 0.382 | 0.873 | 0.667 | 0.356 | 0.464 | 68.0% | 1.00 |
