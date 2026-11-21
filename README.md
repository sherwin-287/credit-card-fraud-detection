# Credit Card Fraud Detection

Detects fraudulent card transactions in a **synthetic** dataset of 100,000 transactions (2.06% fraud).
It compares Logistic Regression, Random Forest, Gradient Boosting and XGBoost, with and without class weights and
SMOTE, using a **time-based split**, chooses the decision threshold by **cost**, and explains the model with
permutation importance and SHAP.

> The data is synthetic (dated through December 2026), so the results show that the pipeline works.
> They do not show how well it would catch real fraud.

## Why this setup
- **Accuracy is useless here:** a model that flags nothing is 98% accurate. Results use **PR-AUC** (chance level = fraud rate, about 0.02), recall and precision.
- **Leakage removed:** `fraud_type` and `fraud_amount_lost` are known only after fraud is confirmed, and ID columns would let a model memorise customers, so none are used as features.
- **Time-based split:** train Jan-Aug, validation Aug-Oct (model and threshold choice), test Oct-Dec. A random split would let the model learn from the future.

## Files
| File | What it is |
|---|---|
| `fraud_pipeline.py` | scikit-learn pipeline: split, 6 model variants, threshold tuning, importance, figures. Needs no extra installs |
| `fraud_detection_colab.ipynb` | Colab notebook adding XGBoost, SMOTE (inside an imblearn pipeline) and SHAP, saved with outputs |
| `credit_card_fraud_detection_2026.csv` | The dataset (synthetic) |
| `metrics.csv`, `recall_by_fraud_type.csv`, `permutation_importance.csv`, `summary.json` | Saved results of the script |
| `fig1_...png` to `fig5_...png` | Figures made by the script |
| `requirements.txt`, `requirements-notebook.txt` | Packages for the script and for the notebook |

## How to run
```
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
python3 -m pip install -r requirements.txt
python3 fraud_pipeline.py           # about 1 minute; writes the result files and figures in the same folder
```
Notebook: upload `fraud_detection_colab.ipynb` to Google Colab, choose Runtime > Run all and upload the CSV when asked.
The notebook installs `imbalanced-learn`, `xgboost` and `shap` itself.

## Results: Colab notebook (4 models x 3 imbalance methods; test = last 20% of the year, 417 fraud cases)
| Model | None: PR-AUC | Class weights: PR-AUC | SMOTE: PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.127 | 0.122 | 0.120 |
| Random Forest | 0.144 | 0.135 | 0.124 |
| Gradient Boosting | 0.137 | 0.151 | 0.139 |
| XGBoost | 0.160 | 0.145 | 0.150 |

- **All variants rank transactions similarly** (ROC-AUC 0.83-0.85; PR-AUC 0.12-0.16, about 6 to 8 times the 0.02 chance level). SMOTE did not improve ranking quality.
- **Selected on validation PR-AUC:** XGBoost with SMOTE (0.156).
- **The threshold matters more than the model.** At 0.5 the selected model flags almost nothing (recall 0.005), so the simulated loss on the test period was 66,500 USD against 67,137 USD with no model. With a cost-based threshold of 0.055 chosen on validation it fell to **33,261 USD, about half** (each flagged transaction costs an assumed 5 USD to review).
- **Fraud types caught at that threshold:** Synthetic-Identity 66%, Card-Not-Present 63%, Account-Takeover 52%, Card-Present 34%.
- **SHAP, most important features:** international transaction, transactions in the last hour, online channel, distance from home, account age, previous chargebacks, high-risk country.

## Results: `fraud_pipeline.py` (scikit-learn only: 3 models x with/without class weights)
- Selected on validation: Gradient Boosting with class weights. Test ROC-AUC 0.839, PR-AUC 0.138.
- Threshold 0.5 gives a simulated test cost of about 35,500 USD; the validation-tuned threshold (0.70) about 37,700 USD. Unlike the XGBoost/SMOTE model, tuning did not help here, because class-weighted models already produce scores that reach 0.5.
- Permutation importance: high-risk country, distance from home, previous chargebacks, transactions in the last hour, account age.

## Limitations
- Synthetic data; precision is low (about 11-12% at the tuned thresholds), so most flagged transactions are legitimate.
- Review cost (5 USD) is an assumption; hyperparameters were not tuned; one time split, no repeated evaluation.
- Differences between models are small and within what a different split could change.
