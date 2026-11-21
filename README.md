# Credit Card Fraud Detection

Detects fraudulent card transactions in a **synthetic** dataset of 100,000 transactions (2.06% fraud).
It compares Logistic Regression, Random Forest and Gradient Boosting (plus XGBoost and SMOTE in the Colab notebook)
using a **time-based split**, picks a decision threshold by **cost**, explains the model with permutation importance
(SHAP in the notebook),.

> The data is synthetic (dated through December 2026), so the results show that the pipeline works.
> They do not show how well it would catch real fraud.

## Why this setup
- **Accuracy is useless here:** a model that flags nothing is 98% accurate. Results use **PR-AUC** (chance level = fraud rate, about 0.02), recall and precision.
- **Leakage removed:** `fraud_type` and `fraud_amount_lost` are known only after fraud is confirmed, and ID columns would let a model memorise customers, so none are used as features.
- **Time-based split:** train Jan-Aug, validation Aug-Oct (model and threshold choice), test Oct-Dec. A random split would let the model learn from the future.

## Files
| File | What it is |
|---|---|
| `src/fraud_pipeline.py` | Full scikit-learn pipeline: split, 6 model variants, threshold tuning, importance, figures. Needs no extra installs |
| `notebooks/fraud_detection_colab.ipynb` | Colab notebook adding XGBoost, SMOTE (inside an imblearn pipeline) and SHAP |
| `data/credit_card_fraud_detection_2026.csv` | The dataset (24 MB) |
| `results/`, `figures/` | Saved metrics tables and figures |

## How to run
```
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
python3 -m pip install -r requirements.txt
python3 src/fraud_pipeline.py       # about 1 minute; writes results/ and figures/
```
Notebook: upload `notebooks/fraud_detection_colab.ipynb` to Google Colab, choose Runtime > Run all and upload the CSV when asked.
The notebook installs `imbalanced-learn`, `xgboost` and `shap` itself.

## Results (saved run of `src/fraud_pipeline.py`; test set = last 20% of the year, 20,000 transactions, 417 fraud)
| Model | Class weights | ROC-AUC | PR-AUC | Precision @0.5 | Recall @0.5 |
|---|---|---|---|---|---|
| Logistic Regression | no | 0.836 | 0.127 | 0.556 | 0.012 |
| Random Forest | no | 0.835 | 0.144 | 0.000 | 0.000 |
| Gradient Boosting | no | 0.836 | 0.129 | 0.359 | 0.034 |
| Logistic Regression | yes | 0.834 | 0.122 | 0.059 | 0.748 |
| Random Forest | yes | 0.839 | 0.141 | 0.375 | 0.029 |
| Gradient Boosting | yes | 0.839 | 0.138 | 0.074 | 0.695 |

- **All models are close on ranking quality** (ROC-AUC 0.83-0.84; PR-AUC 0.12-0.14, about 6 to 7 times the 0.021 chance level). Differences between them are small.
- **Class weights change the threshold behaviour, not the ranking:** without them, models almost never reach a score of 0.5, so recall at 0.5 is near zero. With them, recall rises to about 0.70-0.75 but precision falls to about 6-7%.
- **Selected model** (best validation PR-AUC): Gradient Boosting with class weights. Test ROC-AUC 0.839, PR-AUC 0.138.
- **Cost view** (missed fraud loses its amount; each review costs an assumed 5 USD): with no model, the test period loses about 67,100 USD. Threshold 0.5 gives a total cost of about 35,500 USD and the validation-tuned threshold (0.70) about 37,700 USD. Both roughly halve the cost, but **tuning the threshold on validation did not beat 0.5 on test**: the cost curve is flat across 0.4-0.7, so the exact choice is noise-level.
- **Most important features** (permutation importance): high-risk country, distance from home, previous chargebacks, transactions in the last hour, account age, international transaction.
- **By fraud type** (tuned threshold, test): Synthetic-Identity 62% caught, Card-Not-Present 51%, Account-Takeover 43%, Card-Present 19%. Card-Present fraud is hardest to catch.

Figures: `figures/fig1_model_comparison.png` to `fig5_recall_by_fraud_type.png`.

## Limitations
- Synthetic data; precision stays low (about 11% at the tuned threshold), so most flagged transactions are legitimate.
- Review cost (5 USD) is an assumption; hyperparameters were not tuned; one time split, no repeated evaluation.
- The notebook's XGBoost, SMOTE and SHAP results are produced when you run it in Colab and are not included here.
