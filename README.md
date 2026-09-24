# Diabetes Risk Classification

Predicting diabetes/prediabetes risk from CDC BRFSS 2015 health survey data, with an emphasis on data validation and honest handling of class imbalance.

## Problem

Using self-reported health indicators (BMI, blood pressure, general health, physical activity, etc.), predict whether a person has diabetes or prediabetes. Framed as a binary classification problem relevant to health risk stratification — the kind of task a health insurer might use to flag members for preventive outreach.

## Data

- **Source:** CDC Diabetes Health Indicators dataset (BRFSS 2015), UCI ML Repository (ID 891)
- **Raw size:** 253,680 rows × 22 columns
- **Target:** `Diabetes_binary` (0 = no diabetes, 1 = diabetes/prediabetes)

## Data Validation

- **Missing values:** 0 across all columns
- **Duplicates:** 24,206 exact duplicate rows found and removed (253,680 → 229,474 rows). Given the dataset's mostly binary/low-cardinality features, some duplicates are plausibly different people with identical answers rather than data errors — they were dropped specifically to prevent train/test leakage, not because they were assumed to be bad data.
- **Range checks:** All values fall within the BRFSS codebook's expected ranges (BMI 12–98, GenHlth 1–5, MentHlth/PhysHlth 0–30, Age category 1–13, Education 1–6, Income 1–8)
- **Class balance:** 84.7% no diabetes, 15.3% diabetes/prediabetes — meaningfully imbalanced, addressed explicitly in modeling (see below)

## Exploratory Data Analysis

- No single feature strongly predicts diabetes alone — the strongest individual correlation is GenHlth (self-reported general health) at ~0.27
- Diabetes rate rises monotonically with self-reported health, from 3% (excellent health) to 38% (poor health)
- Diabetes rate climbs steadily with age through the mid-60s/70s, then slightly declines in the oldest bracket (70+) — possibly survivorship bias
- BMI distributions overlap heavily between diabetic and non-diabetic groups, but shift meaningfully higher for diabetics
- Takeaway: these patterns argue for a model that combines multiple weak-to-moderate signals rather than one dominant feature — supporting the inclusion of a tree-based model

## Feature Engineering

Three features were added based on the EDA findings above:

- **`BMI_category`** — bins BMI into standard clinical categories (underweight/normal/overweight/obese/severely obese), since health risk is typically discussed in these bins rather than raw BMI
- **`TotalUnhealthyDays`** — sum of `MentHlth` + `PhysHlth` (days of poor mental/physical health in the past 30), clipped at 30, to capture overall unhealthy-days burden as a single signal
- **`CardioRiskCount`** — count of `HighBP` + `HighChol` + `HeartDiseaseorAttack` + `Stroke`, to combine four related cardiovascular risk flags into one cumulative score

## Modeling

- **Split:** 80/20 train/test, stratified on the target, `random_state=42` (train: 183,579 rows, test: 45,895 rows; both splits preserve the ~15.3% diabetes rate)
- **Baseline — Logistic Regression:** features scaled with `StandardScaler` (fit on train only), `class_weight='balanced'` to address class imbalance
- **Random Forest:** 100 trees, `class_weight='balanced'`, no scaling needed (tree splits are scale-invariant)

### Results (default 0.5 threshold)

| Model               | Accuracy | Precision (diabetic) | Recall (diabetic) | F1 (diabetic) | ROC-AUC |
| ------------------- | -------- | -------------------- | ----------------- | ------------- | ------- |
| Logistic Regression | 0.72     | 0.32                 | 0.76              | 0.45          | 0.812   |
| Random Forest       | 0.81     | 0.39                 | 0.44              | 0.41          | 0.781   |

### The imbalance tradeoff

Random Forest's higher accuracy (0.81 vs 0.72) is misleading: it comes from being much better at the majority class (no diabetes) while catching less than half of actual diabetic cases (recall 0.44). Logistic Regression's balanced class weighting trades overall accuracy for catching 76% of diabetic cases.

In a real risk-screening use case, missing a genuinely high-risk member (a false negative) is more costly than flagging a healthy member for extra follow-up (a false positive) — so recall on the diabetic class matters more than raw accuracy here.

To test whether Random Forest could close this gap without switching models, I lowered its decision threshold from 0.5 to 0.3:

| Random Forest   | Accuracy | Precision (diabetic) | Recall (diabetic) | F1 (diabetic) |
| --------------- | -------- | -------------------- | ----------------- | ------------- |
| Default (0.5)   | 0.81     | 0.39                 | 0.44              | 0.41          |
| Threshold = 0.3 | 0.70     | 0.30                 | 0.73              | 0.43          |

Lowering the threshold closes most of the recall gap with Logistic Regression (0.73 vs 0.76) at a comparable precision, but at the cost of overall accuracy (0.70). This confirms that the right operating point is a business decision, not a fixed default — it depends on the real-world cost of a missed high-risk member vs. an unnecessary follow-up.

## Feature Importance

Cross-checked two methods, since Random Forest's impurity-based importance is known to be biased toward high-cardinality/continuous features:

- **Random Forest (impurity-based):** Age, BMI, Income, GenHlth, and the engineered `CardioRiskCount` rank highest
- **Logistic Regression (coefficient magnitude, scaled features):** GenHlth, Age, and `BMI_category` rank highest

Both methods agree GenHlth, Age, and BMI/BMI_category are the strongest signals. `CardioRiskCount` and `TotalUnhealthyDays` both outrank their individual component features in at least one model, validating the feature engineering. One counterintuitive result: `TotalUnhealthyDays` has a _negative_ logistic regression coefficient despite a positive raw correlation with diabetes — this is a multicollinearity effect from its overlap with `MentHlth`/`PhysHlth`, not a contradiction.

## Limitations

- Data is self-reported survey data (BRFSS), not clinical measurements — subject to recall and reporting bias
- 2015 data; health patterns and risk factors may have shifted since
- No external validation set — results are from a single train/test split, not cross-validation
- Neither model achieves strong precision on the diabetic class (0.30–0.39) — in production this would generate a meaningful number of false positives requiring follow-up
- Duplicate removal assumes exact-match rows should be deduplicated for leakage prevention; this reduces the dataset by ~9.5% and could remove some genuinely distinct individuals with identical survey answers

## Next Steps

- Cross-validation instead of a single split, for more robust performance estimates
- Hyperparameter tuning (grid/random search) for both models
- Try gradient boosting (XGBoost/LightGBM) as a stronger tree-based baseline
- SHAP values for per-prediction explainability, beyond global feature importance
- Precision-recall curve analysis across the full threshold range, not just 0.5 vs 0.3

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python 01_load_data.py   # downloads the dataset to data/
jupyter notebook oscar_diabetes_risk.ipynb
```
