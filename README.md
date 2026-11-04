# German Credit Risk - Multi-Model Classification Demo

## a. Problem Statement

Banks need to decide whether to extend credit to an applicant based on their financial and
personal profile. This project builds and compares five classical machine-learning classifiers
that predict whether a loan applicant is a **good credit risk** or a **bad credit risk**, and
exposes the trained models through an interactive Streamlit web app so a reviewer can upload
test data and see live predictions and evaluation metrics without retraining anything.

## b. Dataset Description

- **Source:** UCI Machine Learning Repository - [Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **Instances:** 1000
- **Features:** 20 (7 numerical: duration, credit amount, installment rate, residence since,
  age, existing credits, number of dependents; 13 categorical: checking account status, credit
  history, purpose, savings account, employment duration, personal status/sex, other
  debtors/guarantors, property, other installment plans, housing, job, telephone, foreign
  worker)
- **Target:** binary - `credit_risk` (`1` = good credit risk, `0` = bad credit risk), derived
  from the original `class` attribute (`1` = good, `2` = bad)
- **Class balance:** 700 good / 300 bad (70% / 30%) in the full dataset
- **Split:** 80% train / 20% test, stratified on the target (`random_state=42`)
- Categorical columns are one-hot encoded and numerical columns are standardized inside a
  scikit-learn `Pipeline`, so the saved model files already contain the full preprocessing step
  and can score raw-format CSVs directly.

## c. GitHub Repository Link

https://github.com/achalofficial/ml-assignment2-german-credit

## d. Models Used

All 5 models were trained on the **same** preprocessed dataset and evaluated on the same held-out
20% test split (`test_data.csv`, 200 rows).

### Comparison Table

| ML Model Name          | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|-------------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression     | 0.710    | 0.760 | 0.777     | 0.821  | 0.799 | 0.284 |
| Decision Tree           | 0.685    | 0.694 | 0.824     | 0.700  | 0.757 | 0.327 |
| KNN                     | 0.695    | 0.682 | 0.734     | 0.886  | 0.803 | 0.172 |
| Naive Bayes             | 0.680    | 0.711 | 0.833     | 0.679  | 0.748 | 0.335 |
| Support Vector Machine  | 0.740    | 0.783 | 0.772     | 0.893  | 0.828 | 0.323 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Solid, well-balanced baseline (Acc 0.71, AUC 0.76). Precision and recall are both reasonable, suggesting the decision boundary between good/bad risk is close to linear once categorical variables are one-hot encoded. |
| Decision Tree | Highest precision among the simpler models (0.824) but the lowest recall of the non-KNN models (0.70) - it is conservative about labelling someone "good risk", missing some actually-good applicants. A single tree (`max_depth=6`) is also the most prone to overfitting/variance of the five. |
| KNN | Highest recall (0.886) but the lowest MCC (0.172) and AUC (0.682) - it leans heavily toward predicting "good risk", which inflates recall at the cost of more false positives (approving risky applicants), and it is the most sensitive to the curse of dimensionality after one-hot encoding expands the feature space. |
| Naive Bayes | Highest MCC (0.335) and precision (0.833) despite the lowest raw accuracy (0.68) - the conditional-independence assumption is clearly violated by correlated features like credit amount/duration, but when it does predict "good risk" it is usually right. |
| Support Vector Machine | **Best overall** - highest accuracy (0.740), AUC (0.783), and F1 (0.828), with the second-highest MCC. The RBF kernel captures non-linear interactions between features (e.g., credit amount x duration x employment) better than the linear/independence-assuming models. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** - it leads on Accuracy, AUC and F1, and is a close second on MCC, making it the most consistently strong classifier for this dataset. Naive Bayes is the runner-up if false positives (approving a bad-risk applicant) are the costlier error, since it has the best precision/MCC trade-off. |

## Project Structure

```
project-folder/
├── app.py                  # Streamlit app (loads saved pipelines, scores uploaded CSV)
├── train.py                 # Trains all 5 models, saves pipelines + metrics
├── requirements.txt
├── README.md
├── test_data.csv             # Held-out 20% test split used for the live demo
├── data/
│   ├── german.data           # Raw UCI source file
│   ├── german.doc             # UCI attribute documentation
│   └── german_credit.csv      # Cleaned full dataset (1000 rows, labeled)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── metrics.json
```

## Running Locally

```bash
pip install -r requirements.txt
python train.py        # optional - re-trains and overwrites model/ and test_data.csv
streamlit run app.py
```

## Live Streamlit App

<!-- TODO: paste your live Streamlit Community Cloud URL here after deploying (see NEXT_STEPS.md) -->

(Deploy via Streamlit Community Cloud - see Step 6 of the assignment.)
