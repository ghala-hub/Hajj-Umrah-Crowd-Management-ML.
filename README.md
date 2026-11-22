# Early Health Risk Alert System for Pilgrims
### Emergency Event Prediction in Hajj & Umrah Crowds

A team project (3 members) developed during the Data Bootcamp organized by the Computer Club at Umm Al-Qura University.

## Project Overview
This project builds and compares classification models that predict whether an emergency event may occur in Hajj and Umrah crowds, based on crowd conditions, environmental factors, and individual-related features.

**Target variable:** `Emergency_Event` (Yes/No)

## Dataset
A synthetic Hajj and Umrah crowd management dataset (10,000 records, 30 columns), used because real data of this kind is sensitive.

Source: [Hajj and Umrah Crowd Management Dataset – Kaggle](https://www.kaggle.com/datasets/ziya07/hajj-and-umrah-crowd-management-dataset)

## Tools
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Google Colab

## Methodology
1. **Data exploration:** structure, summary statistics, outliers (box plots), and correlations (heatmap).
2. **Data cleaning:** checked missing values and duplicates, and standardized column names and text.
3. **Encoding:** mapped ordered categories (age group, crowd density, fatigue, stress) and the target to numbers; applied one-hot encoding to health condition.
4. **Feature engineering:** created `combined_risk`, `heat_risk`, and `crowd_pressure`.
5. **Modeling:** 80/20 train-test split, then trained three models on the same split: Decision Tree, Random Forest, and XGBoost.
6. **Evaluation:** accuracy on training and test data, plus a classification report and confusion matrix for XGBoost.

## Results
| Model | Train accuracy | Test accuracy |
|---|---|---|
| Decision Tree | 52.0% | 52.2% |
| Random Forest | 63.6% | 50.0% |
| XGBoost | 76.2% | 50.3% |

Test accuracy is around 50% for all three models, so none shows reliable emergency prediction. XGBoost performs much better on training data than on test data, which suggests overfitting.

## Conclusion
Because the dataset is synthetic and its features show very weak correlations, these results cannot establish real-world reliability. Testing with representative real-world data is needed before any practical use.

## How to run
Open `Hajj_Umrah_Crowd_Management_ML.ipynb` in Google Colab, upload the dataset from Kaggle, and run the cells in order.

## Files
- `Hajj_Umrah_Crowd_Management_ML.ipynb`: full notebook
