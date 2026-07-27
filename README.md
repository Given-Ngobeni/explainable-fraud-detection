# Explainable Ensemble Machine Learning for Healthcare Insurance Fraud Detection
 
This repository contains the code used to produce the results reported in:
 
**"Enhancing Healthcare Insurance Fraud Detection Using Explainable Ensemble Machine Learning on Imbalanced Datasets"**
 
The pipeline builds a SHAP-guided stacking ensemble for healthcare insurance fraud detection under severe class imbalance, and uses model-level SHAP attribution to identify redundant base learners and construct reduced ensembles.
 
## Data
 
This repository does not include the dataset. The dataset used is a publicly available healthcare reimbursement dataset originally released through the 15th China University Student Service Outsourcing Innovation and Entrepreneurship Competition, and previously used by Wang et al. (2025), "A robust and interpretable ensemble machine learning model for predicting healthcare insurance fraud," Scientific Reports.
 
Access details and mirrors are provided in the Data Availability statement of the paper.
 
## Pipeline overview
 
The code follows the methodology described in Section 3 of the paper:
 
1. **Preprocessing** (Section 3.2): median imputation for missing values, retention of statistical outliers, StandardScaler normalisation, and class imbalance handling evaluated as an explicit experimental factor (original 95:5 distribution, SMOTE, and ADASYN).
2. **Feature selection** (Section 3.3): Pearson correlation filtering (threshold |r| > 0.75) followed by wrapper-based selection using Recursive Feature Elimination with cross-validation (RFECV), compared against Sequential Forward Selection (SFS).
3. **Base learner selection** (Section 3.4): heuristic screening of 32 classifiers using Lazy Predict, retaining models with ROC-AUC >= 0.65.
4. **Stacking ensemble construction** (Section 3.5): five-fold stratified cross-validation with out-of-fold predictions as meta-features, and a logistic regression meta-learner with L2 regularisation.
5. **SHAP-guided ensemble reduction** (Section 3.6): model-level SHAP attribution used to rank base learner contributions and construct reduced ensembles (Reduced-1 through Reduced-8).
6. **Evaluation** (Sections 3.7 to 3.8): AUC-PR as the primary ranking metric, supplemented by MCC, precision, recall, false positive rate, balanced accuracy, ROC-AUC, and F1-score; paired t-tests and McNemar's test used descriptively for comparison across base learners and ensemble configurations.
## Requirements
 
See `requirements.txt`. Key dependencies include:
 
- scikit-learn
- imbalanced-learn (SMOTE, ADASYN)
- shap
- xgboost
- lightgbm
- lazypredict
- pandas, numpy, matplotlib
## How to reproduce
 
1. Obtain the dataset as described in the Data section and place it under `data/`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the preprocessing and feature selection scripts.
4. Run the base learner screening and stacking pipeline.
5. Run the SHAP analysis script to reproduce the model-level attribution results and reduced ensembles.
6. Run the evaluation script to reproduce the reported metrics and figures.
Update the exact commands or notebook order once finalised.
 
## Citation
 
If you use this code, please cite the paper (full citation to be added once the DOI is assigned) and the original dataset paper:
 
Wang, Z., Chen, X., Wu, Y., Jiang, L., Lin, S., Qiu, G. (2025). A robust and interpretable ensemble machine learning model for predicting healthcare insurance fraud. Scientific Reports, 15(1), 218.
 
## License
 
Specify a license here (for example, MIT) before making the repository public.
 
## Contact
 
Add corresponding author contact details here.
