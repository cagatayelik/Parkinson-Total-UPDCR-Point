# Parkinson's Disease UPDRS Regression Analysis

## Introduction

As the global population ages, medical professionals face an increasing volume of patient data, making efficient diagnosis and treatment planning challenging. Machine learning models can assist clinicians by predicting disease severity based on biomedical data. This project focuses on **predicting the total UPDRS score** (a measure of Parkinson’s disease severity) using **regression analysis**.

## Dataset

### Source
The dataset used in this project is the **Oxford Parkinson's Disease Telemonitoring Dataset**, created by **Athanasios Tsanas** and **Max Little** from the University of Oxford. The dataset was collected in collaboration with **Intel Corporation** and **10 medical centers** across the US.

- **Dataset link (UCI ML Repository):** [Parkinson’s Telemonitoring Dataset](http://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring)
- **Dataset link (Kaggle):** [Parkinson's Disease Progression](https://www.kaggle.com/datasets/thedevastator/unlocking-clues-to-parkinson-s-disease-progressi)

### Data Description
This dataset consists of **5,875 voice recordings** from **42 individuals** diagnosed with early-stage Parkinson’s disease. The primary goal is to predict **motor UPDRS** and **total UPDRS** scores from **16 biomedical voice features**.

#### Features:
- **Demographics:** Subject ID, Age, Sex, Test Time (days since trial enrollment)
- **UPDRS Scores:** Motor UPDRS, Total UPDRS
- **Voice Features:**
  - **Jitter Measures** (Frequency variations)
  - **Shimmer Measures** (Amplitude variations)
  - **Harmonics-to-Noise Ratio (HNR)**
  - **Recurrence Period Density Entropy (RPDE)**
  - **Detrended Fluctuation Analysis (DFA)**
  - **Pitch Period Entropy (PPE)**

## Preprocessing Steps
1. **Data Cleaning:**
   - Removed unnecessary columns (`index`, `subject#`).
   - Checked for missing values (none found).
2. **Feature Engineering:**
   - Analyzed correlation between features.
   - Standardized numerical values.
3. **Splitting Data:**
   - 80% training, 20% testing.

## Machine Learning Models and Evaluation
Several regression models were tested for predicting **total UPDRS** scores:

| Model | RMSE (Test) | R² (Test) |
|--------|--------------|--------------|
| **Linear Regression (LR)** | 3.16 | 0.910 |
| **Decision Tree (DT)** | 0.52 | 0.998 |
| **K-Nearest Neighbors (KNN)** | 1.99 | 0.964 |
| **AdaBoost (AdaB)** | 3.04 | 0.917 |
| **Gradient Boosting (GBM)** | 1.49 | 0.980 |
| **Random Forest (RF)** | **0.36** | **0.999** |
| **Support Vector Machine (SVM)** | 3.62 | 0.882 |

### Best Performing Models:
- **Random Forest (RF):** Achieved **0.36 RMSE** and **99.9% R²**, making it the most accurate and reliable model.
- **Gradient Boosting (GBM):** Achieved **1.49 RMSE** and **98% R²**, offering a strong alternative.

## Hyperparameter Optimization
### Random Forest Optimization:
- **Best Parameters:** `{n_estimators: 200, max_depth: 18, min_samples_split: 4, min_samples_leaf: 2, max_features: 'sqrt'}`
- **Best Score:** `99.9% R²`

## Conclusion
- **Random Forest (RF) performed the best**, offering high predictive accuracy and generalization.
- **Gradient Boosting (GBM) is a strong alternative**, with lower overfitting risks than Decision Trees.
- **Linear models (LR, SVM) struggled with non-linear data complexity.**

This project demonstrates that **machine learning can effectively predict Parkinson’s disease severity** using voice biomarkers. Further improvements could include **hyperparameter tuning** and **deep learning models** for enhanced predictions.

## Repository Structure
```
├── data/                   # Raw and pre-processed data
├── models/                 # Trained regression models
├── notebooks/              # Jupyter notebooks for analysis
├── scripts/                # Python scripts for data processing and training
├── results/                # Model evaluation and performance metrics
└── README.md               # Project documentation
```

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Parkinson-UPDRS-Prediction.git
   cd Parkinson-UPDRS-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run data pre-processing:
   ```bash
   python scripts/preprocess_data.py
   ```
4. Train the models:
   ```bash
   python scripts/train_model.py
   ```
5. Evaluate performance:
   ```bash
   python scripts/evaluate_model.py
   ```

## Author
- **Çağatay Elik**

## References
- [Accurate telemonitoring of Parkinson's disease progression (Tsanas et al., 2009)](https://ieeexplore.ieee.org/document/4795365)
- [UCI ML Repository: Parkinson’s Telemonitoring Dataset](http://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring)

