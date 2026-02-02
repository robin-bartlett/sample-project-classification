# Comparison of different classification models 🚀

> We use a dataset from the 2011 UK census to predict approximated social grade. We compare results using different classifiers and varying features, highlighting advantages and disadvantages of the different algorithms.

## 📊 Project overview

**Problem statement:** 
We use a dataset from the 2011 census in the UK. It contains 16 categorical columns (e.g. age, occupatio, health). Using these features we train models to predict the approximated social grade. 

**Aim:** 
We want to compare different multiclass classification algorithms dependent on the features used. 

**Methods:** 
- One Hot Encoding
- Decision Tree Classifier
- Random Forest Classifier
- Logistic Regression
- k-Nearest Neighbour Classifier
- Linear Support Vector Machine
- Artificial Neural Networks




## Setup

Clone the repository
```bash
# Clone repository
git clone [DEIN-REPO-LINK]
cd [REPO-NAME]
```

Instal [uv](https://uv.dev) (if not installed already) and synchronise the dependencies
```bash
# Instal dependencies 
uv sync
```

## Usage of the project

Run the notebooks in the following order:
1. notebooks/01_exploration.ipynb
2. notebooks/02_preprocessing.ipynb
3. notebooks/03_modeling.ipynb
4. notebooks/04_results.ipynb



