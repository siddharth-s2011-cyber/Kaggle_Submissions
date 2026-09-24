# Kaggle Submissions

A structured repository containing machine learning workflows, exploratory data analysis (EDA), model training pipelines, and submissions for Kaggle competitions.

## Projects

### 🚢 Titanic - Machine Learning from Disaster
- **Exploratory Data Analysis**: Feature inspection and statistical summaries (`eda.py`).
- **Data Preprocessing**: Handling missing values, one-hot encoding, and feature scaling using Scikit-Learn pipelines (`preprocessing.py`).
- **Models**:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier (`model.py`)
- **Predictions & Submissions**: Evaluation, generating Kaggle-ready submission files (`prediction.py` and `submissions/`).

## Setup & Usage

```bash
# Clone the repository
git clone https://github.com/siddharth-s2011-cyber/Kaggle_Submissions.git
cd Kaggle_Submissions

# Create virtual environment & install requirements
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
