import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -------------------------
# Load
# -------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# -------------------------
# X / y
# -------------------------
X = train.drop(columns=["Survived", "PassengerId"])
y = train["Survived"]
X_test = test.drop(columns=["PassengerId"])
# -------------------------
# Split
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# Columns
# -------------------------
numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]
categorical_features = [
    "Sex",
    "Embarked"
]
# -------------------------
# Numerical pipeline
# -------------------------
numeric_pipeline = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5, weights="uniform")),
    ("scaler", StandardScaler())
])


# -------------------------
# Categorical pipeline
# -------------------------

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "onehot",
        OneHotEncoder(handle_unknown="ignore")
    )
])


# -------------------------
# ColumnTransformer
# -------------------------

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        numeric_features
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_features
    )
])


# -------------------------
# Full model
# -------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])


# -------------------------
# Train
# -------------------------

model.fit(X_train, y_train)


# -------------------------
# Validation
# -------------------------

y_pred = model.predict(X_val)

print(
    "Validation accuracy:",
    accuracy_score(y_val, y_pred)
)