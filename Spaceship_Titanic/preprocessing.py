from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,RobustScaler,OneHotEncoder,LabelEncoder
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd

train_set=pd.read_csv("train.csv")
test_set=pd.read_csv("test.csv")

X_train=train_set.drop(columns=["Transported","Name"])
y_train=train_set["Transported"]

X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.2,stratify=y_train,random_state=43)

numerical_cols=[
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]
one_hot_encoder_cols=[
    "HomePlanet",
    "Destination",
]
true_false_cols=[
    "CryoSleep",
    "VIP"
]
numeric_pipeline=Pipeline([
    ("scaler",RobustScaler()),
("imputer",KNNImputer(n_neighbors=8,weights="uniform"))
])
one_hot_encoder_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])
true_false_cols_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
])

preprocessing_pipeline=ColumnTransformer([
    ("num",numeric_pipeline,numerical_cols),
    ("one_hot_encoder",one_hot_encoder_pipeline,one_hot_encoder_cols),
    ("true_false",true_false_cols_pipeline,true_false_cols)
])
for col in true_false_cols:
    X_train[col] = X_train[col].map({True: 1, False: 0})
    X_val[col] = X_val[col].map({True: 1, False: 0})
    test_set[col] = test_set[col].map({True: 1, False: 0})

model=Pipeline([
    ("preprocessor",preprocessing_pipeline),
    ("classifier",LogisticRegression(max_iter=1000,random_state=42,class_weight="balanced"))
])
model.fit(X_train,y_train)
y_pred=model.predict(X_val)
print(classification_report(y_val,y_pred))
y_pred=model.predict(test_set)
submission=pd.DataFrame({
    "PassengerId":test_set["PassengerId"],
    "Transported":y_pred
})
submission.to_csv("submission.csv",index=False)
