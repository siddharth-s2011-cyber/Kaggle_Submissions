import os
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from xgboost import XGBClassifier

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

print("Train shape:",train.shape)
print("Test shape:",test.shape)

X=train.drop(columns=["Survived","PassengerId"])
y=train["Survived"]
X_test=test.drop(columns=["PassengerId"])

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

numeric_features=["Pclass","Age","SibSp","Parch","Fare"]
categorical_features=["Sex","Embarked"]

numeric_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

categorical_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])

preprocessor=ColumnTransformer([
    ("num",numeric_pipeline,numeric_features),
    ("cat",categorical_pipeline,categorical_features)
])

logistic_model=Pipeline([
    ("preprocessor",preprocessor),
    ("classifier",LogisticRegression(max_iter=1000))
])

random_forest_model=Pipeline([
    ("preprocessor",preprocessor),
    ("classifier",RandomForestClassifier(n_estimators=300,max_depth=6,random_state=42))
])

xgboost_model=Pipeline([
    ("preprocessor",preprocessor),
    ("classifier",XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.03,subsample=0.8,colsample_bytree=0.8,random_state=42,eval_metric="logloss"))
])

models={
    "Logistic Regression":logistic_model,
    "Random Forest":random_forest_model,
    "XGBoost":xgboost_model
}

results=[]

for name,model in models.items():
    print("\n"+"="*60)
    print(name)
    print("="*60)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_val)
    accuracy=accuracy_score(y_val,y_pred)
    print("\nValidation Accuracy:")
    print(accuracy)
    print("\nClassification Report:")
    print(classification_report(y_val,y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val,y_pred))
    results.append({"Model":name,"Validation Accuracy":accuracy})

results_df=pd.DataFrame(results)
results_df=results_df.sort_values("Validation Accuracy",ascending=False)

print("\n"+"="*60)
print("MODEL COMPARISON")
print("="*60)
print(results_df.to_string(index=False))

print("\n"+"="*60)
print("5-FOLD CROSS VALIDATION")
print("="*60)

cv_results=[]

for name,model in models.items():
    scores=cross_val_score(model,X,y,cv=5,scoring="accuracy")
    print(f"\n{name}")
    print("Fold scores:")
    print(scores)
    print("Mean CV accuracy:")
    print(scores.mean())
    print("Std:")
    print(scores.std())
    cv_results.append({"Model":name,"Mean CV Accuracy":scores.mean(),"CV Std":scores.std()})

cv_results_df=pd.DataFrame(cv_results)

print("\n"+"="*60)
print("CROSS VALIDATION COMPARISON")
print("="*60)
print(cv_results_df.to_string(index=False))

os.makedirs("submissions",exist_ok=True)

print("\n"+"="*60)
print("TRAINING FINAL MODELS")
print("="*60)

for name,model in models.items():
    print(f"\nTraining {name} on full dataset...")
    model.fit(X,y)

print("\n"+"="*60)
print("GENERATING TEST PREDICTIONS")
print("="*60)

predictions={}

for name,model in models.items():
    predictions[name]=model.predict(X_test)
    print(f"{name}: {len(predictions[name])} predictions")

submission_logistic=pd.DataFrame({
    "PassengerId":test["PassengerId"],
    "Survived":predictions["Logistic Regression"]
})
submission_logistic.to_csv("submissions/submission_logistic.csv",index=False)

submission_rf=pd.DataFrame({
    "PassengerId":test["PassengerId"],
    "Survived":predictions["Random Forest"]
})
submission_rf.to_csv("submissions/submission_random_forest.csv",index=False)

submission_xgb=pd.DataFrame({
    "PassengerId":test["PassengerId"],
    "Survived":predictions["XGBoost"]
})
submission_xgb.to_csv("submissions/submission_xgboost.csv",index=False)

print("\n"+"="*60)
print("SUBMISSION FILES")
print("="*60)

print("\nLogistic Regression:")
print(submission_logistic.head())

print("\nRandom Forest:")
print(submission_rf.head())

print("\nXGBoost:")
print(submission_xgb.head())

print("\nFiles created:")
print("submissions/submission_logistic.csv")
print("submissions/submission_random_forest.csv")
print("submissions/submission_xgboost.csv")
print("\nDone!")
