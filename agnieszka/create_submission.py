import prepare_data
import pandas
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np


titanic = prepare_data.prepare("train.csv")
titanic_test = prepare_data.prepare("test.csv")


algorithms = [
    [RandomForestClassifier(random_state=1, n_estimators=10000, min_samples_split=5, min_samples_leaf=2), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]


predictions = []
for alg, predictors in algorithms:
    alg.fit(titanic[predictors], titanic["Survived"])
    predictions.append(alg.predict_proba(titanic_test[predictors].astype(float))[:,1])
test_predictions = (predictions[0] * 3 + predictions[1]) / 4

test_predictions[test_predictions <= .5] = 0
test_predictions[test_predictions > .5] = 1


submission = pandas.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": test_predictions.astype(int)
})
    
submission.to_csv("kaggle.csv", index = False)
