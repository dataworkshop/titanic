import prepare_data
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


titanic = prepare_data.prepare("train.csv")
titanic_test = prepare_data.prepare("test.csv")


kf = KFold(titanic.shape[0], n_folds=3, random_state=1)


algorithms = [
    [RandomForestClassifier(random_state=1, n_estimators=10000, min_samples_split=5, min_samples_leaf=2), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]


predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    for alg, predictors in algorithms:
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    test_predictions = (full_test_predictions[0] * 3 + full_test_predictions[1]) / 4
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)


accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)

