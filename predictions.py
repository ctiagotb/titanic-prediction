#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import csv
from time import time
from sklearn.metrics import f1_score, r2_score, make_scorer
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.grid_search import GridSearchCV


def train_classifier(clf, X_train, y_train):
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print "Time to train  {:.4f} seconds".format(end - start)


def predict_labels(clf, features, target):
    start = time()
    y_pred = clf.predict(features)
    end = time()
    print "Time to predict {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred) #, pos_label='Surv')


def train_predict(clf, X_train, y_train, X_test, y_test):
    print "Training a {} with {} points. . .".format(clf.__class__.__name__, len(X_train))
    train_classifier(clf, X_train, y_train)
    print "F1 score for training: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test: {:.4f}.".format(predict_labels(clf, X_test, y_test))


def performance_metric_r2(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score


train_file = 'data/train_processed.csv'
test_file = 'data/test_processed.csv'
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

target_col = train_data.columns[1]
feature_cols = list(train_data.columns[2:])
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Name']

X_all = train_data[feature_cols]
y_all= train_data[target_col]

X_all['FamillySize'] = X_all['SibSp'] + X_all['Parch']

X_test_original = test_data[feature_cols]
X_test_original['FamillySize'] = X_test_original['SibSp'] + X_test_original['Parch']

num_train = 750
num_test = X_all.shape[0] - num_train

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, stratify=y_all, random_state=55)

clf_A = tree.DecisionTreeClassifier(random_state=15)
clf_B = GaussianNB()
clf_C = KNeighborsClassifier()
clf_D = SVC(random_state=15, probability=True)
clf_E = linear_model.SGDClassifier(random_state=15)
clf_F = RandomForestClassifier(random_state=15)
clf_G = AdaBoostClassifier(random_state=15)
clf_H = GradientBoostingClassifier()
clf_I = LinearDiscriminantAnalysis()
clf_J = QuadraticDiscriminantAnalysis()
clf_K = LogisticRegression()

for clf in [clf_A, clf_B, clf_C, clf_D, clf_E, clf_F, clf_G, clf_H, clf_I, clf_J, clf_K]:
    train_predict(clf, X_train, y_train, X_test, y_test)
    print '='*80

cv_sets = ShuffleSplit(X_train.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

clf = RandomForestClassifier(random_state=15)
f1_scorer = make_scorer(f1_score)
parameters =  {'n_estimators': range(30,40), 'max_depth': range(1,10)}
grid_obj = GridSearchCV(clf, parameters, scoring=f1_scorer, cv=cv_sets)
grid_obj.fit(X_train, y_train)

clf = grid_obj.best_estimator_

print grid_obj.best_params_
print "F1 score {:.4f} in training set.".format(predict_labels(clf, X_train, y_train))
print "F1 score {:.4f} in test set.".format(predict_labels(clf, X_test, y_test))

y_test = clf.predict(X_test_original)

myfile = open('data/kaggle_answer.txt', 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_NONE, delimiter=',')
wr.writerow(['PassengerId', 'Survived'])
counter = 892
for y in y_test:
    wr.writerow([str(counter), str(y)])
    counter += 1