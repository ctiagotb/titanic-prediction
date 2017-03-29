#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def preprocess_features(X, features_2_process):
    output = pd.DataFrame(index=X.index)
    output.fillna(0)

    for col, col_data in X.iteritems():
        if col in features_2_process:
            if col_data.dtype == object:
                if col == 'Sex':
                    col_data = col_data.replace(['male', 'female'], [0, 1])
                    col_data = col_data.astype(int)
                elif col == 'Name':
                    titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Rev.', 'Don.', 'Sir.', 'Dr.', 'Mme.', 'Ms.',
                              'Major.', 'Lady.', 'Mlle.', 'Col.', 'Capt.', 'Countess', 'Jonkheer.', 'Dona.']
                    for key, col_data_unity in enumerate(col_data):
                        for key_t, title in enumerate(titles):
                            if title in col_data_unity:
                                col_data[key] = key_t
                else:
                    col_data = col_data.replace(['C', 'Q', 'S', np.nan], [0, 1, 2, -1])
                    col_data = col_data.astype(int)
            if col_data.dtype == float:
                col_data = col_data.replace([np.nan], [0])
                col_data = col_data.astype(int)

        output = output.join(col_data)

    return output


train_file = 'data/train.csv'
test_file = 'data/test.csv'
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

target_col = train_data.columns[1]
feature_cols = list(train_data.columns[2:])
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Name']

passengerId = train_data['PassengerId']
X_all = train_data[feature_cols]
y_all = train_data[target_col]


features_2_process = ['Sex', 'Age', 'Fare', 'Embarked', 'Name']
X_all = preprocess_features(X_all, features_2_process)

result = pd.concat([passengerId, y_all, X_all], axis=1)
result.columns.values[0] = "PassangerId"
result.to_csv('data/train_processed.csv', index=False)


passengerIdTest = test_data['PassengerId']
X_test = test_data[feature_cols]
X_test = preprocess_features(X_test, features_2_process)


result = pd.concat([passengerIdTest, X_test], axis=1)
result.columns.values[0] = "PassangerId"
result.to_csv('data/test_processed.csv', index=False)


