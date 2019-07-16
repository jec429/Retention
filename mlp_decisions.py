import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

X = pd.read_pickle("./data_x_numeric.pkl")
Y = pd.read_pickle("./data_y.pkl")
X['Status'] = Y['Status']
X_resigned = X[X['Status'] == True]
X_resigned.info()

y = pd.Series()
y = X['Status']
X = X.drop(['Status'], axis=1)

y_resigned = pd.Series()
y_resigned = X_resigned['Status']
X_resigned = X_resigned.drop(['Status'], axis=1)

print(X.head(10))
X.info()
cols = X.columns
X = np.array(X.values)
X_resigned = np.array(X_resigned.values)
y = np.array(y.values.astype(int))
print(len(X), len(y))

X = StandardScaler().fit_transform(X)
X_resigned = StandardScaler().fit_transform(X_resigned)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=0)

#y_train = np.where(y_train==0, -1, y_train)
#y_test = np.where(y_test==0, -1, y_test)

mlp = MLPClassifier(max_iter=100)

parameter_space = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

mlp.fit(X_train, y_train)

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train, y_train)

#print(estimator.score(X_test, y_test))

# Best parameter set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_true, y_pred = y_test, clf.predict(X_test)

print('true=', y_true)

tp = 0
fn = 0
tp2 = 0
fn2 = 0
for t, p in zip(y_true, y_pred):
    #print('s=', t, p)
    if t == 1:
        if p == 1:
            tp += 1
    if t == 0:
        if p == 1:
            fn += 1
    if t == 1:
        if p == 0:
            tp2 += 1
    if t == 0:
        if p == 0:
            fn2 += 1

print(tp, fn, tp2, fn2)

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
print('Results on the test set:')
print(classification_report(y_true, y_pred, target_names=['Active', 'Resigned']))
#print('accuracy=', accuracy_score(y_true, y_pred))
#print('recall=', recall_score(y_true, y_pred, pos_label=0))
#print('f1=', f1_score(y_true, y_pred, pos_label=1))
