import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


X = pd.read_pickle("./data_x_numeric.pkl")
Y = pd.read_pickle("./data_y.pkl")
X['Status'] = Y['Status']
X = X.sample(frac=1).reset_index(drop=True)
#X = X[:-1150]
X_hold = X[-1150:]
print(X_hold.shape[0])
X_hold = X_hold[X_hold['Status'] != True]
print(X_hold.shape[0])
# sys.exit()
X_resigned = X[X['Status'] == True]
X_resigned.info()

X_resigned_new = pd.read_excel("./Brazil2019JantoJunVolTerms.xlsx")
X_resigned_new.info()
X_resigned_new = X_resigned_new[X_resigned_new['Termination_Reason'] == 'Resignation']
X_resigned_new.info()

resigs = X_resigned_new.WWID.values.tolist()
print(len(resigs), resigs)

new_status = []
for s, w in zip(X['Status'], X['WWID']):
    if w in resigs:
        new_status.append(True)
    else:
        new_status.append(s)

#X['Status'] = new_status

X_resigned_new2 = X[X.WWID.isin(resigs)]
X_resigned_new3 = X[(~X.WWID.isin(resigs))]

#X_resigned_new2 = X_hold[X_hold.WWID.isin(resigs)]
#X_resigned_new3 = X_hold[(~X_hold.WWID.isin(resigs))]

X_resigned_new2.info()

y = X['Status']
print(len(y), sum(y))

X = X.drop(['Status'], axis=1)
X = X.drop(['WWID'], axis=1)

y_resigned = X_resigned['Status']
X_resigned = X_resigned.drop(['Status'], axis=1)
X_resigned = X_resigned.drop(['WWID'], axis=1)

y_resigned_new2 = X_resigned_new2['Status']
X_resigned_new2 = X_resigned_new2.drop(['Status'], axis=1)
X_resigned_new2 = X_resigned_new2.drop(['WWID'], axis=1)

y_resigned_new3 = X_resigned_new3['Status']
X_resigned_new3 = X_resigned_new3.drop(['Status'], axis=1)
X_resigned_new3 = X_resigned_new3.drop(['WWID'], axis=1)

cols = X.columns
X = np.array(X.values)
X_resigned = np.array(X_resigned.values)
y = np.array(y.values.astype(int))

X_resigned_new2 = np.array(X_resigned_new2.values)
y_resigned_new2 = np.array(y_resigned_new2.values.astype(int))

X_resigned_new3 = np.array(X_resigned_new3.values)
y_resigned_new3 = np.array(y_resigned_new3.values.astype(int))

X = StandardScaler().fit_transform(X)
X_resigned = StandardScaler().fit_transform(X_resigned)
X_resigned_new2 = StandardScaler().fit_transform(X_resigned_new2)
X_resigned_new3 = StandardScaler().fit_transform(X_resigned_new3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=0)

print('y_train=', y_train)
print('y_test=', y_test)

y_train = np.where(y_train == 0, -1, y_train)
y_test = np.where(y_test == 0, -1, y_test)
y_resigned_new2 = np.where(y_resigned_new2 == 0, -1, y_resigned_new2)
y_resigned_new3 = np.where(y_resigned_new3 == 0, -1, y_resigned_new3)

print('y_train=', y_train)
print('y_test=', y_test)

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

if True:
    y_true, y_pred = y_resigned_new2, clf.predict_proba(X_resigned_new2)
    y_true2, y_pred2 = y_resigned_new3, clf.predict_proba(X_resigned_new3)
    #print(y_pred2)
    #print(list(y_true2))

    # y_true = [1 for yy in y_resigned_new2]
    # y_true2 = [-1 for yy in y_resigned_new3]

    active = [x[1] for x, y in zip(y_pred2, y_true2) if y == -1]
    resigned = [x[1] for x, y in zip(y_pred, y_true)]
    # active = [x[1] for x, y in zip(y_pred2, y_true2)]
    # resigned = [x[1] for x, y in zip(y_pred, y_true)]

    a1 = [x for x in active if x > 0.5]
    a2 = [x for x in active if x < 0.5]
    r1 = [x for x in resigned if x > 0.5]
    r2 = [x for x in resigned if x < 0.5]

    print('False positive=', len(a1))
    print('True negative=', len(a2))
    print('True positive=', len(r1))
    print('False negative=', len(r2))

    # y_true2 = [-1 for yy in y_resigned_new3]
    print(accuracy_score(y_true, clf.predict(X_resigned_new2) ) )
    print(accuracy_score(y_true2, clf.predict(X_resigned_new3)))

    # print(list(base_model.predict(X_resigned_new3)))
    # print(sum(base_model.predict(X_resigned_new3)))

    # print(roc_auc_score(np.concatenate((y_true, y_true2), axis=0), np.concatenate((y_pred[:, 1], y_pred2[:, 1]),
    #                                                                              axis=0)))

    print(confusion_matrix(np.concatenate((y_true, y_true2), axis=0),
                           np.concatenate((clf.predict(X_resigned_new2), clf.predict(X_resigned_new3)),
                                          axis=0)))
