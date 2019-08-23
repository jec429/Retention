import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
import pickle

def evaluate(model, test_features, test_labels):
    print('L=', test_labels)
    predictions = model.predict(test_features)

    errors = abs(predictions - test_labels)

    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


X_merged = pd.read_pickle("./data_files/merged_Brazil_combined_x_numeric_new.pkl")

X = X_merged[(X_merged['Report_Year'] < 2018) & (X_merged['Working_Country'] == 37)]

# X_merged = X_merged[(X_merged['Report_Year'] != 2018) & (X_merged['Working_Country'] == 37)]
# X = X_merged[(X_merged['Status']==False)][:1500]
# X_temp = X.append(X_merged[X_merged['Status']==True])
# X = X_temp

X = X.drop(['Report_Year', 'Working_Country'], axis=1)
X = X.sample(frac=1).reset_index(drop=True)

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(-999)
X = X.sample(frac=1).reset_index(drop=True)

y = X['Status']
print(len(y), sum(y))

X = X.drop(['Status'], axis=1)
X = X.drop(['WWID'], axis=1)
X = np.array(X.values)
y = np.array(y.values.astype(int))
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
# X_train, y_train = X, y

print('y_train=', y_train)
print('y_test=', y_test)

y_train = np.where(y_train == 0, -1, y_train)
y_test = np.where(y_test == 0, -1, y_test)

print('y_train=', y_train)
print('y_test=', y_test)


mlp = MLPClassifier()

parameter_space = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    'activation': ['tanh', 'relu'],
    'max_iter': [1000, 1500],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': 10.0 ** -np.arange(1, 7),
    'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'learning_rate': ['constant', 'adaptive'],
}

mlp.fit(X_train, y_train)

best_par = {'solver': ['sgd'], 'random_state': [5], 'max_iter': [1500], 'learning_rate': ['adaptive'],
            'hidden_layer_sizes': [(50, 50, 50)], 'alpha': [0.001], 'activation': ['tanh']}


# clf = RandomizedSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, scoring='balanced_accuracy', verbose=2)
# clf = RandomizedSearchCV(mlp, best_par, n_jobs=-1, cv=3, scoring='precision_weighted', verbose=2)
clf = RandomizedSearchCV(mlp, parameter_space, n_jobs=-1, cv=4, scoring='precision_weighted', verbose=2)
# clf = RandomizedSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, scoring='recall_weighted', verbose=2)
# clf = RandomizedSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, scoring='f1_weighted', verbose=2)
# clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, scoring='precision_weighted', verbose=2)
# clf = GridSearchCV(mlp, best_par, n_jobs=-1, cv=3, scoring='precision_weighted', verbose=2)
clf.fit(X_train, y_train)

# print(estimator.score(X_test, y_test))

# Best parameter set
print('Best parameters found:\n', clf.best_params_)

y_true, y_pred = y_test, clf.best_estimator_.predict(X_test)
print('Results on the test set:')
print(classification_report(y_true, y_pred, target_names=['Active', 'Resigned']))


# All results
#means = clf.cv_results_['mean_test_score']
#stds = clf.cv_results_['std_test_score']
#for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

X_merged = pd.read_pickle("./data_files/merged_Brazil_combined_x_numeric_new.pkl")

X2 = X_merged[(X_merged['Report_Year'] == 2018) & (X_merged['Working_Country'] == 37)]
X2 = X2.drop(['Report_Year', 'Working_Country'], axis=1)
X = X2.sample(frac=1).reset_index(drop=True)

X_resigned_new = pd.read_excel("./data_files/Brazil2019JantoJunVolTerms.xlsx")
X_resigned_new.info()
X_resigned_new = X_resigned_new[X_resigned_new['Termination_Reason'] == 'Resignation']
X_resigned_new.info()

resigs = X_resigned_new.WWID.values.tolist()
print('Resignations 2019:', len(resigs), resigs)

new_status = []
for s, w in zip(X['Status'], X['WWID']):
    if w in resigs:
        new_status.append(True)
    else:
        new_status.append(s)

# X['Status'] = new_status
y = X['Status']
print(len(y), sum(y))

X = X.drop(['Status'], axis=1)
X = X.drop(['WWID'], axis=1)
X = np.array(X.values)
y_resigned_new2 = np.array(y.values.astype(int))
y_resigned_new2 = np.where(y_resigned_new2 == 0, -1, y_resigned_new2)
X_resigned_new2 = StandardScaler().fit_transform(X)

best_grid = clf.best_estimator_
filename = 'finalized_model.sav'
pickle.dump(best_grid, open(filename, 'wb'))


import sys
sys.exit()

if True:
    y_true, y_pred = y_resigned_new2, best_grid.predict_proba(X_resigned_new2)
    # y_true2, y_pred2 = y_resigned_new3, best_random.predict_proba(X_resigned_new3)
    # print(y_pred2)
    # print(list(y_true2))

    # y_true = [1 for yy in y_resigned_new2]
    # y_true2 = [-1 for yy in y_resigned_new3]

    active = [x[1] for x, y in zip(y_pred, y_true) if y == -1]
    resigned = [x[1] for x, y in zip(y_pred, y_true)if y == 1]
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
    print(accuracy_score(y_true, best_grid.predict(X_resigned_new2)))
    # print(accuracy_score(y_true2, best_random.predict(X_resigned_new3)))

    # print(list(base_model.predict(X_resigned_new3)))
    # print(sum(base_model.predict(X_resigned_new3)))

    # print(roc_auc_score(np.concatenate((y_true, y_true2), axis=0), np.concatenate((y_pred[:, 1], y_pred2[:, 1]),
    #                                                                              axis=0)))

    print(roc_auc_score(y_true, (y_pred[:, 1])))

    # print(confusion_matrix(np.concatenate((y_true, y_true2), axis=0),
    #                       np.concatenate((best_random.predict(X_resigned_new2), best_random.predict(X_resigned_new3)),
    #                                      axis=0)))

    print(confusion_matrix(y_true, best_grid.predict(X_resigned_new2)))

    fpr, tpr, _ = roc_curve(y_true, (y_pred[:, 1]))

    plt.clf()
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    # plt.show()

    # sys.exit()

    fig, ax = plt.subplots()
    ax.hist(active, 40, density=1, label='Active', alpha=0.5, color="blue")
    ax.hist(active, bins=40, density=True, histtype='step', cumulative=1,
            label='')
    # ax.hist(resigned, 40, density=1, label='Resigned', alpha=0.5, color="red")
    plt.xlim(0.0, 1.0)
    ax.legend(loc='best')
    plt.xlabel("Resignation Probability")

    fig, ax = plt.subplots()
    # ax.hist(active, 40, density=1, label='Active', alpha=0.5, color="blue")
    ax.hist(resigned, 40, density=1, label='Resigned', alpha=0.5, color="red")
    ax.hist(resigned, bins=40, density=True, histtype='step', cumulative=-1,
            label='')
    plt.xlim(0.0, 1.0)
    ax.legend(loc='best')
    plt.xlabel("Resignation Probability")


y_true, y_pred = y_resigned_new2, best_grid.predict(X_resigned_new2)
print('Results on the test set:')
print(classification_report(y_true, y_pred, target_names=['Active', 'Resigned']))

plt.show()
