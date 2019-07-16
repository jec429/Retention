import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Nearest Neighbors",
         "Linear SVM",
         "RBF SVM",
         #"Gaussian Process",
         "Decision Tree",
         "Random Forest",
         "Neural Net",
         #"AdaBoost",
         "Naive Bayes"]#,
         #"QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma='auto', C=1, probability=True),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    MLPClassifier(alpha=1),
    #AdaBoostClassifier(),
    GaussianNB()]#,
    #QuadraticDiscriminantAnalysis()]

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
X = np.array(X.values)
X_resigned = np.array(X_resigned.values)
y = np.array(y.values.astype(int))
print(len(X), len(y))
#sys.exit()

linearly_separable = (X, y)

datasets = [linearly_separable]
i = 1

# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_resigned = StandardScaler().fit_transform(X_resigned)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    #print('X train =', X_train.shape[0])
    #print('Y train =', sum(y_train))
    #print('X test =', X_test.shape[0])
    #print('Y test =', sum(y_test))

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = 0

        X_test_r = [x for x, y in zip(X_test, y_test) if y == 1]
        #print('tr=', len(X_test_r), sum(y_test))

        print('Score=', clf.score(X_test, y_test))

        print('Test')
        for t, y_r in zip(X_test, y_test):
            if y_r == 1:
                score += clf.predict([t])[0]
            #if 'QDA' in name:
            #    print(y_r, clf.predict([t]))
        print(name, score, score / sum(y_test) * 100)

        print('Trained')
        score2 = 0
        for t, y_r in zip(X, y):
            if y_r == 1:
                score2 += clf.predict([t])[0]

        print(name, score2, score2/X_resigned.shape[0] * 100)

        Z = np.array([])
        if hasattr(clf, "predict_proba"):
            Z = clf.predict_proba(X_test)
        print(Z.shape)

        #if hasattr(clf, "decision_path"):
        #    print('PATH=', clf.decision_path(X_train[12].reshape(1, -1)))
    #if hasattr(clf, "decision_function"):
        #    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
        #else:
        #    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])[:, 1]

    for name, clf in zip(names, classifiers):
        clf.fit(X, y)
        score = 0
        print('Trained All')
        score2 = 0
        for t, y_r in zip(X, y):
            if y_r == 1:
                score2 += clf.predict([t])[0]

        print(name, score2, score2/X_resigned.shape[0] * 100)

        Z = np.array([])
        if hasattr(clf, "predict_proba"):
            Z = clf.predict_proba(X_test)
        #print(Z.shape)

        #if hasattr(clf, "decision_function"):
        #    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
        #else:
        #    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])[:, 1]
