import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = 0.50  # step size in the mesh

names = ["Nearest Neighbors",
         "Linear SVM",
         "RBF SVM",
         #"Gaussian Process",
         "Decision Tree",
         "Random Forest",
         "Neural Net",
         "AdaBoost",
         "Naive Bayes",
         "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X = pd.read_pickle("./data_x_numeric.pkl")
Y = pd.read_pickle("./data_y.pkl")
X.info()
X = X[['Total_Base_Pay___Local', 'Length_of_Service_in_Years_inclu', 'Compa_Ratio']]
X.info()
X['Status'] = Y['Status']
X = X[X['Compa_Ratio'] > 0.1]
y = pd.Series()
y = X['Status']
X = X.drop(['Status'], axis=1)

print(X.head(10))
X = X[['Compa_Ratio', 'Length_of_Service_in_Years_inclu', 'Total_Base_Pay___Local']]
X.info()
X = np.array(X.values)
y = np.array(y.values.astype(int))
print(len(X), len(y))
#sys.exit()

linearly_separable = (X, y)

datasets = [linearly_separable, linearly_separable, linearly_separable]
#datasets = [linearly_separable]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets

for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    z_min, z_max = X[:, 2].min() - .5, X[:, 2].max() + .5

    xcol_min = []
    xcol_max = []
    for icol in range(0, 3):
        xcol_min_i, xcol_max_i = X[:, icol].min() - .5, X[:, icol].max() + .5
        xcol_min.append(xcol_min_i)
        xcol_max.append(xcol_max_i)

    #sys.exit()
    print(x_min, x_max)
    print(y_min, y_max)
    print(z_min, z_max)

    xcol_min = np.array(xcol_min)
    xcol_max = np.array(xcol_max)

    #print(x_max, x_max)
    #print(y_max, y_max)
    #print(z_max, z_max)

    print('ds count=', ds_cnt)

    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h),
                             np.arange(z_min, z_max, h))


    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")

    if ds_cnt == 0:
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

    elif ds_cnt == 1:
        # Plot the training points
        ax.scatter(X_train[:, 1], X_train[:, 2], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 1], X_test[:, 2], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')

        ax.set_xlim(yy.min(), yy.max())
        ax.set_ylim(zz.min(), zz.max())
        ax.set_xticks(())
        ax.set_yticks(())

    elif ds_cnt == 2:
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 2], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 2], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(zz.min(), zz.max())
        ax.set_xticks(())
        ax.set_yticks(())

    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        #print('Name=', name)
        #print('Classifier=', clf)
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        #print('orig=', xx, xx.shape)
        #print('Ravel=', xx.ravel(), xx.ravel().shape)
        #print('c_=', np.c_[xx.ravel(), yy.ravel(), zz.ravel()], np.c_[xx.ravel(), yy.ravel(), zz.ravel()].shape)
        #print('man=', np.array([xcol_min, xcol_max]), np.array([xcol_min, xcol_max]).shape)

        grid_array = np.array([xcol_min,
                               xcol_min + [0 if w != 1 else (xcol_max[1] - xcol_min[1]) / 2 for w in
                                           range(0, len(xcol_min))],

                               xcol_min + [0 if w != 0 else (xcol_max[0] - xcol_min[0]) / 2 for w in
                                           range(0, len(xcol_min))],

                               xcol_min + [0 if w != 0 else (xcol_max[0] - xcol_min[0]) / 2 for w in
                                           range(0, len(xcol_min))]
                                        + [0 if w != 1 else (xcol_max[1] - xcol_min[1]) / 2 for w in
                                           range(0, len(xcol_min))],

                               xcol_min + [0 if w != 0 else (xcol_max[0] - xcol_min[0]) for w in
                                           range(0, len(xcol_min))]
                                        + [0 if w != 1 else (xcol_max[1] - xcol_min[1]) / 2 for w in
                                           range(0, len(xcol_min))],

                               xcol_min + [0 if w != 0 else (xcol_max[0] - xcol_min[0]) / 2 for w in
                                           range(0, len(xcol_min))]
                                        + [0 if w != 1 else (xcol_max[1] - xcol_min[1]) for w in
                                           range(0, len(xcol_min))],

                               xcol_min + [0 if w != 1 else (xcol_max[1] - xcol_min[1]) for w in
                                           range(0, len(xcol_min))],

                               xcol_min + [0 if w != 0 else (xcol_max[0] - xcol_min[0]) for w in
                                           range(0, len(xcol_min))],

                               #xcol_min + [(xcol_max[0] - xcol_min[0]) / 2, 0, ],
                               xcol_max])

        print('array=', grid_array, grid_array.shape)
        #sys.exit()
        #print('Done fitting')
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
            #Z = clf.decision_function(grid_array)
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])[:, 1]
            #Z = clf.predict_proba(grid_array)[:, 1]

        if ds_cnt == 0:
            # Put the result into a color plot
            xx2 = xx[:, :, 0]
            yy2 = yy[:, :, 0]
            #xx2 = grid_array[:, :1].reshape(3, 3)
            #yy2 = grid_array[:, 1:2].reshape(3, 3)
            print(xx)
            print(grid_array[:, :1].reshape(3, 3))
            print(grid_array[:, 1:2].reshape(3, 3))
            print(xx.shape)

            #print(xx2)
            print(xx2.shape)
            #print(Z)
            print(Z.shape)
            #print(grid_array.shape)
            #Z = Z.reshape(3, 3)
            Z = np.array([s[:xx.shape[1]] for s in np.split(Z, xx.shape[0])]).reshape(xx2.shape)
            print(Z)
            print(Z.shape)
            ax.contourf(xx2, yy2, Z, cmap=cm, alpha=.8)
            #sys.exit()
            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())

        elif ds_cnt == 1:
            # Put the result into a color plot
            #print(xx.shape)
            xx2 = yy[:, 0, :]
            yy2 = zz[:, 0, :]
            #xx2 = grid_array[:, 1:2].reshape(3, 3)
            #yy2 = grid_array[:, 2:3].reshape(3, 3)
            #print(len(np.split(Z, xx.shape[0])))
            Z = np.array([s[:xx.shape[2]] for s in np.split(Z, xx.shape[0])]).reshape(xx2.shape)
            #Z = Z.reshape(3, 3)
            ax.contourf(xx2, yy2, Z, cmap=cm, alpha=.8)

            # Plot the training points
            ax.scatter(X_train[:, 1], X_train[:, 2], c=y_train, cmap=cm_bright,
                       edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, 1], X_test[:, 2], c=y_test, cmap=cm_bright,
                       edgecolors='k', alpha=0.6)

            ax.set_xlim(yy.min(), yy.max())
            ax.set_ylim(zz.min(), zz.max())
            ax.set_xticks(())
            ax.set_yticks(())

        elif ds_cnt == 2:
            # Put the result into a color plot
            #print(xx.shape)
            xx2 = xx[0, :, :]
            yy2 = zz[0, :, :]
            #xx2 = grid_array[:, :1].reshape(3, 3)
            #yy2 = grid_array[:, 2:3].reshape(3, 3)
            #print(len(np.split(Z, xx.shape[0])))
            Z = np.array([s[:xx.shape[2]] for s in np.split(Z, xx.shape[1])]).reshape(xx2.shape)
            #Z = Z.reshape(3, 3)
            ax.contourf(xx2, yy2, Z, cmap=cm, alpha=.8)

            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 2], c=y_train, cmap=cm_bright,
                       edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, 0], X_test[:, 2], c=y_test, cmap=cm_bright,
                       edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(zz.min(), zz.max())
            ax.set_xticks(())
            ax.set_yticks(())

        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1


plt.tight_layout()
#plt.savefig('plots/1_2.png')
#plt.close(figure)
plt.show()
