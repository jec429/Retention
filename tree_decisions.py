import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=0)

print('y_train=', y_train)
print('y_test=', y_test)

y_train = np.where(y_train==0, -1, y_train)
y_test = np.where(y_test==0, -1, y_test)

print('y_train=', y_train)
print('y_test=', y_test)

#sys.exit()

estimator = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)

estimator.fit(X_train, y_train)

# The decision estimator has an attribute called tree_  which stores the entire
# tree structure and allows access to low level attributes. The binary tree
# tree_ is represented as a number of parallel arrays. The i-th element of each
# array holds information about the node `i`. Node 0 is the tree's root. NOTE:
# Some of the arrays only apply to either leaves or split nodes, resp. In this
# case the values of nodes of the other type are arbitrary!
#
# Among those arrays, we have:
#   - left_child, id of the left child of the node
#   - right_child, id of the right child of the node
#   - feature, feature used for splitting the node
#   - threshold, threshold value at the node
#

# Using those arrays, we can parse the tree structure:

#n_nodes = estimator.tree_.node_count
n_nodes_ = [t.tree_.node_count for t in estimator.estimators_]
children_left_ = [t.tree_.children_left for t in estimator.estimators_]
children_right_ = [t.tree_.children_right for t in estimator.estimators_]
feature_ = [t.tree_.feature for t in estimator.estimators_]
threshold_ = [t.tree_.threshold for t in estimator.estimators_]


def explore_tree(estimator, n_nodes, children_left, children_right, feature, threshold,
                suffix='', print_tree=False, sample_id=0, feature_names=None):

    if not feature_names:
        feature_names = feature


    assert len(feature_names) == X.shape[1], "The feature names do not match the number of features."
    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes"
          % n_nodes)
    if print_tree:
        print("Tree structure: \n")
        for i in range(n_nodes):
            if is_leaves[i]:
                print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
            else:
                print("%snode=%s test node: go to node %s if %s <= %s else to "
                      "node %s."
                      % (node_depth[i] * "\t",
                         i,
                         children_left[i],
                         feature_names[i],
                         threshold[i],
                         children_right[i],
                         ))
            print("\n")
        print()

    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    node_indicator = estimator.decision_path(X_test)

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = estimator.apply(X_test)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.

    #sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    print(X_test[sample_id, :])

    print('Rules used to predict sample %s: ' % sample_id)
    for node_id in node_index:
        # tabulation = " "*node_depth[node_id] #-> makes tabulation of each level of the tree
        tabulation = ""
        if leave_id[sample_id] == node_id:
            #print("%s==> Predicted leaf index \n" % (tabulation))
            continue

        if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("%sdecision id node %s : (X_test[%s, '%s'] (= %s) %s %s)"
              % (tabulation,
                 node_id,
                 sample_id,
                 feature_names[feature[node_id]],
                 X_test[sample_id, feature[node_id]],
                 threshold_sign,
                 threshold[node_id]))
    print("%sPrediction for sample %d: %s"%(tabulation,
                                          sample_id,
                                          estimator.predict(X_test)[sample_id]))

    # For a group of samples, we have the following common node.
    sample_ids = [sample_id, 2]
    common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                    len(sample_ids))

    common_node_id = np.arange(n_nodes)[common_nodes]

    print("\nThe following samples %s share the node %s in the tree"
          % (sample_ids, common_node_id))
    print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))

    for sample_id_ in sample_ids:
        print("Prediction for sample %d: %s"%(sample_id_,
                                          estimator.predict(X_test)[sample_id_]))


print(list(cols))

for i, e in enumerate(estimator.estimators_):
    print("Tree %d\n" % i)
    explore_tree(estimator.estimators_[i], n_nodes_[i], children_left_[i],
                 children_right_[i], feature_[i], threshold_[i],
                 suffix=i, sample_id=0, feature_names=list(cols), print_tree=True)
    print('\n'*2)


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)
# Fit the random search model
#rf_random.fit(X_train, y_train)

#print(rf_random.best_params_)


def evaluate(model, test_features, test_labels):
    print('L=', test_labels)
    predictions = model.predict(test_features)
    #predictions = predictions.replace(0, -1)

    errors = abs(predictions - test_labels)

    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


base_model = RandomForestClassifier(n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_features='auto',
                            max_depth=50, bootstrap=False)
base_model.fit(X_train, y_train)

y_true, y_pred = y_test, base_model.predict_proba(X_test)

print(y_pred)


active = [x[1] for x, y in zip(y_pred, y_true) if y == -1]
resigned = [x[1] for x, y in zip(y_pred, y_true) if y == 1]

fig, ax = plt.subplots()
ax.hist(active, 40, density=1, label='Active', alpha=0.5, color="blue")
ax.hist(resigned, 40, density=1, label='Resigned', alpha=0.5, color="red")
plt.xlim(0.0,1.0)
ax.legend(loc='best')
plt.xlabel("Resignation Probability")

plt.show()

sys.exit()

base_accuracy = evaluate(base_model, X_test, y_test)
print(base_accuracy)
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)
print(random_accuracy)

print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))

from sklearn.metrics import classification_report

y_true, y_pred = y_test, best_random.predict(X_test)
print('Results on the test set:')
print(classification_report(y_true, y_pred, target_names=['Active', 'Resigned']))


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 40, 50, 100],
    'max_features': [2, 3, 'auto'],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [2, 4, 6, 8, 10],
    'n_estimators': [200, 400, 600, 800, 1000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_test, y_test)

print('Improvement of {:0.2f}%.'.format(100 * (grid_accuracy - base_accuracy) / base_accuracy))

y_true, y_pred = y_test, best_grid.predict(X_test)
print('Results on the test set:')
print(classification_report(y_true, y_pred, target_names=['Active', 'Resigned']))
