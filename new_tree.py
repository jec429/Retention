import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from retention_utils import get_data, get_ourvoice, get_2019_data, get_current_data
import pickle


def evaluate(model, test_feat, test_lab):
    print('L=', test_lab)
    predictions = model.predict(test_feat)

    errors = abs(predictions - test_lab)

    mape = 100 * np.mean(errors / test_lab)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


region = 'SEA'
train_features, \
    train_labels, \
    test_features, \
    test_labels, \
    new_test_features, \
    new_test_labels, \
    features = get_data(region)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=1000, stop=2000, num=2)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num=3)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': [2000],  # n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               # 'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
               'bootstrap': bootstrap}
print(random_grid)

# best_values = {'n_estimators': [1000], 'min_samples_split': [2], 'min_samples_leaf': [1],
# 'max_features': ['auto'], 'max_depth': [50], 'bootstrap': [False]}

# Use the random grid to search for best hyper parameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, cv=3, verbose=2,
                               n_jobs=-1, n_iter=10,
                               scoring='precision_weighted'
                               # scoring='f1_weighted'
                               )
# Fit the random search model
# rf_random.fit(train_features, train_labels)

# print(rf_random.best_params_)

base_model = RandomForestClassifier(n_estimators=10, random_state=42)
base_model.fit(train_features, train_labels)

base_accuracy = evaluate(base_model, test_features, test_labels)
print(base_accuracy)
# best_random = rf_random.best_estimator_
best_random = base_model
random_accuracy = evaluate(best_random, test_features, test_labels)
print(random_accuracy)

print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))

importances = best_random.feature_importances_
std = np.std([tree.feature_importances_ for tree in best_random.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(train_features.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

y_true, y_pred = new_test_labels, best_random.predict_proba(new_test_features)
# y_true2, y_pred2 = y_resigned_new3, best_random.predict_proba(X_resigned_new3)
# print(y_pred2)
# print(list(y_true2))

print(len(y_pred), len(new_test_wwids))
pickle_name = 'parrot.pkl'
mylist = [new_test_wwids, predicted_labels, new_test_labels]
if SEA:
    pickle_name = 'parrot_fixed_sea.pkl'
elif CHINA:
    print(len(new_test_labels), sum(new_test_labels))
    pickle_name = 'parrot_fixed_china.pkl'
with open(pickle_name, 'wb') as f:
    pickle.dump(mylist, f)

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
print(accuracy_score(y_true, best_random.predict(new_test_features)))
# print(accuracy_score(y_true2, best_random.predict(X_resigned_new3)))

# print(list(base_model.predict(X_resigned_new3)))
# print(sum(base_model.predict(X_resigned_new3)))

# print(roc_auc_score(np.concatenate((y_true, y_true2), axis=0), np.concatenate((y_pred[:, 1], y_pred2[:, 1]),
#                                                                              axis=0)))

print(roc_auc_score(y_true, (y_pred[:, 1])))

# print(confusion_matrix(np.concatenate((y_true, y_true2), axis=0),
#                       np.concatenate((best_random.predict(X_resigned_new2), best_random.predict(X_resigned_new3)),
#                                      axis=0)))

print(confusion_matrix(y_true, best_random.predict(new_test_features)))

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


y_true, y_pred = new_test_labels, best_random.predict(new_test_features)
print('Results on the test set:')
print(classification_report(y_true, y_pred, target_names=['Active', 'Resigned']))

y_true, y_pred = new_test_labels, best_random.predict_proba(new_test_features)
# print('Results on the test set:')
# print(classification_report(y_true, y_pred, target_names=['Active', 'Resigned']))


active = [x[1] for x, y in zip(y_pred, y_true) if y == -1]
resigned = [x[1] for x, y in zip(y_pred, y_true) if y == 1]
xs = [i/10 for i in range(1, 10)]
ps = []
rs = []
fs = []
for i in xs:
    a1 = [x for x in active if x > i]
    a2 = [x for x in active if x < i]
    r1 = [x for x in resigned if x > i]
    r2 = [x for x in resigned if x < i]
    tp = len(r1)
    fp = len(a1)
    tn = len(a2)
    fn = len(r2)
    print('x=%f, tp=%d, fp=%d, tn=%d, fn=%d' % (i, tp, fp, tn, fn))
    if tp+fp > 0:
        precision = tp/(tp+fp)
    else:
        precision = -0.01
    if tp+fn > 0:
        recall = tp/(tp+fn)
    else:
        recall = -0.01
    ps.append(precision)
    rs.append(recall)
    if precision + recall > 0:
        fs.append(2*precision*recall/(precision + recall))
    else:
        fs.append(-0.01)

print(xs)
print(ps)
print(rs)
print(fs)
x = np.linspace(0, 10)
x = x/10
y = x/np.inf + 0.5
_ = plt.figure()
plt.plot(np.multiply(xs, 100), np.multiply(ps, 100), color='red', label='Precision')
plt.plot(np.multiply(xs, 100), np.multiply(rs, 100), color='blue', label='Recall')
plt.plot(np.multiply(xs, 100), np.multiply(fs, 100), color='green', label='F1 Score')
plt.plot(np.multiply(x, 100), y*100, color='black', linestyle=':')
plt.xlabel('Probability Threshold [%]')
plt.ylabel('Percentage [%]')
plt.legend()
# plt.show()

if 'OURVOICE' in region:
    test_features_2017, test_features_2019, id_2017, id_2019 = get_ourvoice()
    predicted_labels_2017 = best_random.predict_proba(test_features_2017)
    pickle_name = 'parrot_tree_ourvoice_2017.pkl'
    with open(pickle_name, 'wb') as f:
        pickle.dump([id_2017, predicted_labels_2017], f)

    predicted_labels_2019 = best_random.predict_proba(test_features_2019)
    pickle_name = 'parrot_tree_ourvoice_2019.pkl'
    with open(pickle_name, 'wb') as f:
        pickle.dump([id_2019, predicted_labels_2019], f)

if 'CHINA' not in region and 'SEA' not in region:
    plt.show()

new_test_features, new_test_wwids, new_test_labels2 = get_2019_data(region)

predicted_labels = best_random.predict_proba(new_test_features)
print(predicted_labels)
mylist = [new_test_wwids, predicted_labels, new_test_labels2]
print(len(new_test_labels), sum(new_test_labels2))
if 'CHINA' in region:
    pickle_name = 'parrot_china_fixed_2019.pkl'
elif 'SEA' in region:
    pickle_name = 'parrot_sea_fixed_2019.pkl'
with open(pickle_name, 'wb') as f:
    pickle.dump(mylist, f)

y_true, y_pred = new_test_labels2, [x[0] for x in predicted_labels]

active = [x for x, y in zip(y_pred, y_true) if y == 0]
resigned = [x for x, y in zip(y_pred, y_true) if y == 1]

# fpr, tpr, _ = roc_curve(y_true, y_pred)
# plt.clf()
# plt.plot(fpr, tpr)
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.title('ROC curve')
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

# cm = confusion_matrix(new_test_labels2, np.round(predicted_labels))
cm = confusion_matrix(new_test_labels2, [[1] if x[0] > 0.2 else [0] for x in predicted_labels])

plt.matshow(cm, alpha=0)
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

for (i, j), z in np.ndenumerate(cm):
    plt.text(j, i, str(z), ha='center', va='center')

xs = [i/10 for i in range(1, 10)]
ps = []
rs = []
fs = []
for i in xs:
    a1 = [x for x in active if x > i]
    a2 = [x for x in active if x < i]
    r1 = [x for x in resigned if x > i]
    r2 = [x for x in resigned if x < i]
    tp = len(r1)
    fp = len(a1)
    tn = len(a2)
    fn = len(r2)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    ps.append(precision)
    rs.append(recall)
    if precision + recall > 0:
        fs.append(2 * precision * recall / (precision + recall))
    else:
        fs.append(0)

print(xs)
print(ps)
print(rs)
print(fs)
x = np.linspace(0, 10)
x = x/10
y = x/np.inf + 0.5
_ = plt.figure()
plt.plot(np.multiply(xs, 100), np.multiply(ps, 100), color='red', label='Precision')
plt.plot(np.multiply(xs, 100), np.multiply(rs, 100), color='blue', label='Recall')
plt.plot(np.multiply(xs, 100), np.multiply(fs, 100), color='green', label='F1 Score')
plt.plot(np.multiply(x, 100), y*100, color='black', linestyle=':')
plt.xlabel('Probability Threshold [%]')
plt.ylabel('Percentage [%]')
plt.legend()

new_test_features, new_test_wwids = get_current_data(region)

predicted_labels = best_random.predict_proba(new_test_features)
mylist = [new_test_wwids, predicted_labels]
if 'CHINA' in region:
    pickle_name = 'parrot_china_current_2019.pkl'
elif 'SEA' in region:
    pickle_name = 'parrot_sea_current_2019.pkl'
with open(pickle_name, 'wb') as f:
    pickle.dump(mylist, f)

plt.show()
