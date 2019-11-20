import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle
import time
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def make_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu',
                           input_shape=(train_features.shape[-1],)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(1, activation='sigmoid'),
    ])

    metrics = [
        keras.metrics.Accuracy(name='accuracy'),
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        'acc', 'binary_crossentropy'
    ]

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=metrics)

    return model


def evaluate(model, test_feat, test_lab):
    # print('L=', list(test_lab))
    predictions = model.predict(test_feat)
    predictions = np.array([1 if x == 1 else -1 for x in predictions])
    test_lab = np.array([1 if x == 1 else -1 for x in test_lab])

    errors = abs(predictions - test_lab)

    mape = 100 * np.mean(errors / test_lab)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


def getKey(item):
    return item[1]


timestamp = time.strftime("%d_%b_%Y_%H_%M_%S", time.localtime())
X_merged = pd.read_pickle("./data_files/D2Plus/merged_D2Plus_combined_fixed_x_numeric_newer.pkl")
X_merged2 = pd.DataFrame()
too_good = ['Report_Year', 'WWID', 'Termination_Category',
            # 'MRC_Code__IA__Host_All_Other__Pr',
            # 'Legal_Entity_Code__IA__Host_All_',
            'Employee_Direct_Reports'
            ]
for c in too_good:
    X_merged2[c] = X_merged[c]

raw_df = X_merged[(X_merged['Report_Year'] < 2018)]
raw_df = raw_df.drop(['Report_Year', 'WWID'], axis=1)

print(raw_df.columns.to_list())
to_drop = []

# x for x in raw_df.columns.to_list() if 'Location' in x or 'Function' in x or 'Highest_Degree' in x]
# to_drop.append('Bonus_Flag')
# to_drop.append('Merit_Flag')
# to_drop.append('Bonus_Merit_Flag')
# to_drop.append('Promotion')
# to_drop.append('Demotion')
# to_drop.append('Lateral')

raw_df = raw_df.drop(to_drop, axis=1)

# Use a utility from sklearn to split and shuffle our dataset.
train_df, test_df = train_test_split(raw_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# test_df = test_df[test_df['Working_Country_Fixed'] == 11]

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('Termination_Category'))
val_labels = np.array(val_df.pop('Termination_Category'))
test_labels = np.array(test_df.pop('Termination_Category'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)
features = train_df.columns.to_list()

train_features[train_features == np.inf] = 999
val_features[val_features == np.inf] = 999
test_features[test_features == np.inf] = 999
# Normalize the input features using the sklearn StandardScaler.
# This will set the mean to 0 and standard deviation to 1.
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)

neg, pos = np.bincount(train_labels)
total = neg + pos
print('{} positive samples out of {} training samples ({:.2f}% of total)'.format(
    pos, total, 100 * pos / total))

X_merged = pd.read_pickle("./data_files/D2Plus/merged_D2Plus_combined_fixed_x_numeric_newer.pkl")
X_merged2 = pd.DataFrame()
for c in too_good:
    X_merged2[c] = X_merged[c]

new_test_df = X_merged[(X_merged['Report_Year'] == 2018)]
new_test_df = new_test_df.drop(['Report_Year'], axis=1)
new_test_df = new_test_df.drop(to_drop, axis=1)
# new_test_df.info()
new_test_wwids = np.array(new_test_df.pop('WWID'))
new_test_labels = np.array(new_test_df.pop('Termination_Category'))
new_test_features = np.array(new_test_df)

neg, pos = np.bincount(new_test_labels)
total = neg + pos
print('{} positive samples out of {} testing samples ({:.2f}% of total)'.format(
    pos, total, 100 * pos / total))

new_test_features[new_test_features == np.inf] = 999
new_test_features = scaler.transform(new_test_features)

# 2019
X_merged = pd.read_pickle("./data_files/D2Plus/merged_D2Plus_combined_fixed_x_numeric_newer.pkl")
X_merged2 = pd.DataFrame()
for c in too_good:
    X_merged2[c] = X_merged[c]

new_2019_test_df = X_merged[(X_merged['Report_Year'] == 2019)]
new_2019_test_df = new_2019_test_df.drop(['Report_Year'], axis=1)
new_2019_test_df = new_2019_test_df.drop(to_drop, axis=1)
# new_test_df.info()
new_2019_test_wwids = np.array(new_2019_test_df.pop('WWID'))
new_2019_test_labels = np.array(new_2019_test_df.pop('Termination_Category'))
new_2019_test_features = np.array(new_2019_test_df)

new_2019_test_features[new_2019_test_features == np.inf] = 999
new_2019_test_features = scaler.transform(new_2019_test_features)

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

base_accuracy = evaluate(base_model, new_test_features, new_test_labels)
print(base_accuracy)
# best_random = rf_random.best_estimator_
best_random = base_model
random_accuracy = evaluate(best_random, new_test_features, new_test_labels)
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

grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}  # l1 lasso l2 ridge
grid = {"penalty": ["l1"]}  # l1 lasso l2 ridge
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, grid, cv=10)
logreg_cv.fit(train_features, train_labels)
# All results
means = logreg_cv.cv_results_['mean_test_score']
stds = logreg_cv.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, logreg_cv.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

best_grid = logreg_cv.best_estimator_
print('Coefficients')

li = [[a, abs(b)] for a, b in zip(features, np.std(train_features, 0)*best_grid.coef_[0])]
li = sorted(li, key=getKey, reverse=True)

# The estimated coefficients will all be around 1:
for a in li:
    print(a[0], a[1])

y_true, y_pred = new_test_labels, best_random.predict_proba(new_test_features)

active = [x[1] for x, y in zip(y_pred, y_true) if y == 0]
resigned = [x[1] for x, y in zip(y_pred, y_true)if y == 1]
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
    print('x=%f, tp=%d, fp=%d, tn=%d, fn=%d' %(i, tp, fp, tn, fn))
    precision = tp/(tp+fp) if tp+fp > 0 else 0
    recall = tp/(tp+fn) if tp+fn > 0 else 0
    ps.append(precision)
    rs.append(recall)
    if precision+recall > 0:
        fs.append(2*precision*recall/(precision + recall))
    else:
        fs.append(0)

print(xs)
print(ps)
print(rs)
print(fs)

# sys.exit()

model = make_model()

EPOCHS = 20
BATCH_SIZE = 512

history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=0,
    validation_data=(val_features, val_labels))

epochs = range(EPOCHS)

# plt.title('Accuracy')
# plt.plot(epochs,  history.history['accuracy'], color='blue', label='Train')
# plt.plot(epochs, history.history['val_accuracy'], color='orange', label='Val')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
#
# _ = plt.figure()
# plt.title('Loss')
# plt.plot(epochs, history.history['loss'], color='blue', label='Train')
# plt.plot(epochs, history.history['val_loss'], color='orange', label='Val')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
#
# _ = plt.figure()
# plt.title('False Negatives')
# plt.plot(epochs, history.history['fn'], color='blue', label='Train')
# plt.plot(epochs, history.history['val_fn'], color='orange', label='Val')
# plt.xlabel('Epoch')
# plt.ylabel('False Negatives')
# plt.legend()
#
# _ = plt.figure()
# plt.title('True Positives')
# plt.plot(epochs, history.history['tp'], color='blue', label='Train')
# plt.plot(epochs, history.history['val_tp'], color='orange', label='Val')
# plt.xlabel('Epoch')
# plt.ylabel('True Positives')
# plt.legend()
#
results = model.evaluate(new_test_features, new_test_labels)
for name, value in zip(model.metrics_names, results):
    print(name, ': ', value)

predicted_labels = model.predict(new_test_features)

# print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
# print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
# print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
# print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
# print('Total Fraudulent Transactions: ', np.sum(cm[1]))


weight_for_0 = 1 / neg
weight_for_1 = 1 / pos

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2e}'.format(weight_for_0))
print('Weight for class 1: {:.2e}'.format(weight_for_1))

weighted_model = make_model()

weighted_history = weighted_model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=0,
    validation_data=(val_features, val_labels),
    class_weight=class_weight)

print('Weighted Results')
weighted_results = weighted_model.evaluate(new_test_features, new_test_labels)
for name, value in zip(weighted_model.metrics_names, weighted_results):
    print(name, ': ', value)

predicted_labels = weighted_model.predict(new_test_features)
predicted_labels_2019 = weighted_model.predict(new_2019_test_features)
cm = confusion_matrix(new_test_labels, np.round(predicted_labels))


print(len(predicted_labels), len(new_test_wwids), len([x[0] for x in predicted_labels if x[0] > 0.5]))
pickle_name = 'predictions/parrot_D2Plus_' + timestamp + '.pkl'
mylist = [new_test_wwids, predicted_labels, new_test_labels]
with open(pickle_name, 'wb') as f:
    pickle.dump(mylist, f)
f.close()

pickle_name = 'predictions/parrot_D2Plus_2019_' + timestamp + '.pkl'
mylist = [new_2019_test_wwids, predicted_labels_2019, new_2019_test_labels]
with open(pickle_name, 'wb') as f:
    pickle.dump(mylist, f)
f.close()

plt.matshow(cm, alpha=0)
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

for (i, j), z in np.ndenumerate(cm):
    plt.text(j, i, str(z), ha='center', va='center')

y_true, y_pred = new_test_labels, [x[0] for x in predicted_labels]

active = [x for x, y in zip(y_pred, y_true) if y == 0]
resigned = [x for x, y in zip(y_pred, y_true) if y == 1]
print('Active/Resigned = ', len(active), len(resigned))
print(np.max(active), np.min(active))
print(np.max(resigned), np.min(resigned))

fig, ax = plt.subplots()
ax.hist(active, 40, density=0, label='Active', alpha=0.5, color="blue")
# ax.hist(active, bins=40, density=True, histtype='step', cumulative=1, label='')
# ax.hist(resigned, 40, density=1, label='Resigned', alpha=0.5, color="red")
plt.xlim(0.0, 1.0)
ax.legend(loc='best')
plt.xlabel("Resignation Probability")

fig, ax = plt.subplots()
# ax.hist(active, 40, density=1, label='Active', alpha=0.5, color="blue")
ax.hist(resigned, 40, density=0, label='Resigned', alpha=0.5, color="red")
# ax.hist(resigned, bins=40, density=True, histtype='step', cumulative=-1, label='')
plt.xlim(0.0, 1.0)
ax.legend(loc='best')
plt.xlabel("Resignation Probability")

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
    print('x=%f, tp=%d, fp=%d, tn=%d, fn=%d' %(i, tp, fp, tn, fn))
    precision = tp/(tp+fp) if tp+fp > 0 else 0
    recall = tp/(tp+fn) if tp+fn > 0 else 0
    ps.append(precision)
    rs.append(recall)
    if precision+recall > 0:
        fs.append(2*precision*recall/(precision + recall))
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

# plt.show()

# with default args this will oversample the minority class to have an equal
# number of observations
smote = SMOTE()
res_features, res_labels = smote.fit_sample(train_features, train_labels)

res_neg, res_pos = np.bincount(res_labels)
res_total = res_neg + res_pos
print('{} positive samples out of {} training samples ({:.2f}% of total)'.format(
    res_pos, res_total, 100 * res_pos / res_total))

resampled_model = make_model()

resampled_history = resampled_model.fit(
    res_features,
    res_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=0,
    validation_data=(val_features, val_labels))

resampled_results = resampled_model.evaluate(new_test_features, new_test_labels)
for name, value in zip(resampled_model.metrics_names, resampled_results):
    print(name, ': ', value)

print('New test features:')
resampled_results = resampled_model.evaluate(new_test_features, new_test_labels)
for name, value in zip(resampled_model.metrics_names, resampled_results):
    print(name, ': ', value)

# plt.title('Accuracy')
# plt.plot(epochs,  resampled_history.history['accuracy'], color='blue', label='Train')
# plt.plot(epochs, resampled_history.history['val_accuracy'], color='orange', label='Val')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
#
# _ = plt.figure()
# plt.title('Loss')
# plt.plot(epochs, resampled_history.history['loss'], color='blue', label='Train')
# plt.plot(epochs, resampled_history.history['val_loss'], color='orange', label='Val')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
#
# _ = plt.figure()
# plt.title('False Negatives')
# plt.plot(epochs, resampled_history.history['fn'], color='blue', label='Train')
# plt.xlabel('Epoch')
# plt.ylabel('False Negatives')
# plt.legend()
# _ = plt.figure()
# plt.title('False Negatives')
# plt.plot(epochs, resampled_history.history['val_fn'], color='orange', label='Val')
# plt.xlabel('Epoch')
# plt.ylabel('False Negatives')
# plt.legend()
#
# _ = plt.figure()
# plt.title('True Positives')
# plt.plot(epochs, resampled_history.history['tp'], color='blue', label='Train')
# plt.xlabel('Epoch')
# plt.ylabel('True Positives')
# plt.legend()
# _ = plt.figure()
# plt.title('True Positives')
# plt.plot(epochs, resampled_history.history['val_tp'], color='orange', label='Val')
# plt.xlabel('Epoch')
# plt.ylabel('True Positives')
# plt.legend()

results = resampled_model.evaluate(new_test_features, new_test_labels)
for name, value in zip(resampled_model.metrics_names, results):
    print(name, ': ', value)

predicted_labels = resampled_model.predict(new_test_features)

y_true, y_pred = new_test_labels, [x[0] for x in predicted_labels]

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

cm = confusion_matrix(new_test_labels, np.round(predicted_labels))
cm = confusion_matrix(new_test_labels, [[1] if x[0] > 0.2 else [0] for x in predicted_labels])

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
    precision = tp / (tp + fp) if tp+fp > 0 else 0
    recall = tp / (tp + fn) if tp+fn > 0 else 0
    ps.append(precision)
    rs.append(recall)
    if precision+recall > 0:
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

plt.show()
