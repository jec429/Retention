import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import functools
import tensorflow.feature_column as fc
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE


X_merged = pd.read_pickle("./data_files/merged_Brazil_combined_x_numeric_new.pkl")

raw_df = X_merged[(X_merged['Report_Year'] < 2018) & (X_merged['Working_Country'] == 37)]
raw_df = raw_df.drop(['Report_Year', 'Working_Country', 'WWID'], axis=1)

# Use a utility from sklearn to split and shuffle our dataset.
train_df, test_df = train_test_split(raw_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('Status'))
val_labels = np.array(val_df.pop('Status'))
test_labels = np.array(test_df.pop('Status'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

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


def make_model():
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu',
                           input_shape=(train_features.shape[-1],)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
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
        keras.metrics.AUC(name='auc')
    ]

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=metrics)

    return model


model = make_model()

EPOCHS = 10
BATCH_SIZE = 2048

# history = model.fit(
#     train_features,
#     train_labels,
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=(val_features, val_labels))

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
# results = model.evaluate(test_features, test_labels)
# for name, value in zip(model.metrics_names, results):
#     print(name, ': ', value)
#
# predicted_labels = model.predict(test_features)
# cm = confusion_matrix(test_labels, np.round(predicted_labels))
#
# plt.matshow(cm, alpha=0)
# plt.title('Confusion matrix')
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
#
# for (i, j), z in np.ndenumerate(cm):
#     plt.text(j, i, str(z), ha='center', va='center')
#
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

# weighted_model = make_model()
#
# weighted_history = weighted_model.fit(
#     train_features,
#     train_labels,
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=(val_features, val_labels),
#     class_weight=class_weight)
#
# weighted_results = weighted_model.evaluate(test_features, test_labels)
# for name, value in zip(weighted_model.metrics_names, weighted_results):
#     print(name, ': ', value)

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
    validation_data=(val_features, val_labels))

resampled_results = resampled_model.evaluate(test_features, test_labels)
for name, value in zip(resampled_model.metrics_names, resampled_results):
    print(name, ': ', value)


X_merged = pd.read_pickle("./data_files/merged_Brazil_combined_x_numeric_new.pkl")
new_test_df = X_merged[(X_merged['Report_Year'] == 2018) & (X_merged['Working_Country'] == 37)]
new_test_df = new_test_df.drop(['Report_Year', 'Working_Country', 'WWID'], axis=1)

new_test_labels = np.array(new_test_df.pop('Status'))
new_test_features = np.array(new_test_df)

new_test_features[new_test_features == np.inf] = 999
new_test_features = scaler.transform(new_test_features)

print('New test features:')
resampled_results = resampled_model.evaluate(new_test_features, new_test_labels)
for name, value in zip(resampled_model.metrics_names, resampled_results):
    print(name, ': ', value)

plt.title('Accuracy')
plt.plot(epochs,  resampled_history.history['accuracy'], color='blue', label='Train')
plt.plot(epochs, resampled_history.history['val_accuracy'], color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

_ = plt.figure()
plt.title('Loss')
plt.plot(epochs, resampled_history.history['loss'], color='blue', label='Train')
plt.plot(epochs, resampled_history.history['val_loss'], color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

_ = plt.figure()
plt.title('False Negatives')
plt.plot(epochs, resampled_history.history['fn'], color='blue', label='Train')
plt.plot(epochs, resampled_history.history['val_fn'], color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('False Negatives')
plt.legend()

_ = plt.figure()
plt.title('True Positives')
plt.plot(epochs, resampled_history.history['tp'], color='blue', label='Train')
plt.plot(epochs, resampled_history.history['val_tp'], color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('True Positives')
plt.legend()

results = resampled_model.evaluate(new_test_features, new_test_labels)
for name, value in zip(resampled_model.metrics_names, results):
    print(name, ': ', value)

predicted_labels = resampled_model.predict(new_test_features)
cm = confusion_matrix(new_test_labels, np.round(predicted_labels))

plt.matshow(cm, alpha=0)
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

for (i, j), z in np.ndenumerate(cm):
    plt.text(j, i, str(z), ha='center', va='center')

resampled_model.save('my_model.h5')

plt.show()