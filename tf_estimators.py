import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import functools
import tensorflow.feature_column as fc


def easy_input_function(df, label_key, num_epochs, shuffle, batch_size):
    label = df[label_key]
    ds = tf.data.Dataset.from_tensor_slices((dict(df),label))

    if shuffle:
        ds = ds.shuffle(10000)

    ds = ds.batch(batch_size).repeat(num_epochs)

    return ds


df = pd.read_pickle("./data_x_numeric.pkl")
Y = pd.read_pickle("./data_y.pkl")
df['Status'] = Y['Status']

df.columns = [x.replace('=', '_') for x in df.columns]

print(df.shape[0])

msk = np.random.rand(len(df)) < 0.6
train_df = df[msk]
test_df = df[~msk]

#print(train_df.Status.value_counts())
#print(test_df.Status.value_counts())
#sys.exit()
ds = easy_input_function(df, label_key='Status', num_epochs=5, shuffle=True, batch_size=10)

for feature_batch, label_batch in ds.take(1):
  print('Some feature keys:', list(feature_batch.keys())[:5])
  print()
  print('A batch of Lengths  :', feature_batch['Length_of_Service_in_Years_inclu'])
  print()
  print('A batch of Labels:', label_batch)

train_inpf = functools.partial(easy_input_function, train_df, label_key='Status', num_epochs=40,
                               shuffle=True, batch_size=64)
test_inpf = functools.partial(easy_input_function, test_df, label_key='Status', num_epochs=1,
                              shuffle=False, batch_size=64)

Length_of_Service_in_Years_inclu = fc.numeric_column('Length_of_Service_in_Years_inclu')

num_features = []
for c in train_df.columns:
    if 'Status' in c:
        continue
    if 'Function' in c:
        continue
    num_features.append(fc.numeric_column(c))

print(num_features)
print(Length_of_Service_in_Years_inclu)
#sys.exit()


model = tf.estimator.LinearClassifier(
    feature_columns=num_features,
    optimizer=tf.keras.optimizers.Ftrl(learning_rate=0.1))

model.train(train_inpf)
results = model.evaluate(test_inpf)

for key, value in sorted(results.items()):
    print('%s: %0.2f' % (key, value))

model_l1 = tf.estimator.LinearClassifier(
    feature_columns=num_features,
    optimizer=tf.keras.optimizers.Ftrl(
        learning_rate=0.1,
        l1_regularization_strength=10.0,
        l2_regularization_strength=0.0))

#model_l1.train(train_inpf)
#results = model_l1.evaluate(test_inpf)

#for key, value in sorted(results.items()):
#    print('%s: %0.2f' % (key, value))

model_l2 = tf.estimator.LinearClassifier(
    feature_columns=num_features,
    optimizer=tf.keras.optimizers.Ftrl(
        learning_rate=0.1,
        l1_regularization_strength=0.0,
        l2_regularization_strength=10.0))

#model_l2.train(train_inpf)
#results = model_l2.evaluate(test_inpf)

#for key in sorted(results):
#    print('%s: %0.2f' % (key, results[key]))

n_batches = 1
est = tf.estimator.BoostedTreesClassifier(num_features,
                                          n_batches_per_layer=n_batches)

# The model will stop training once the specified number of trees is built, not
# based on the number of steps.
est.train(train_inpf, max_steps=100)

# Eval.
results = est.evaluate(test_inpf)
#print('Accuracy : ', results['accuracy'])
#print('Dummy model: ', results['accuracy_baseline'])
for key, value in sorted(results.items()):
    print('%s: %0.2f' % (key, value))

from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

pred_dicts = list(est.predict(test_inpf))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

fpr, tpr, _ = roc_curve(test_df.Status, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)
plt.show()
