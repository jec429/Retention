import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

METRIC_ACCURACY = 'accuracy'


def train_test_model(hparams):
    model = tf.keras.models.Sequential([
        #tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        #tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(x_train, y_train, epochs=10)  # Run with 1 epoch to speed things up for demo purposes
    _, accuracy = model.evaluate(x_test, y_test)
    print('calc=', accuracy)
    return accuracy


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


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

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)


HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32, 64, 128, 256]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.4))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            hparams = {
                HP_NUM_UNITS: num_units,
                HP_DROPOUT: dropout_rate,
                HP_OPTIMIZER: optimizer,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run('logs/hparam_tuning/' + run_name, hparams)
            session_num += 1