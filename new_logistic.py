import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from retention_utils import get_data


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


train_features, \
    train_labels, \
    test_features, \
    test_labels, \
    new_test_features, \
    new_test_labels, \
    features = get_data('OURVOICE')

grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}  # l1 lasso l2 ridge
grid = {"penalty": ["l1"]}  # l1 lasso l2 ridge
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, grid, cv=10)
logreg_cv.fit(train_features, train_labels)

# logreg_cv = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)

# print(estimator.score(X_test, y_test))

# Best parameter set
print('Best parameters found:\n', logreg_cv.best_params_)

# All results
means = logreg_cv.cv_results_['mean_test_score']
stds = logreg_cv.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, logreg_cv.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

best_grid = logreg_cv.best_estimator_
print('Coefficients')


def getKey(item):
    return item[1]


l = [[a, abs(b)] for a, b in zip(features, np.std(train_features, 0)*best_grid.coef_[0])]
l = sorted(l, key=getKey, reverse=True)

# The estimated coefficients will all be around 1:
for a in l:
    print(a[0], a[1])

# Those values, however, will show that the second parameter
# is more influential
# print(np.std(X_train, 0)*best_grid.coef_)

if True:
    y_true, y_pred = new_test_labels, best_grid.predict_proba(new_test_features)
    #y_true2, y_pred2 = y_resigned_new3, best_random.predict_proba(X_resigned_new3)
    #print(y_pred2)
    #print(list(y_true2))

    # y_true = [1 for yy in y_resigned_new2]
    # y_true2 = [-1 for yy in y_resigned_new3]

    active = [x[1] for x, y in zip(y_pred, y_true) if y == -1]
    resigned = [x[1] for x, y in zip(y_pred, y_true)if y == 1]
    # active = [x[1] for x, y in zip(y_pred2, y_true2)]
    # resigned = [x[1] for x, y in zip(y_pred, y_true)]

    xs = [i / 10 for i in range(1, 10)]
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
        if tp+fp == 0:
            fp = 1
        print('x=%f, tp=%d, fp=%d, tn=%d, fn=%d' % (i, tp, fp, tn, fn))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        ps.append(precision)
        rs.append(recall)
        if precision + recall == 0:
            precision = 0.1
        fs.append(2 * precision * recall / (precision + recall))

    print(xs)
    print(ps)
    print(rs)
    print(fs)
    x = np.linspace(0, 10)
    x = x / 10
    y = x / np.inf + 0.5
    _ = plt.figure()
    plt.plot(np.multiply(xs, 100), np.multiply(ps, 100), color='red', label='Precision')
    plt.plot(np.multiply(xs, 100), np.multiply(rs, 100), color='blue', label='Recall')
    plt.plot(np.multiply(xs, 100), np.multiply(fs, 100), color='green', label='F1 Score')
    plt.plot(np.multiply(x, 100), y * 100, color='black', linestyle=':')
    plt.xlabel('Probability Threshold [%]')
    plt.ylabel('Percentage [%]')
    plt.legend()

    a1 = [x for x in active if x > 0.3]
    a2 = [x for x in active if x < 0.3]
    r1 = [x for x in resigned if x > 0.3]
    r2 = [x for x in resigned if x < 0.3]

    print('False positive=', len(a1))
    print('True negative=', len(a2))
    print('True positive=', len(r1))
    print('False negative=', len(r2))

    # y_true2 = [-1 for yy in y_resigned_new3]
    print(accuracy_score(y_true, best_grid.predict(new_test_features)))
    #print(accuracy_score(y_true2, best_random.predict(X_resigned_new3)))

    # print(list(base_model.predict(X_resigned_new3)))
    # print(sum(base_model.predict(X_resigned_new3)))

    #print(roc_auc_score(np.concatenate((y_true, y_true2), axis=0), np.concatenate((y_pred[:, 1], y_pred2[:, 1]),
    #                                                                              axis=0)))

    print(roc_auc_score(y_true, (y_pred[:, 1])))

    # print(confusion_matrix(np.concatenate((y_true, y_true2), axis=0),
    #                       np.concatenate((best_random.predict(X_resigned_new2), best_random.predict(X_resigned_new3)),
    #                                      axis=0)))

    print(confusion_matrix(y_true, best_grid.predict(new_test_features)))
    # plt.show()
    # fpr, tpr, _ = roc_curve(y_true, (y_pred[:, 1]))
    #
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


y_true, y_pred = new_test_labels, best_grid.predict(new_test_features)
print('Results on the test set:')
print(classification_report(y_true, y_pred, target_names=['Active', 'Resigned']))

plt.show()

# if CHINA:
#     X_merged = pd.read_pickle("./data_files/CHINA/merged_China_combined_x_numeric_newer.pkl")
# elif SEA:
#     X_merged = pd.read_pickle("./data_files/SEA/merged_Sea_combined_x_numeric_newer.pkl")
#     x_one_jnj = pd.read_csv('data_files/SEA/Sea_2018.csv', sep=',')
#     one_jnj_wwids = x_one_jnj[x_one_jnj['One JNJ Count'] == 'Yes']['WWID'].to_list()
#     print(len(one_jnj_wwids))
#     X_merged = X_merged[(X_merged['WWID'].isin(one_jnj_wwids))]
# new_test_df = X_merged[(X_merged['Report_Year'] == 2019)]
# new_test_df = new_test_df.drop(['Report_Year', 'Compensation_Range___Midpoint'], axis=1)
# new_test_df = new_test_df.drop(to_drop, axis=1)
# new_test_df.info()
# if CHINA:
#     df_res_2019 = pd.read_excel('./data_files/CHINA/ChinaData_Jan-June 2019.xlsx', sheet_name='Data')
# elif SEA:
#     df_res_2019 = pd.read_excel('./data_files/SEA/SEAdata - JantoJune2019.xlsx', sheet_name='Data')
# res_wwids = list(df_res_2019[df_res_2019['Termination_Reason'] == 'Resignation']['WWID'])
# # print('res=', res_wwids)
# new_test_wwids = np.array(new_test_df.pop('WWID'))
# new_test_labels = np.array(new_test_df.pop('Status'))
# new_test_labels2 = [1 if x in res_wwids else 0 for x in new_test_wwids]
# new_test_features = np.array(new_test_df)
# new_test_features[new_test_features == np.inf] = 999
# new_test_features = StandardScaler().fit_transform(new_test_features)
# predicted_labels = best_grid.predict_proba(new_test_features)
# mylist = [new_test_wwids, predicted_labels, new_test_labels2]
# print(len(new_test_labels), sum(new_test_labels2))
#
#
# y_true, y_pred = new_test_labels2, [x[0] for x in predicted_labels]
#
# active = [x for x, y in zip(y_pred, y_true) if y == 0]
# resigned = [x for x, y in zip(y_pred, y_true) if y == 1]
#
# # fpr, tpr, _ = roc_curve(y_true, y_pred)
# # plt.clf()
# # plt.plot(fpr, tpr)
# # plt.xlabel('FPR')
# # plt.ylabel('TPR')
# # plt.title('ROC curve')
# # plt.show()
#
# # sys.exit()
#
# fig, ax = plt.subplots()
# ax.hist(active, 40, density=1, label='Active', alpha=0.5, color="blue")
# ax.hist(active, bins=40, density=True, histtype='step', cumulative=1,
#         label='')
# # ax.hist(resigned, 40, density=1, label='Resigned', alpha=0.5, color="red")
# plt.xlim(0.0, 1.0)
# ax.legend(loc='best')
# plt.xlabel("Resignation Probability")
#
# fig, ax = plt.subplots()
# # ax.hist(active, 40, density=1, label='Active', alpha=0.5, color="blue")
# ax.hist(resigned, 40, density=1, label='Resigned', alpha=0.5, color="red")
# ax.hist(resigned, bins=40, density=True, histtype='step', cumulative=-1,
#         label='')
# plt.xlim(0.0, 1.0)
# ax.legend(loc='best')
# plt.xlabel("Resignation Probability")
#
# # cm = confusion_matrix(new_test_labels2, np.round(predicted_labels))
# cm = confusion_matrix(new_test_labels2, [[1] if x[0] > 0.2 else [0] for x in predicted_labels])
#
# plt.matshow(cm, alpha=0)
# plt.title('Confusion matrix')
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
#
# for (i, j), z in np.ndenumerate(cm):
#     plt.text(j, i, str(z), ha='center', va='center')
#
# xs = [i/10 for i in range(1, 10)]
# ps = []
# rs = []
# fs = []
# for i in xs:
#     a1 = [x for x in active if x > i]
#     a2 = [x for x in active if x < i]
#     r1 = [x for x in resigned if x > i]
#     r2 = [x for x in resigned if x < i]
#     tp = len(r1)
#     fp = len(a1)
#     tn = len(a2)
#     fn = len(r2)
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     ps.append(precision)
#     rs.append(recall)
#     if precision + recall > 0:
#         fs.append(2 * precision * recall / (precision + recall))
#     else:
#         fs.append(0)
#
# print(xs)
# print(ps)
# print(rs)
# print(fs)
# x = np.linspace(0, 10)
# x = x/10
# y = x/np.inf + 0.5
# _ = plt.figure()
# plt.plot(np.multiply(xs, 100), np.multiply(ps, 100), color='red', label='Precision')
# plt.plot(np.multiply(xs, 100), np.multiply(rs, 100), color='blue', label='Recall')
# plt.plot(np.multiply(xs, 100), np.multiply(fs, 100), color='green', label='F1 Score')
# plt.plot(np.multiply(x, 100), y*100, color='black', linestyle=':')
# plt.xlabel('Probability Threshold [%]')
# plt.ylabel('Percentage [%]')
# plt.legend()
#
# plt.show()
