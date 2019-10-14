import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fname = "parrot_china_2019.pkl"
with open(fname, "rb") as fin:
    list_lists2 = pickle.load(fin)
wwids = list(list_lists2[0])
print('Resgs=', len(list_lists2[2]), sum(list_lists2[2]))
prob_tf = [[x[0], y] for x, y in zip(list_lists2[1], list_lists2[2])]

org_dict = dict(zip(wwids, prob_tf))
# print(org_dict)

df1 = pd.read_excel('./data_files/HighRiskCandidatesChina2019bySector1.xlsx', sheet_name='MDActives2018YE')
df2 = pd.read_excel('./data_files/HighRiskCandidatesChina2019bySector1.xlsx', sheet_name='PharmaActives2018YE')
df3 = pd.read_excel('./data_files/HighRiskCandidatesChina2019bySector1.xlsx', sheet_name='ConsumerActives2018YE')
df4 = pd.read_excel('./data_files/HighRiskCandidatesChina2019bySector1.xlsx', sheet_name='SupplyChainActives2018YE')

df_merged = df1.append(df2, sort=True)
df_merged = df_merged.append(df3, sort=True)
df_merged = df_merged.append(df4, sort=True)
my_dict = dict(zip(df_merged['WWID'], df_merged['Prob']))
print(len(my_dict.keys()))

active1 = []
resigned1 = []
active2 = []
resigned2 = []
for key in my_dict.keys():
    if key in org_dict.keys():
        #print(key, my_dict[key], org_dict[key][0])
        if org_dict[key][1] == 0:
            active1.append(my_dict[key])
            active2.append(org_dict[key][0])
        else:
            resigned1.append(my_dict[key])
            resigned2.append(org_dict[key][0])
    else:
        print('WWID not found ', key)

fig, ax = plt.subplots()
ax.hist(resigned1, 20, density=0, label='resigned S', alpha=0.5, color="red")
ax.hist(resigned2, 20, density=0, label='resigned J', alpha=0.5, color="orange")
ax.legend(loc='best')
fig, ax = plt.subplots()
ax.hist(active1, 20, density=0, label='active S', alpha=0.5, color="blue")
ax.hist(active2, 20, density=0, label='active J', alpha=0.5, color="green")
ax.legend(loc='best')

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
xs = [i/10 for i in range(1, 10)]
ps = []
rs = []
fs = []
for threshold in thresholds:
    print('S:')
    print('tp=', sum(x > threshold for x in resigned1))
    print('fn=', sum(x < threshold for x in resigned1))
    print('fp=', sum(x > threshold for x in active1))
    print('tn=', sum(x < threshold for x in active1))
    tp = sum(x > threshold for x in resigned1)
    fp = sum(x > threshold for x in active1)
    tn = sum(x < threshold for x in active1)
    fn = sum(x < threshold for x in resigned1)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    ps.append(precision)
    rs.append(recall)
    fs.append(2 * precision * recall / (precision + recall))
    print(precision, recall, f1)
    print('J:')
    print('tp=', sum(x > threshold for x in resigned2))
    print('fn=', sum(x < threshold for x in resigned2))
    print('fp=', sum(x > threshold for x in active2))
    print('tn=', sum(x < threshold for x in active2))
    tp = sum(x > threshold for x in resigned2)
    fp = sum(x > threshold for x in active2)
    tn = sum(x < threshold for x in active2)
    fn = sum(x < threshold for x in resigned2)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(precision, recall, f1)

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
