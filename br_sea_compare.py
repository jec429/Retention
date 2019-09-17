import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# x_sea = pd.read_pickle("./data_files/SEA/merged_Sea_combined_x_numeric_newer.pkl")
# x_bra = pd.read_pickle("./data_files/BRAZIL/merged_Brazil_combined_x_numeric_newer.pkl")

x_sea = pd.read_csv('data_files/SEA/merged_Sea_combined.csv', sep=',')
x_bra = pd.read_csv('data_files/BRAZIL/merged_Brazil_combined.csv', sep=',')

# print(x_sea.columns)
# print(x_bra.columns)

shared = [x for x in x_sea.columns.to_list() if x in x_bra.columns.to_list()]
print('Shared')
print(shared)
sea_only = [x for x in x_sea.columns.to_list() if x not in x_bra.columns.to_list()]
print('SEA')
print(sea_only)
bra_only = [x for x in x_bra.columns.to_list() if x not in x_sea.columns.to_list()]
print('BRAZIL')
print(bra_only)

for h in shared:
    if 'Unnamed' in h:
        continue
    if 'Compensation_Range' in h:
        continue
    if 'Manager_WWID' in h:
        continue
    if 'Job_Sub_Function' in h:
        continue
    if 'WWID' in h:
        continue
    if 'Working_Country_Fixed' in h:
        continue
    print(h)

    if x_bra[h].dtype == float:
        fig, ax = plt.subplots()
        x = [v for v in x_bra[h].values if v > -998]
        y = [v for v in x_sea[h].values if v > -998]
        ax.hist(x, 20, density=1, label='BRAZIL', alpha=0.5, color="blue")
        ax.hist(y, 20, density=1, label='SEA', alpha=0.5, color="red")
        # plt.xlim(0.0, 1.0)
        ax.legend(loc='best')
        plt.xlabel(h)
    else:

        fig, ax = plt.subplots()
        x_bra[h].value_counts().plot(kind='barh', color='blue', label='BRAZIL')
        x_sea[h].value_counts().plot(kind='barh', color='red', alpha=0.5, label='SEA')
        ax.legend(loc='best')
        fig.subplots_adjust(left=0.4)
        plt.xlabel(h)

plt.show()
