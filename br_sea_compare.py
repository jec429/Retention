import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

# x_sea = pd.read_pickle("./data_files/SEA/merged_Sea_combined_x_numeric_newer.pkl")
# x_bra = pd.read_pickle("./data_files/BRAZIL/merged_Brazil_combined_x_numeric_newer.pkl")

x_sea = pd.read_csv('data_files/SEA/merged_Sea_combined.csv', sep=',')
x_bra = pd.read_csv('data_files/BRAZIL/merged_Brazil_combined.csv', sep=',')
x_chi = pd.read_csv('data_files/CHINA/merged_China_combined.csv', sep=',')

# print(x_sea.columns)
# print(x_bra.columns)

shared = [x for x in x_sea.columns.to_list() if x in x_bra.columns.to_list() and x in x_chi.columns.to_list()]
print('Shared')
print(shared)
sea_only = [x for x in x_sea.columns.to_list() if x not in x_bra.columns.to_list()]
print('SEA')
print(sea_only)
bra_only = [x for x in x_bra.columns.to_list() if x not in x_sea.columns.to_list()]
print('BRAZIL')
print(bra_only)

# x_bra['Lateral'] = x_bra['Lateral'].astype(int)
# x_sea['Lateral'] = x_sea['Lateral'].astype(int)
# x_chi['Lateral'] = x_chi['Lateral'].astype(int)

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
    if 'Location_Code' in h:
        continue
    print(h, x_bra[h].dtype)

    if x_bra[h].dtype == float or x_bra[h].dtype == 'int64':
        print(h)
        fig, ax = plt.subplots()
        x = [v for v in x_bra[h].values if v > -998]
        y = [v for v in x_sea[h].values if v > -998]
        z = [v for v in x_chi[h].values if v > -998]
        ax.hist(x, 20, density=1, label='BRAZIL', alpha=0.5, color="blue")
        ax.hist(y, 20, density=1, label='SEA', alpha=0.5, color="red")
        ax.hist(z, 20, density=1, label='CHINA', alpha=0.5, color="green")
        # plt.xlim(0.0, 1.0)
        ax.legend(loc='best')
        plt.xlabel(h)
        plt.savefig('./plots/compare/' + h + '.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        x = [v for v, s in zip(x_bra[h].values, x_bra['Status']) if v > -998 and s == 1]
        y = [v for v, s in zip(x_bra[h].values, x_bra['Status']) if v > -998 and s == 0]
        ax.hist(x, 20, density=1, label='Resigned', alpha=0.5, color="red")
        ax.hist(y, 20, density=1, label='Active', alpha=0.5, color="blue")
        # plt.xlim(0.0, 1.0)
        ax.legend(loc='best')
        plt.xlabel(h+'_BRAZIL')
        plt.savefig('./plots/compare/' + h + '_bra.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        x = [v for v, s in zip(x_sea[h].values, x_sea['Status']) if v > -998 and s == 1]
        y = [v for v, s in zip(x_sea[h].values, x_sea['Status']) if v > -998 and s == 0]
        ax.hist(x, 20, density=1, label='Resigned', alpha=0.5, color="red")
        ax.hist(y, 20, density=1, label='Active', alpha=0.5, color="blue")
        # plt.xlim(0.0, 1.0)
        ax.legend(loc='best')
        plt.xlabel(h + '_SEA')
        plt.savefig('./plots/compare/' + h + '_sea.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        x = [v for v, s in zip(x_chi[h].values, x_chi['Status']) if v > -998 and s == 1]
        y = [v for v, s in zip(x_chi[h].values, x_chi['Status']) if v > -998 and s == 0]
        ax.hist(x, 20, density=1, label='Resigned', alpha=0.5, color="red")
        ax.hist(y, 20, density=1, label='Active', alpha=0.5, color="blue")
        # plt.xlim(0.0, 1.0)
        ax.legend(loc='best')
        plt.xlabel(h + '_CHINA')
        plt.savefig('./plots/compare/' + h + '_chi.png')
        plt.close(fig)
    else:
        fig, ax = plt.subplots()
        x_chi[h].value_counts().plot(kind='barh', color='green', label='CHINA')
        x_bra[h].value_counts().plot(kind='barh', color='blue', alpha=0.5, label='BRAZIL')
        x_sea[h].value_counts().plot(kind='barh', color='red', alpha=0.3, label='SEA')
        ax.legend(loc='best')
        fig.subplots_adjust(left=0.4)
        plt.xlabel(h)
        plt.savefig('./plots/compare/' + h + '.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        x_bra[x_bra['Status'] == 1][h].value_counts().plot(kind='barh', color='red', alpha=0.5, label='Resigned')
        x_bra[x_bra['Status'] == 0][h].value_counts().plot(kind='barh', color='blue', alpha=0.5, label='Active')
        ax.legend(loc='best')
        fig.subplots_adjust(left=0.4)
        plt.xlabel(h + '_BRAZIL')
        plt.savefig('./plots/compare/' + h + '_bra.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        x_sea[x_sea['Status'] == 1][h].value_counts().plot(kind='barh', color='red', alpha=0.5, label='Resigned')
        x_sea[x_sea['Status'] == 0][h].value_counts().plot(kind='barh', color='blue', alpha=0.5, label='Active')
        ax.legend(loc='best')
        fig.subplots_adjust(left=0.4)
        plt.xlabel(h + '_SEA')
        plt.savefig('./plots/compare/' + h + '_sea.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        x_chi[x_chi['Status'] == 1][h].value_counts().plot(kind='barh', color='red', alpha=0.5, label='Resigned')
        x_chi[x_chi['Status'] == 0][h].value_counts().plot(kind='barh', color='blue', alpha=0.5, label='Active')
        ax.legend(loc='best')
        fig.subplots_adjust(left=0.4)
        plt.xlabel(h + '_CHINA')
        plt.savefig('./plots/compare/' + h + '_chi.png')
        plt.close(fig)

text = '<!DOCTYPE html>\n<html>\n<body>\n'

for g in glob.glob('./plots/compare/*.png'):
    text += '<img src = "' + g.replace('\\', '/')+'" >\n'

# for h in shared:
#     if h == 'WWID':
#         continue
#
#     text += '<img src = "./plots/compare/'+h+'.png" >\n'
#     text += '<img src = "./plots/compare/' + h + '_bra.png" >\n'
#     text += '<img src = "./plots/compare/' + h + '_sea.png" >\n'
#     text += '<img src = "./plots/compare/' + h + '_chi.png" >\n'
# # print(text)

text += '</body>\n</html>\n'

f = open('index_compare.html', 'w')
f.write(text)
f.close()
