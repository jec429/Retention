import pandas as pd
from preprocessing import OneHotEncoder
import sys
import numpy as np

#df_pre_original = pd.read_excel('./data_files/GlobalRetentionDataset.xlsx', sheet_name='Data')
df_original = pd.read_csv('data_files/HRRetentionDataforOVMerged.csv', sep=',')
df_original = df_original.replace(r'^\s*$', np.nan, regex=True)
# df_pre_original.info()
# df_original.info()
# df_original = df_original.fillna(0)
# df_pre_original = df_pre_original.fillna(0)
# l1 = df_pre_original['Length_of_Service_in_Years_inclu']
# l2 = df_original['Length_of_Service_in_Years_inclu']
#
# for x, y in zip(l1, l2):
#     try:
#         fx = float(x)
#         fy = float(y)
#     except:
#         print(x, y)
#         fx = 0.0
#         fy = 0.0
#
#     if fx != fy:
#         print(fx, fy)
# sys.exit()

df_original.info()
df_2017 = df_original[df_original['Report_Date'].str.contains('2017', na=False)]
df_2017.info()
print(df_2017.head())
# df = df_2017.sample(frac=1).reset_index(drop=True)
# print('size=', df.shape[0])
# data_x = df_original.drop(["Termination_Reason ", "2019 Status"], axis=1)
data_x = df_2017.drop(["Termination_Date"], axis=1)
to_drop = []
for c in data_x.columns:
    if '2019' in c:
        to_drop.append(c)

to_drop.append('ManagerID_2017')
to_drop.append('LatestHireDate_2017')
to_drop.append('client_id')
to_drop.append('date')
to_drop.append('lang')
to_drop.append('Job_Sub_Function__IA__Host_All_O')
to_drop.append('Location_Code__IA__Host_All_Othe')
# data_x = data_x.drop(["Level1", "Level2", "Level3", "Level4", "Level5", "Level6", "Level7"], axis=1)
# data_x = data_x.drop(["Manager ID", "Location", "Improve Performance", "Improve speed ", "Improve innovation ",
#                       "Company"], axis=1)

data_x = data_x.drop(to_drop, axis=1)

for c in data_x.columns:
    try:
        data_x[c] = data_x[c].astype('float64')
    except:
        print(c, ' is of type object')
    # if 'Planned_as_a' in c or 'CompaRatio' in c:
    #     data_x[c] = data_x[c].astype('float64')

    if 'Manager_Direct_Reports' in c or 'Job_Profile_Grade__IA__Host_All' in c:
        data_x[c] = data_x[c].astype(int)

    # if 'Length_of_Service_in_Yea' in c:
    #     data_x[c] = data_x[c].astype('float64')
    #
    # if 'EN_2017' in c or 'IN_2017' in c or 'HW_2017' in c or 'PL_2017' in c or 'TD_2017' in c or 'CR_2017' in c:
    #     data_x[c] = data_x[c].astype('float64')
    # if 'SA_2017' in c or 'IN5_2017' in c or 'PL1_2017' in c or 'PL5_2017' in c or 'TD3_2017' in c or 'CR1_2017' in c:
    #     data_x[c] = data_x[c].astype('float64')
    # if 'EN9_2017' in c or 'IN2_2017' in c or 'HW1_2017' in c or 'PL2_2017' in c or 'TD2_2017' in c or 'CR3_2017' in c:
    #     data_x[c] = data_x[c].astype('float64')
    # if 'EN7_2017' in c or 'IN3_2017' in c or 'HW4_2017' in c or 'PL3_2017' in c or 'TD1_2017' in c or 'CR4_2017' in c:
    #     data_x[c] = data_x[c].astype('float64')
    # if 'EN2_2017' in c or 'IN4_2017' in c or 'HW3_2017' in c or 'PL4_2017' in c or 'EN8_2017' in c or 'CR2_2017' in c:
    #     data_x[c] = data_x[c].astype('float64')
    # if 'DI_2017' in c or 'EN3_2017' in c or 'EN1_2017' in c or 'EN4_2017' in c or 'SA2_2017' in c or 'SA3_2017' in c:
    #     data_x[c] = data_x[c].astype('float64')
    # if 'DI2_2017' in c or 'EN5_2017' in c or 'EN6_2017' in c or 'EX_2017' in c or 'DI1_2017' in c or 'HW2_2017' in c:
    #     data_x[c] = data_x[c].astype('float64')

for EM in ['Employee', 'Manager']:
    for c in ['1', '2', '3']:
        print(data_x[EM + '_Rating_' + c].value_counts())

        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace('1', 1)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace('2', 1)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace('3', 1)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace('4', 1)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace('5', 1)

        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace('6', 2)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace('7', 2)

        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace('8', 3)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace('9', 3)

        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Insufficient Data to Rate / Insufficient Data to Rate', 0)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Does Not Meet / Partially Meets', 1)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Does Not Meet / Fully Meets', 1)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Fully Meets / Partially Meets', 1)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Partially Meets / Partially Meets', 1)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Partially Meets / Does Not Meet', 1)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Does Not Meet / Does Not Meet', 1)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Partially Meets / Fully Meets', 1)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Fully Meets / Does Not Meet', 1)

        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Fully Meets / Fully Meets', 2)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Fully Meets / Exceeds', 2)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Exceeds / Fully Meets', 2)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Partially Meets / Exceeds', 2)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Exceeds / Partially Meets', 2)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Exceeds / Does Not Meet', 2)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Does Not Meet / Exceeds', 2)

        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].replace(
            'Exceeds / Exceeds', 3)
        print('Cleaned')
        print(data_x[EM + '_Rating_' + c].value_counts())

for c in data_x.columns:
    print(c)
    if data_x[c].dtype == object:
        data_x[c] = data_x[c].fillna('Missing')
        data_x[c] = data_x[c].astype('category')
    else:
        data_x[c] = data_x[c].fillna(-999)
data_x['Working_Country_Fixed'] = data_x['Working_Country_Fixed'].cat.codes
data_x.info()

data_x_numeric = OneHotEncoder().fit_transform(data_x)

print(data_x_numeric.head())

# for c in data_x_numeric.columns:
#     if 'Missing' in c:
#         print(c)
#         data_x_numeric = data_x_numeric.drop(c, axis=1)

data_x_numeric.info()
data_x_numeric = data_x_numeric.fillna(-999)

# test_1 = np.random.normal(loc=0.0, scale=0.01, size=data_x_numeric[data_x_numeric['Status'] == 1].shape[0])
# test_2 = np.random.normal(loc=1.0, scale=0.01, size=data_x_numeric[data_x_numeric['Status'] == 0].shape[0])
# assert data_x_numeric.shape[0] == len(test_1) + len(test_2)
# test_l = list(test_1) + list(test_2)
# data_x_numeric['test_gauss'] = test_l

print(data_x_numeric.columns.to_list())
data_x_numeric.to_pickle("./data_files/OurVoice/ourvoice_merged_x_numeric_newer.pkl")
# data_x_numeric = data_x_numeric[:100]
data_x_numeric.to_csv("./data_files/OurVoice/ourvoice_merged_x_numeric_newer.csv", sep=',', encoding='utf-8')
