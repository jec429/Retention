import pandas as pd
from preprocessing import OneHotEncoder
import sys
import numpy as np


df_original = pd.read_csv('data_files/OVMergedforRention2019.csv', sep=',')
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
df_2019 = pd.DataFrame()
df_2019['Job_Profile_Grade__IA__Host_All'] = df_original['Job_Profile_Grade__IA__Host_All']
df_2019['Location_Code__IA__Host_All_Othe'] = df_original['Location_Code__IA__Host_All_Othe']
df_2019['Job_Function__IA__Host_All_Other'] = df_original['Job_Function__IA__Host_All_Other']
df_2019['Job_Sub_Function__IA__Host_All_O'] = df_original['Job_Sub_Function__IA__Host_All_O']
df_2019['Length_of_Service_in_Years_inclu'] = df_original['Length_of_Service_in_Years_inclu']
df_2019['Employee_Rating_1'] = df_original['Employee_Rating_1']
df_2019['Employee_Rating_Year_Flag_1'] = (df_original['Employee_Rating_Year_1'] == 2018)
# df_2019['Employee_Rating_Year_1'] = df_original['Employee_Rating_Year_1']
# df_2019['Employee_Rating_2'] = df_original['Employee_Rating_2']
# df_2019['Employee_Rating_Year_2'] = df_original['Employee_Rating_Year_2']
# df_2019['Employee_Rating_3'] = df_original['Employee_Rating_3']
# df_2019['Employee_Rating_Year_3'] = df_original['Employee_Rating_Year_3']
df_2019['Working_Country_Fixed'] = df_original['Working_Country_Fixed']
df_2019['Region_Fixed'] = df_original['Region_Fixed']
df_2019['Sector_Fixed'] = df_original['Sector_Fixed']
df_2019['Compa_Ratio'] = df_original['Compa_Ratio']
df_2019['Participation_Rate'] = df_original['Participation_Rate']
df_2019['Promo'] = df_original['Promo_2019']
df_2019['Lateral'] = df_original['Lateral_2019']
df_2019['Cross_Move'] = df_original['Cross_Move_2019']
df_2019['Lateral'] = df_original['Mgr_Change_2019']
df_2019['Lateral'] = df_original['SkipLevel_Mgr_Change_2019']
df_2019['Manager_Rating_1'] = df_original['Manager_Rating_1']
df_2019['Manager_Rating_Year_Flag_1'] = (df_original['Manager_Rating_Year_1'] == 2018)
# df_2019['Manager_Rating_2'] = df_original['Manager_Rating_2']
# df_2019['Manager_Rating_3'] = df_original['Manager_Rating_3']
# df_2019['Manager_Rating_Year_1'] = df_original['Manager_Rating_Year_1']
# df_2019['Manager_Rating_Year_2'] = df_original['Manager_Rating_Year_2']
# df_2019['Manager_Rating_Year_3'] = df_original['Manager_Rating_Year_3']
df_2019['Rehire_YN'] = df_original['Rehire_YN']
df_2019['Employee_Direct_Reports'] = df_original['Employee_Direct_Reports']
df_2019['Manager_Direct_Reports'] = df_original['Manager_Direct_Reports']
df_2019['Highest_Degree_Received'] = df_original['Highest_Degree_Received']
df_2019['Planned_as_a___of_Merit_Tar_1'] = df_original['@_018_Planned_as_a___of_Merit_Tar']
df_2019['Planned_as_a___of_Merit_Tar_Flag_1'] = (df_original['@_016_Planned_as_a___of_Merit_Tar'] == 0)
# df_2019['Planned_as_a___of_Bonus_Tar_1'] = df_original['@_018_Planned_as_a___of_Bonus_Tar']
# df_2019['Planned_as_a___of_LTI_Targe_1'] = df_original['@_018_Planned_as_a___of_LTI_Targe']
# df_2019['Planned_as_a___of_Merit_Tar_2'] = df_original['@_017_Planned_as_a___of_Merit_Tar']
# df_2019['Planned_as_a___of_Bonus_Tar_2'] = df_original['@_017_Planned_as_a___of_Bonus_Tar']
# df_2019['Planned_as_a___of_LTI_Targe_2'] = df_original['@_017_Planned_as_a___of_LTI_Targe']
df_2019['Year_Month'] = df_original['Year_Month']
df_2019['Employee_Pay_Grade'] = df_original['Employee_Pay_Grade']
df_2019['Time_Type__IA__Host_All_Other__P'] = df_original['Time_Type__IA__Host_All_Other__P']
df_2019['Legal_Entity_Code__IA__Host_All'] = df_original['Legal_Entity_Code__IA__Host_All']
df_2019['Legal_Entity_Description__IA__Ho'] = df_original['Legal_Entity_Description__IA__Ho']
df_2019['MRC_Code__IA__Host_All_Other__Pr'] = df_original['MRC_Code__IA__Host_All_Other__Pr']
df_2019['Report_Date'] = df_original['Report_Date']
df_2019['Employee_Type__IA__Host_All_Othe'] = df_original['Employee_Type__IA__Host_All_Othe']
df_2019['MRC_Description__IA__Host_All_Ot'] = df_original['MRC_Description__IA__Host_All_Ot']
df_2019['Manager_WWID__IA__Host_All_Other'] = df_original['Manager_WWID__IA__Host_All_Other']
df_2019['Hired_2019'] = 0  # (df_original['LatestHireDate_2019'].str.contains('2019'))

# df_2019['Manager_ID'] = df_original['ManagerID_2019']
df_2019['Tenure'] = df_original['Tenure_OV2019']
df_2019['Frustrated'] = df_original['frustrated_2019']
df_2019['Sad'] = df_original['Sad_2019']
# df_2019['Neutral'] = df_original['Neutral_2017']
df_2019['Surprised'] = df_original['Surprised_2019']
df_2019['Happy'] = df_original['Happy_2019']
# df_2019[''] = df_original['CO1_2017']
# df_2019[''] = df_original['CO2_2017']
# df_2019[''] = df_original['CO3_2017']
# df_2019[''] = df_original['CO4_2017']
df_2019['CO'] = df_original['CO2019']
# df_2019[''] = df_original['EX1_2017']
# df_2019[''] = df_original['EX2_2017']
# df_2019[''] = df_original['EX3_2017']
# df_2019[''] = df_original['EX4_2017']
# df_2019[''] = df_original['EX5_2017']
# df_2019[''] = df_original['EX6_2017']
df_2019['EX'] = df_original['EX2019']
# df_2019[''] = df_original['DI1_2017']
# df_2019[''] = df_original['DI2_2017']
# df_2019[''] = df_original['DI3_2017']
# df_2019[''] = df_original['DI4_2017']
# df_2019[''] = df_original['DI5_2017']
df_2019['DI'] = df_original['DI2019']
# df_2019[''] = df_original['EN1_2017']
# df_2019[''] = df_original['EN2_2017']
# df_2019[''] = df_original['EN3_2017']
# df_2019[''] = df_original['EN4_2017']
# df_2019[''] = df_original['EN5_2017']
# df_2019[''] = df_original['EN6_2017']
# df_2019[''] = df_original['EN7_2017']
df_2019['EN8'] = df_original['EN8_2019']
# df_2019[''] = df_original['EN9_2017']
df_2019['EN'] = df_original['EN2019']
# df_2019[''] = df_original['HW1_2017']
# df_2019[''] = df_original['HW2_2017']
# df_2019[''] = df_original['HW3_2017']
# df_2019[''] = df_original['HW4_2017']
df_2019['HW'] = df_original['HW2019']
# df_2019[''] = df_original['IN1_2017']
# df_2019[''] = df_original['IN2_2017']
# df_2019[''] = df_original['IN3_2017']
# df_2019[''] = df_original['IN4_2017']
# df_2019[''] = df_original['IN5_2017']
# df_2019[''] = df_original['IN6_2017']
df_2019['IN'] = df_original['IN2019']
# df_2019[''] = df_original['SA1_2017']
# df_2019[''] = df_original['SA2_2017']
# df_2019[''] = df_original['SA3_2017']
df_2019['SA'] = df_original['SA2019']
# df_2019[''] = df_original['PL1_2017']
# df_2019[''] = df_original['PL2_2017']
# df_2019[''] = df_original['PL3_2017']
# df_2019[''] = df_original['PL4_2017']
# df_2019[''] = df_original['PL5_2017']
df_2019['PL'] = df_original['PL2019']
# df_2019[''] = df_original['TD1_2017']
# df_2019[''] = df_original['TD2_2017']
# df_2019[''] = df_original['TD3_2017']
df_2019['TD'] = df_original['TD2019']
# df_2019[''] = df_original['CR1_2017']
# df_2019[''] = df_original['CR2_2017']
# df_2019[''] = df_original['CR3_2017']
# df_2019[''] = df_original['CR4_2017']
df_2019['CR'] = df_original['CR2019']

df_2019['Status'] = 0
df_2019['Survey_taken'] = df_original['ECMember_OV2019']

df_2019.info()
print(df_2019.head())
# df = df_2019.sample(frac=1).reset_index(drop=True)
# print('size=', df.shape[0])
# data_x = df_original.drop(["Termination_Reason ", "2019 Status"], axis=1)
data_x = df_2019

to_drop = []
for c in data_x.columns:
    if '_2019' in c and len(c) == 8:
        if 'Sad_2019' in c or 'EN8_2019' in c:
            continue
        print(c)
        to_drop.append(c)

# sys.exit()

# to_drop.append('ManagerID_2017')
# to_drop.append('LatestHireDate_2017')
# to_drop.append('client_id')
# to_drop.append('date')
# to_drop.append('lang')
to_drop.append('Job_Sub_Function__IA__Host_All_O')
# to_drop.append('Location_Code__IA__Host_All_Othe')
# to_drop.append('Manager_WWID__IA__Host_All_Other')
to_drop.append('MRC_Description__IA__Host_All_Ot')
to_drop.append('Legal_Entity_Description__IA__Ho')
# to_drop.append('Promo_2016')
# to_drop.append('Lateral_2016')
# to_drop.append('Cross_Move_2016')
# to_drop.append('Mgr_Change_2016')
# to_drop.append('SkipLevel_Mgr_Change_2016')
# to_drop.append('SkipLevel_Mgr_Change')
# to_drop.append('SkipLevel_Mgr_Change_YN')

# data_x = data_x.drop(["Level1", "Level2", "Level3", "Level4", "Level5", "Level6", "Level7"], axis=1)
# data_x = data_x.drop(["ImprovePerformance", "Improvespeed", "Improveinnovation"], axis=1)

data_x = data_x.drop(to_drop, axis=1)

for c in data_x.columns:
    try:
        data_x[c] = data_x[c].astype('float64')
    except:
        print(c, ' is of type object')

#    if 'Manager_Direct_Reports' in c or 'Job_Profile_Grade__IA__Host_All' in c:
#        data_x[c] = data_x[c].astype(int)


for EM in ['Employee', 'Manager']:
    for c in ['1']:
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
            'Insufficient Data to Rate', 0)
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
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].fillna(-999)
        data_x[EM + '_Rating_' + c] = data_x[EM + '_Rating_' + c].astype(int)
        print(data_x[EM + '_Rating_' + c].value_counts())


data_x['Highest_Degree_Received'] = data_x['Highest_Degree_Received'].replace('BE - Doctorate', 'Doctorate (PHD) or Equivalent')
data_x['Highest_Degree_Received'] = data_x['Highest_Degree_Received'].replace("BE - Master's Degree (University)", 'Masters Degree or Equivalent')
data_x['Highest_Degree_Received'] = data_x['Highest_Degree_Received'].replace('BE - Post-university education', 'Post-Graduate Diploma/Degree')
data_x['Highest_Degree_Received'] = data_x['Highest_Degree_Received'].replace("BE - Bachelor's degree (high-school, non-university level)", 'University/Bachelors Degree or Equivalent')
data_x['Highest_Degree_Received'] = data_x['Highest_Degree_Received'].replace("BE - Bachelor's Degree (University)", 'University/Bachelors Degree or Equivalent')
data_x['Highest_Degree_Received'] = data_x['Highest_Degree_Received'].replace("BE - Master's degree (high-school, non-university level)", 'Masters Degree or Equivalent')

# data_x = data_x.drop(["Highest_Degree_Received"], axis=1)
data_x['Highest_Degree_Received'] = data_x['Highest_Degree_Received'].astype('category')
data_x['Highest_Degree_Received'] = data_x['Highest_Degree_Received'].cat.codes

data_x['Employee_Rating_1'] = data_x['Employee_Rating_1'].astype(object)
data_x['Manager_Rating_1'] = data_x['Manager_Rating_1'].astype(object)
data_x['Job_Profile_Grade__IA__Host_All'] = data_x['Job_Profile_Grade__IA__Host_All'].astype(object)
data_x['Planned_as_a___of_Merit_Tar_Flag_1'] = (data_x['Planned_as_a___of_Merit_Tar_1'] == 0)
data_x['ID'] = range(data_x.shape[0])
data_x = data_x[(~data_x['Status'].isnull())]
print('Shape')
print('Original=', data_x.shape[0])
data_x.info()
data_x = data_x[(data_x['Job_Profile_Grade__IA__Host_All'] > 0) & (~data_x['Survey_taken'].isnull()) &
                (data_x['Compa_Ratio'] > 0) & (~data_x['Planned_as_a___of_Merit_Tar_1'].isnull()) &
                (data_x['Employee_Type__IA__Host_All_Othe'].str.contains('Intern')==False) &
                (data_x['Employee_Type__IA__Host_All_Othe'].str.contains('Fixed Term')==False) &
                (data_x['Location_Code__IA__Host_All_Othe'].str.contains('MX303')==False)]
print('Filtered=', data_x.shape[0])

print('Manager Ratings')
avg_mgr_rating1 = []
avg_mgr_rating2 = []
avg_mgr_rating3 = []
for w, r1 in zip(data_x['Manager_WWID__IA__Host_All_Other'], data_x['Manager_Rating_1'],
                 # data_x['Manager_Rating_2'], data_x['Manager_Rating_3']):
                 ):
    r1s = data_x[data_x['Manager_WWID__IA__Host_All_Other'] == w]['Manager_Rating_1']
    r1m = r1s.mean(axis=0, skipna=True)
    avg_mgr_rating1.append(r1m)
    # r2s = data_x[data_x['Manager_WWID__IA__Host_All_Other'] == w]['Manager_Rating_2']
    # r2m = r2s.mean(axis=0, skipna=True)
    # avg_mgr_rating2.append(r2m)
    # r3s = data_x[data_x['Manager_WWID__IA__Host_All_Other'] == w]['Manager_Rating_3']
    # r3m = r3s.mean(axis=0, skipna=True)
    # avg_mgr_rating3.append(r3m)

data_x['Avg_Mgr_Rating_1'] = avg_mgr_rating1
# data_x['Avg_Mgr_Rating_2'] = avg_mgr_rating2
# data_x['Avg_Mgr_Rating_3'] = avg_mgr_rating3

for c in data_x.columns:
    if data_x[c].dtype == object:
        print(c, data_x[c].value_counts())
        data_x[c] = data_x[c].fillna('Missing')
        data_x[c] = data_x[c].astype('category')
    else:
        data_x[c] = data_x[c].fillna(-999)
data_x['Working_Country_Fixed'] = data_x['Working_Country_Fixed'].cat.codes
# data_x['Status'] = data_x['Status'].cat.codes
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
data_x_numeric.to_pickle("./data_files/OurVoice/ourvoice_merged_2019_fixed_x_numeric_newer.pkl")
data_x_numeric.to_csv("./data_files/OurVoice/ourvoice_merged_2019_fixed_x_numeric_newer.csv", sep=',', encoding='utf-8')
