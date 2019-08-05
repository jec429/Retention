import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys


#df = pd.read_excel(r'C:\Users\jchaves6\Documents\Brazil_Retention_Dataset2.xlsx')
df = pd.read_csv('Brazil_Retention_Dataset_07102019.csv', sep=',')

print(df.head())
df = df.drop_duplicates(subset='WWID', keep="last")
print(df.head())

df = df.drop(["Legal_Name", "First_Name", "Last_Name", "Time_in_Position__IA__Host_All_O",
              "Time_in_Position__Core_or_Core_V", "Hourly_Rate___Currency", "Manager_Name__IA__Host_All_Other",
              "Supervisory_Organization___Hiera", "Worker"
              ], axis=1)

print(df['Termination_Date'][0], pd.to_datetime('1999/1/31'))
#df['Termination_Date'] = df['Termination_Date'].replace(
#    df['Termination_Date'][0], pd.to_datetime('1999/1/31')).astype(np.int64)

df['Termination_Date'] = df['Termination_Date']#/1e11 - 9.177408e+06

print('len df=', len(df.columns))
for c in df.columns:
    #print(c, len(df[c].value_counts()))
    if len(df[c].value_counts()) < 2:
        df = df.drop(c, axis=1)
print('len2 df=', len(df.columns))

#df['First_Year_Attended'] = df['First_Year_Attended'].astype(np.int64)*1e-9
#df['Last_Year_Attended'] = df['Last_Year_Attended'].astype(np.int64)*1e-9
#df['Year_Degree_Received'] = df['Year_Degree_Received'].astype(np.int64)*1e-9
#df['Original_Hire_Date'] = df['Original_Hire_Date'].astype(np.int64)*1e-9
#df['Hire_Date__Most_Recent_'] = df['Hire_Date__Most_Recent_'].astype(np.int64)*1e-9


print(df.size)
df2 = pd.DataFrame()
df2['WWID'] = df['WWID']
df2['Termination_Reason'] = df['Termination_Reason']

for c in ['Compensation_Range___Midpoint', 'Total_Base_Pay___Local', 'Job_Sub_Function__IA__Host_All_O',
          'Length_of_Service_in_Years_inclu', 'Job_Function__IA__Host_All_Other', 'Promotion', 'Demotion',
          'Lateral', 'Cross_Move', 'Trainings_Completed', 'Mgr_Change', 'SkipLevel_Mgr_Change', 'Rehire_YN',
          '_018_Planned_as_a___of_Bonus_Tar', '_017_Planned_as_a___of_Bonus_Tar', '_016_Planned_as_a___of_Bonus_Tar',
          'Highest_Degree_Received', 'Actual_Sales_Incentive__2016', 'Actual_Sales_Incentive__2017',
          'Actual_Sales_Incentive__2018', 'Target_Sales_Incentive__2016', 'Target_Sales_Incentive__2017',
          'Target_Sales_Incentive__2018']:
    df2[c] = df[c]

for EM in ['Employee', 'Manager']:
    for c in range(1, 4):
        c = str(c)
        df2[EM+'_Rating_'+c] = df[EM+'_Rating_'+c]
        #print(EM+'_Rating_'+c, df2[EM+'_Rating_'+c].value_counts())

df_filtered = df2.query('Termination_Reason != "End of Contract/Assignment Completed"')

df_filtered['Compensation_Range___Midpoint'] = df_filtered['Compensation_Range___Midpoint'].replace(0, 1e9)

df_filtered['Compa_Diff_Ratio'] = (df_filtered['Total_Base_Pay___Local']-df_filtered['Compensation_Range___Midpoint'])\
                                  / df_filtered['Compensation_Range___Midpoint']
df_filtered['Compa_Ratio'] = df_filtered['Total_Base_Pay___Local']/df_filtered['Compensation_Range___Midpoint']

df_filtered['Length_of_Service_in_Years_log'] = np.log(df_filtered['Length_of_Service_in_Years_inclu']+1)

df_filtered['Sales_Incentive_2016'] = df_filtered['Actual_Sales_Incentive__2016'] - \
                                      df_filtered['Target_Sales_Incentive__2016']
df_filtered['Sales_Incentive_2017'] = df_filtered['Actual_Sales_Incentive__2017'] - \
                                      df_filtered['Target_Sales_Incentive__2017']
df_filtered['Sales_Incentive_2018'] = df_filtered['Actual_Sales_Incentive__2018'] - \
                                      df_filtered['Target_Sales_Incentive__2018']

df_filtered = df_filtered.drop(['Actual_Sales_Incentive__2016', 'Actual_Sales_Incentive__2017',
                                'Actual_Sales_Incentive__2018', 'Target_Sales_Incentive__2016',
                                'Target_Sales_Incentive__2017', 'Target_Sales_Incentive__2018'], axis=1)

for EM in ['Employee', 'Manager']:
    for c in range(1, 4):
        c = str(c)
        df_filtered[EM+'_Rating_'+c+'_W'] = df_filtered[EM+'_Rating_'+c].str.split('/').str.get(0).str.strip()
        df_filtered[EM+'_Rating_'+c+'_H'] = df_filtered[EM+'_Rating_'+c].str.split('/').str.get(1).str.strip()
        df_filtered = df_filtered.drop(EM+'_Rating_' + c, axis=1)
        print(df_filtered['Employee_Rating_'+c+'_W'].value_counts())
        for s in ['W', 'H']:
            df_filtered[EM + '_Rating_' + c + '_' + s] = df_filtered[EM + '_Rating_' + c + '_' + s].replace(
                '3', None)
            df_filtered[EM + '_Rating_' + c + '_' + s] = df_filtered[EM + '_Rating_' + c + '_' + s].replace(
                '4', None)
            df_filtered[EM + '_Rating_' + c + '_' + s] = df_filtered[EM + '_Rating_' + c + '_' + s].replace(
                '5', None)
            df_filtered[EM + '_Rating_' + c + '_' + s] = df_filtered[EM + '_Rating_' + c + '_' + s].replace(
                '6', None)
            df_filtered[EM + '_Rating_' + c + '_' + s] = df_filtered[EM + '_Rating_' + c + '_' + s].replace(
                '7', None)
            df_filtered[EM + '_Rating_' + c + '_' + s] = df_filtered[EM + '_Rating_' + c + '_' + s].replace(
                '8', None)
            df_filtered[EM + '_Rating_' + c + '_' + s] = df_filtered[EM + '_Rating_' + c + '_' + s].replace(
                '9', None)
            df_filtered[EM+'_Rating_'+c+'_'+s] = df_filtered[EM+'_Rating_'+c+'_'+s].replace('Exceeds', 4)
            df_filtered[EM+'_Rating_'+c+'_'+s] = df_filtered[EM+'_Rating_'+c+'_'+s].replace('Fully Meets', 3)
            df_filtered[EM+'_Rating_'+c+'_'+s] = df_filtered[EM+'_Rating_'+c+'_'+s].replace('Partially Meets', 2)
            df_filtered[EM + '_Rating_' + c + '_' + s] = df_filtered[EM + '_Rating_' + c + '_' + s].replace(
                'Does Not Meet', 1)
            df_filtered[EM + '_Rating_' + c + '_' + s] = df_filtered[EM + '_Rating_' + c + '_' + s].replace(
                'Insufficient Data to Rate', None)

print(df_filtered['Job_Function__IA__Host_All_Other'].value_counts())

Job_Function_encoded = []
for x in df_filtered['Job_Function__IA__Host_All_Other'].values:
    if 'Operations' in x:
        Job_Function_encoded.append(0)
    elif 'Sales' in x:
        Job_Function_encoded.append(1)
    elif 'Quality' in x or 'General Management':
        Job_Function_encoded.append(2)
    elif 'Finance' in x or 'Facilities':
        Job_Function_encoded.append(3)
    elif 'R&D' in x or 'Legal':
        Job_Function_encoded.append(4)
    elif 'General Administration' in x or 'Strategic Planning':
        Job_Function_encoded.append(5)
    elif 'Engineering' in x or 'Public Affairs':
        Job_Function_encoded.append(6)
    elif 'Marketing' in x or 'Regulatory Affairs':
        Job_Function_encoded.append(7)
    elif 'Info Technology' in x or 'Human Resources':
        Job_Function_encoded.append(8)
    else:
        Job_Function_encoded.append(10)

df_filtered['Job_Function_encoded'] = Job_Function_encoded

for c in df_filtered.columns:
    if df_filtered[c].dtype == object:
        df_filtered[c] = df_filtered[c].fillna('Missing')
    else:
        df_filtered[c] = df_filtered[c].fillna(-999)

df_filtered['delta'] = [1 if x == 'Resignation' else 0 for x in df_filtered['Termination_Reason']]

df_filtered['Highest_Degree_Received'] = df_filtered['Highest_Degree_Received'].replace('Missing', -999)
df_filtered['Highest_Degree_Received'] = df_filtered['Highest_Degree_Received'].replace('General Equivalency Diploma (GED)', 0)
df_filtered['Highest_Degree_Received'] = df_filtered['Highest_Degree_Received'].replace('High School', 1)
df_filtered['Highest_Degree_Received'] = df_filtered['Highest_Degree_Received'].replace('Vocational, Certificate, Technical or Associates', 2)
df_filtered['Highest_Degree_Received'] = df_filtered['Highest_Degree_Received'].replace('University/Bachelors Degree or Equivalent', 3)
df_filtered['Highest_Degree_Received'] = df_filtered['Highest_Degree_Received'].replace('Master of Business Administration (MBA)', 4)
df_filtered['Highest_Degree_Received'] = df_filtered['Highest_Degree_Received'].replace('Masters Degree or Equivalent', 5)
df_filtered['Highest_Degree_Received'] = df_filtered['Highest_Degree_Received'].replace('Post-Graduate Diploma/Degree', 6)
df_filtered['Highest_Degree_Received'] = df_filtered['Highest_Degree_Received'].replace('Doctorate (PHD) or Equivalent', 7)
df_filtered['Highest_Degree_Received'] = df_filtered['Highest_Degree_Received'].replace('Fellow', 8)
df_filtered['Highest_Degree_Received'] = df_filtered['Highest_Degree_Received'].replace('Medical Doctor (MD)', 9)
df_filtered['Highest_Degree_Received'] = df_filtered['Highest_Degree_Received'].replace('Juris Doctor (JD) / Qualified Lawyer or Equivalent', 10)
df_filtered['Highest_Degree_Received'] = df_filtered['Highest_Degree_Received'].replace('Registered Nurse (RN)', 11)
df_filtered['Highest_Degree_Received'] = df_filtered['Highest_Degree_Received'].replace('Doctorate (PharmD)', 12)
df_filtered['Highest_Degree_Received'] = df_filtered['Highest_Degree_Received'].replace('Doctor of Dental Surgery (DDS)', 13)

print(df_filtered['Highest_Degree_Received'].value_counts())

print(df.shape, df_filtered.shape)
file_name = 'clean_data.csv'
df_filtered.to_csv(file_name, sep=',', encoding='utf-8')
