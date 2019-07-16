import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys


df = pd.read_excel(r'C:\Users\jchaves6\Documents\Brazil_Retention_Dataset2.xlsx')

print(df.head())
df = df.drop_duplicates(subset='WWID', keep="last")
print(df.head())

df = df.drop(["Legal_Name", "First_Name", "Last_Name", "Time_in_Position__IA__Host_All_O",
              "Time_in_Position__Core_or_Core_V", "Hourly_Rate___Currency", "WWID", "Manager_Name__IA__Host_All_Other",
              "Supervisory_Organization___Hiera", "Worker"
              ], axis=1)

print(df['Termination_Date'][0], pd.to_datetime('1999/1/31'))
df['Termination_Date'] = df['Termination_Date'].replace(
    df['Termination_Date'][0], pd.to_datetime('1999/1/31')).astype(np.int64
                                                                   )

df['Termination_Date'] = df['Termination_Date']/1e11 - 9.177408e+06

print('len df=', len(df.columns))
for c in df.columns:
    #print(c, len(df[c].value_counts()))
    if len(df[c].value_counts()) < 2:
        df = df.drop(c, axis=1)
print('len2 df=', len(df.columns))

df['First_Year_Attended']  = df['First_Year_Attended'].astype(np.int64)*1e-9
df['Last_Year_Attended']   = df['Last_Year_Attended'].astype(np.int64)*1e-9
df['Year_Degree_Received'] = df['Year_Degree_Received'].astype(np.int64)*1e-9
df['Original_Hire_Date']   = df['Original_Hire_Date'].astype(np.int64)*1e-9
df['Hire_Date__Most_Recent_'] = df['Hire_Date__Most_Recent_'].astype(np.int64)*1e-9


print(df.size)
df2 = pd.DataFrame()
df2['Termination_Reason'] = df['Termination_Reason']

for c in ['Compensation_Range___Midpoint', 'Total_Base_Pay___Local', 'Job_Sub_Function__IA__Host_All_O',
          'Length_of_Service_in_Years_inclu', 'Job_Function__IA__Host_All_Other', 'Promotion', 'Demotion',
          'Lateral', 'Cross_Move', 'Trainings_Completed', 'Mgr_Change', 'SkipLevel_Mgr_Change', 'Rehire_YN',
          '_018_Planned_as_a___of_Bonus_Tar', '_017_Planned_as_a___of_Bonus_Tar', '_016_Planned_as_a___of_Bonus_Tar']:
    df2[c] = df[c]

for EM in ['Employee', 'Manager']:
    for c in range(1, 4):
        c = str(c)
        df2[EM+'_Rating_'+c] = df[EM+'_Rating_'+c]
        #print(EM+'_Rating_'+c, df2[EM+'_Rating_'+c].value_counts())

df_filtered = df2.query('Termination_Reason != "End of Contract/Assignment Completed"')

#print(df_filtered.size)
#df_filtered = df_filtered[~df_filtered.duplicated(keep="first")]


df_filtered['Compensation_Range___Midpoint'] = df_filtered['Compensation_Range___Midpoint'].replace(0, 1e9)

df_filtered['Compa_Diff_Ratio'] = (df_filtered['Total_Base_Pay___Local']-df_filtered['Compensation_Range___Midpoint'])\
                                  /df_filtered['Compensation_Range___Midpoint']
df_filtered['Compa_Ratio'] = df_filtered['Total_Base_Pay___Local']/df_filtered['Compensation_Range___Midpoint']

df_filtered['Length_of_Service_in_Years_log'] = np.log(df_filtered['Length_of_Service_in_Years_inclu']+1)


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

#df_filtered[EM+'_Rating_'+c+'_'+s] = df_filtered[EM+'_Rating_'+c+'_'+s].fillna(0)
            #df_filtered[EM + '_Rating_' + c + '_' + s] = df_filtered[EM + '_Rating_' + c + '_' + s].astype('int64')

df_filtered_1 = df_filtered.query('Termination_Reason == "Resignation"')
df_filtered_2 = df_filtered.query('Termination_Reason == "Active"')

num_bins = 40


print(len(df_filtered_1.columns))

for c in df_filtered.columns:
    if 'Job_Function' in c or 'Rating' in c:
        continue
    fig, ax = plt.subplots()
    #print(c, df_filtered[c].dtype)
    ax.set_title(c)
    if df_filtered[c].dtype == object:
        df_filtered_1[c] = df_filtered_1[c].fillna('missing')
        df_filtered_2[c] = df_filtered_2[c].fillna('missing')
        df_filtered_1[c] = df_filtered_1[c].astype('category')
        df_filtered_2[c] = df_filtered_2[c].astype('category')
        df_filtered_1[c] = df_filtered_1[c].cat.codes
        df_filtered_2[c] = df_filtered_2[c].cat.codes
        ax.hist(df_filtered_2[c], num_bins, density=1, label='Active', alpha=0.5, color="blue")
        ax.hist(df_filtered_1[c], num_bins, density=1, label='Resigned', alpha=0.5, color="red")
        ax.legend(loc='best')
        plt.savefig('plots/'+c + '.png')
    else:
        df_filtered_1[c] = df_filtered_1[c].fillna(-1)
        df_filtered_2[c] = df_filtered_2[c].fillna(-1)
        ax.hist(df_filtered_2[c], num_bins, density=1, label='Active', alpha=0.5, color="blue")
        ax.hist(df_filtered_1[c], num_bins, density=1, label='Resigned', alpha=0.5, color="red")
        ax.legend(loc='best')
        plt.xlabel(c.replace('_', ' '))
        plt.savefig('plots/'+c+'.png')
    plt.close(fig)


for c1 in df_filtered_1.columns:
    if 'Job_Function' in c1 or 'Rating' in c1:
        continue
    for c2 in df_filtered_1.columns:
        if 'Job_Function' in c2 or 'Rating' in c2:
            continue
        if c1 == c2:
            continue
        print(c1, c2, df_filtered_1[c1].dtype, df_filtered_1[c2].dtype)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.scatter(df_filtered_2[c1], df_filtered_2[c2], alpha=0.3, c="blue", edgecolors='none', s=30, label="Active")
        ax2.scatter(df_filtered_1[c1], df_filtered_1[c2], alpha=0.8, c="red", edgecolors='none', s=30, label="Resigned")
        plt.xlabel(c1.replace('_', ' '))
        plt.ylabel(c2.replace('_', ' '))
        plt.legend(loc='best')
        fig2.tight_layout()
        plt.savefig('plots/'+c1+'_'+c2+'.png')
        plt.close(fig2)

c = 'Comp_away_from_midpoint'
fig, ax = plt.subplots()
ax.set_title(c)
ax.hist(df_filtered_1['Total_Base_Pay___Local']-df_filtered_1['Compensation_Range___Midpoint'],
        num_bins, density=1, label='Resigned', alpha=0.5, color="red")
ax.hist(df_filtered_2['Total_Base_Pay___Local']-df_filtered_2['Compensation_Range___Midpoint'],
        num_bins, density=1, label='Active', alpha=0.5, color="blue")
ax.legend(loc='best')
plt.savefig('plots/' + c + '.png')
plt.close(fig)

c = 'Comp_ratio'
fig, ax = plt.subplots()
ax.set_title(c)
df_filtered_1['Compensation_Range___Midpoint'] = df_filtered_1['Compensation_Range___Midpoint'].replace(0, 1e9)
df_filtered_2['Compensation_Range___Midpoint'] = df_filtered_2['Compensation_Range___Midpoint'].replace(0, 1e9)
ax.hist(df_filtered_1['Total_Base_Pay___Local']/df_filtered_1['Compensation_Range___Midpoint'],
        num_bins, density=1, label='Resigned', alpha=0.5, color="red")
ax.hist(df_filtered_2['Total_Base_Pay___Local']/df_filtered_2['Compensation_Range___Midpoint'],
        num_bins, density=1, label='Active', alpha=0.5, color="blue")
ax.legend(loc='best')
plt.savefig('plots/' + c + '.png')
plt.close(fig)
#plt.show()

c = 'Job_Function__IA__Host_All_Other'
fig, ax = plt.subplots()
ax.set_title(c)

#print(df_filtered_1[c].value_counts(), df_filtered_2[c].value_counts())

ax.barh(df_filtered_1[c].value_counts().index, df_filtered_1[c].value_counts(), label='Resigned', alpha=0.8, color="red")
ax.barh(df_filtered_2[c].value_counts().index, df_filtered_2[c].value_counts(), label='Active', alpha=0.5, color="blue")
ax.legend(loc='best')
fig.tight_layout()
plt.savefig('plots/' + c + '.png')
plt.close(fig)
#plt.show()

years = ['Year 3', 'Year 2', 'Year 1']

fig, ax = plt.subplots()
#ax.set_title('Average Rating')
#print(df_filtered['Employee_Rating_3_W'].value_counts(), df_filtered['Employee_Rating_3_W'].mean())
#print(df_filtered['Employee_Rating_2_W'].value_counts(), df_filtered['Employee_Rating_2_W'].mean())
#print(df_filtered_1['Employee_Rating_1_W'].value_counts(), df_filtered_1['Employee_Rating_1_W'].mean())
#print(df_filtered_2['Employee_Rating_1_W'].value_counts(), df_filtered_2['Employee_Rating_1_W'].mean())
#print(df_filtered['Employee_Rating_1_W'].value_counts(), df_filtered['Employee_Rating_1_W'].mean())
#print(stats.ttest_ind(df_filtered_1['Employee_Rating_3_W'], df_filtered_1['Employee_Rating_2_W'], equal_var=False))

plt.plot(years,
         [df_filtered_1['Employee_Rating_3_W'].mean(), df_filtered_1['Employee_Rating_2_W'].mean(), df_filtered_1['Employee_Rating_1_W'].mean()],
         label='Resigned', color="red")
plt.plot(years,
         [df_filtered_2['Employee_Rating_3_W'].mean(), df_filtered_2['Employee_Rating_2_W'].mean(), df_filtered_2['Employee_Rating_1_W'].mean()],
         label='Active', color="blue")
plt.legend(loc='best')
plt.title('Average_Rating_E_W')
plt.savefig('plots/Average_Rating_E_W.png')
plt.close(fig)

fig, ax = plt.subplots()
plt.plot(years,
         [df_filtered_1['Employee_Rating_3_H'].mean(), df_filtered_1['Employee_Rating_2_H'].mean(), df_filtered_1['Employee_Rating_1_H'].mean()],
         label='Resigned', color="red")
plt.plot(years,
         [df_filtered_2['Employee_Rating_3_H'].mean(), df_filtered_2['Employee_Rating_2_H'].mean(), df_filtered_2['Employee_Rating_1_H'].mean()],
         label='Active', color="blue")
plt.legend(loc='best')
plt.title('Average_Rating_E_H')
plt.savefig('plots/Average_Rating_E_H.png')
plt.close(fig)

fig, ax = plt.subplots()
plt.plot(years,
         [df_filtered_1['Manager_Rating_3_W'].mean(), df_filtered_1['Manager_Rating_2_W'].mean(), df_filtered_1['Manager_Rating_1_W'].mean()],
         label='Resigned', color="red")
plt.plot(years,
         [df_filtered_2['Manager_Rating_3_W'].mean(), df_filtered_2['Manager_Rating_2_W'].mean(), df_filtered_2['Manager_Rating_1_W'].mean()],
         label='Active', color="blue")
plt.legend(loc='best')
plt.title('Average_Rating_M_W')
plt.savefig('plots/Average_Rating_M_W.png')
plt.close(fig)

fig, ax = plt.subplots()
plt.plot(years,
         [df_filtered_1['Manager_Rating_3_H'].mean(), df_filtered_1['Manager_Rating_2_H'].mean(), df_filtered_1['Manager_Rating_1_H'].mean()],
         label='Resigned', color="red")
plt.plot(years,
         [df_filtered_2['Manager_Rating_3_H'].mean(), df_filtered_2['Manager_Rating_2_H'].mean(), df_filtered_2['Manager_Rating_1_H'].mean()],
         label='Active', color="blue")
plt.legend(loc='best')
plt.title('Average_Rating_M_H')
plt.savefig('plots/Average_Rating_M_H.png')
plt.close(fig)

