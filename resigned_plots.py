import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

fname = "parrot.pkl"
with open(fname, "rb") as fin:
    list_lists2 = pickle.load(fin)
wwids = list_lists2[0].to_list()
prob_mlp = list_lists2[1]
prob_tf = list_lists2[2]

x_bra = pd.read_csv('data_files/Brazil_2018.csv', sep=',')
x_bra = x_bra.drop_duplicates(subset='WWID', keep="first")
# print(x_sea.columns)
# print(x_bra.columns)
to_drop = ['Unnamed: 0', 'Legal_Name', 'First_Name', 'Last_Name', 'Home_Country__Primary_', 'Is_International_Assignee',
           'Position_ID__IA__Host_All_Other_', 'Job_Profile_ID__IA__Host_All_Oth', 'Time_Type__IA__Host_All_Other__P',
           'Legal_Entity_Code__IA__Host_All_', 'MRC_Code__IA__Host_All_Other__Pr', 'Manager_WWID__IA__Host_All_Other',
           'Manager_Name__IA__Host_All_Other', 'Workday_Function__IA__Host_All_O', 'Termination_Category',
           'MRC_Description__IA__Host_All_Ot', 'MRC_Code__for_Headcount_Reportin', 'MRC_Code_Description__for_Headco',
           'Time_in_Position__IA__Host_All_O', 'Time_in_Position__Core_or_Core_V', 'Working_Country_Fixed',
           'Region_Fixed', 'Report_Type', 'Report_Date', 'Start_End', 'Worker_Active', 'Target_Job_2_Title',
           'Primary_Compensation_Basis___Fre', 'Acquired_Company', 'Acquisition_Date', 'Hourly_Salaried',
           'Working_Location_Address__IA__Ho', 'Working_Location_Postal_Code__IA', 'Year_Month', 'Original_Hire_Date',
           'Working_Country__IA__Host_All_Ot', 'Region__IA__Host_All_Other__Prim', 'Primary_Position_is_Job_Exempt_',
           'Flag_For_Catalano', 'Home_Address_1_Postal_Code', 'Worker', 'Worker_Type', 'Target_Job_1_Title',
           'Position__IA__Host_All_Other__Pr', 'Supervisory_Organization___Hiera', 'Target_Job_3_Title',
           'Hire_Date__Most_Recent_', 'Job_Profile__IA__Host_All_Other_', 'Employee_Rating_Year_1',
           'Employee_Rating_Year_2', 'Employee_Rating_Year_3', 'Job_Profile_Grade__IA__Host_All_', 'Termination_Reason',
           'HR_Tier_Change', 'Compensation_Grade_Profile_Ref_I', 'Manager_Rating_1', 'Manager_Rating_2',
           'Manager_Rating_3', 'First_Year_Attended', 'Last_Year_Attended', 'Grade_Average', 'Year_Degree_Received'
           ]
x_bra = x_bra.drop(to_drop, axis=1)

resigned = []
for wwid in x_bra['WWID']:
    if wwid in wwids:
        index = wwids.index(wwid)
    else:
        continue
    prob = prob_tf[index]
    if prob > 0.4:
        resigned.append(wwid)

print(len(resigned))
x_bra_selected = x_bra[x_bra['WWID'].isin(resigned)]
x_bra_selected.info()
shared = x_bra_selected.columns.to_list()
# print(shared)
print(len(shared))
for h in shared:
    if h == 'WWID':
        continue
    print(h)

    if 'Compensation_Range__M' in h:
        x_bra_selected[h] = x_bra_selected[h].astype(float)

    if x_bra_selected[h].dtype == float or x_bra_selected[h].dtype == int:
        fig, ax = plt.subplots()
        x = [v for v in x_bra_selected[h].values if v > -998]
        ax.hist(x, 20, density=1, label='BRAZIL', alpha=0.5, color="blue")
        # plt.xlim(0.0, 1.0)
        ax.legend(loc='best')
        plt.xlabel(h)
    else:
        fig, ax = plt.subplots()
        x_bra_selected[h].value_counts().plot(kind='barh', color='blue', label='BRAZIL')
        ax.legend(loc='best')
        fig.subplots_adjust(left=0.4)
        plt.xlabel(h)

    plt.savefig('./plots/resigned/'+h+'.png')
    plt.close(fig)

text = '<!DOCTYPE html>\n<html>\n<body>\n'
for h in shared:
    if h == 'WWID':
        continue
    text += '<img src = "./plots/resigned/'+h+'.png" >\n'
print(text)

text += '</body>\n</html>\n'

f = open('index.html', 'w')
f.write(text)
f.close()

## SEA

fname = "parrot_sea.pkl"
with open(fname, "rb") as fin:
    list_lists2 = pickle.load(fin)
wwids = list(list_lists2[0])
prob_tf = list(list_lists2[1])

x_sea = pd.read_csv('data_files/SEA/Sea_2018.csv', sep=',')
x_sea = x_sea.drop_duplicates(subset='WWID', keep="first")
# print(x_sea.columns)
# print(x_sea.columns)
to_drop = ['Unnamed: 0', 'Legal_Name', 'First_Name', 'Last_Name', 'Home_Country__Primary_', 'Is_International_Assignee',
           'Position_ID__IA__Host_All_Other_', 'Job_Profile_ID__IA__Host_All_Oth', 'Time_Type__IA__Host_All_Other__P',
           'Legal_Entity_Code__IA__Host_All_', 'MRC_Code__IA__Host_All_Other__Pr', 'Manager_WWID__IA__Host_All_Other',
           'Manager_Name__IA__Host_All_Other', 'Workday_Function__IA__Host_All_O', 'Termination_Category',
           'MRC_Description__IA__Host_All_Ot', 'MRC_Code__for_Headcount_Reportin', 'MRC_Code_Description__for_Headco',
           'Time_in_Position__IA__Host_All_O', 'Time_in_Position__Core_or_Core_V', 'Working_Country_Fixed',
           'Region_Fixed', 'Report_Type', 'Report_Date', 'Start_End', 'Worker_Active', 'Target_Job_2_Title',
           'Primary_Compensation_Basis___Fre', 'Acquired_Company', 'Acquisition_Date', 'Hourly_Salaried',
           'Working_Location_Address__IA__Ho', 'Working_Location_Postal_Code__IA', 'Year_Month', 'Original_Hire_Date',
           'Working_Country__IA__Host_All_Ot', 'Region__IA__Host_All_Other__Prim', 'Primary_Position_is_Job_Exempt_',
           'Flag_For_Catalano', 'Home_Address_1_Postal_Code', 'Worker', 'Worker_Type', 'Target_Job_1_Title',
           'Position__IA__Host_All_Other__Pr', 'Supervisory_Organization___Hiera', 'Target_Job_3_Title',
           'Hire_Date__Most_Recent_', 'Job_Profile__IA__Host_All_Other_', 'Employee_Rating_Year_1',
           'Employee_Rating_Year_2', 'Employee_Rating_Year_3', 'Job_Profile_Grade__IA__Host_All_', 'Termination_Reason',
           'HR_Tier_Change', 'Compensation_Grade_Profile_Ref_I', 'Manager_Rating_1', 'Manager_Rating_2',
           'Manager_Rating_3', 'First_Year_Attended', 'Last_Year_Attended', 'Grade_Average', 'Year_Degree_Received'#,
           # 'Target_Job_1_Readiness', 'Target_Job_2_Readiness', 'Target_Job_3_Readiness', 'Potential_From_PG',
           # 'Employee_Rating_3', 'Stock_Plan_Name', 'Stock_Salary___Currency', 'YE_Comp_Wage_Ind',
           # 'Highest_Degree_Received', 'Education', 'Skill_Reference_ID', 'School_Name'
           ]
x_sea = x_sea.drop(to_drop, axis=1)
print('Sea')
x_sea.info()
resigned = []
for wwid in x_sea['WWID']:
    if wwid in wwids:
        index = wwids.index(wwid)
    else:
        continue
    prob = prob_tf[index]
    if prob > 0.4:
        resigned.append(wwid)

print(len(resigned))
x_sea_selected = x_sea[x_sea['WWID'].isin(resigned)]
# x_sea_selected = x_sea
print('Selected')
x_sea_selected.info()

shared = x_sea_selected.columns.to_list()
# print(shared)
print(len(shared))
for h in shared:
    if h == 'WWID':
        continue
    print(h)

    if 'Compensation_Range__M' in h:
        x_sea_selected[h] = x_sea_selected[h].astype(float)

    if x_sea_selected[h].dtype == float or x_sea_selected[h].dtype == int:
        x_sea_selected[h] = x_sea_selected[h].fillna(-999)
        fig, ax = plt.subplots()
        x = [v for v in x_sea_selected[h].values if v > -998]
        ax.hist(x, 20, density=1, label='SEA', alpha=0.5, color="blue")
        # plt.xlim(0.0, 1.0)
        ax.legend(loc='best')
        plt.xlabel(h)
    else:
        x_sea_selected[h] = x_sea_selected[h].fillna('Missing')
        fig, ax = plt.subplots()
        x_sea_selected[h].value_counts().plot(kind='barh', color='blue', label='SEA')
        ax.legend(loc='best')
        fig.subplots_adjust(left=0.4)
        plt.xlabel(h)

    plt.savefig('./plots/resigned/'+h+'_sea.png')
    plt.close(fig)

text = '<!DOCTYPE html>\n<html>\n<body>\n'
for h in shared:
    if h == 'WWID':
        continue
    text += '<img src = "./plots/resigned/'+h+'_sea.png" >\n'
# print(text)

text += '</body>\n</html>\n'

f = open('index_sea.html', 'w')
f.write(text)
f.close()
