import pandas as pd
import re


def split_files():
    df_original = pd.read_excel('./data_files/Multiple year-Director+ - 2015 data added_1.xlsx', sheet_name='Data')
    df_original = df_original.drop('Termination_Date', axis=1)
    df_original.info()
    print(df_original['Report_Date'].str.split('/').head())

    df_2020 = df_original[df_original['Report_Date'].str.contains('2019')]
    df_2020.info()
    df_2019 = df_original[df_original['Report_Date'].str.contains('2018')]
    df_2019.info()
    df_2018 = df_original[df_original['Report_Date'].str.contains('2017')]
    df_2018.info()
    df_2017 = df_original[df_original['Report_Date'].str.contains('2016')]
    df_2017.info()
    df_2016 = df_original[df_original['Report_Date'].str.contains('2015')]
    df_2016.info()

    df_2020.to_csv('./data_files/D2Plus/D2Plus_2020.csv', sep=',', encoding='utf-8')
    df_2019.to_csv('./data_files/D2Plus/D2Plus_2019.csv', sep=',', encoding='utf-8')
    df_2018.to_csv('./data_files/D2Plus/D2Plus_2018.csv', sep=',', encoding='utf-8')
    df_2017.to_csv('./data_files/D2Plus/D2Plus_2017.csv', sep=',', encoding='utf-8')
    df_2016.to_csv('./data_files/D2Plus/D2Plus_2016.csv', sep=',', encoding='utf-8')
    # df_2015.to_csv('Brazil_2015.csv', sep=',', encoding='utf-8')


def clean_single_dataframe():
    from preprocessing import OneHotEncoder
    from datetime import datetime
    df = pd.read_excel('./data_files/EnterpriseD1datawithLTI.xlsx', sheet_name='Data')

    df2 = pd.DataFrame()

    df['Employee_Direct_Reports'] = df['Employee_Direct_Reports'].fillna(0)
    df['Manager_Direct_Reports'] = df['Manager_Direct_Reports'].fillna(0)
    df = df[~df['Compa_Ratio'].isnull()]

    to_drop = ['Termination_Date', 'Legal_Name', 'First_Name', 'Last_Name']
    to_drop.append('Position__IA__Host_All_Other__Pr')
    to_drop.append('Position_ID__IA__Host_All_Other_')
    to_drop.append('Hire_Date__Most_Recent_')
    to_drop.append('Worker_Active')
    to_drop.append('Manager_Name__IA__Host_All_Other')
    to_drop.append('Manager_WWID__IA__Host_All_Other')
    to_drop.append('Supervisory_Organization___Hiera')
    to_drop.append('Original_Hire_Date')
    to_drop.append('Year_Month')
    to_drop.append('Report_Date')

    df = df.drop(to_drop, axis=1)

    for c in df.columns:
        if df[~df[c].isna()][c].shape[0] < 7000:
            print('Missing=', c, df[~df[c].isna()][c].shape[0])
        else:
            if df[c].dtype == object:
                df[c] = df[c].fillna('Missing')
                # print(c)
                df[c] = df[c].astype('category')
            else:
                df[c] = df[c].fillna(-1)
            if len(df[c].unique()) == 1:
                continue

            print(c, len(df[c].unique()), df[c].dtype)
            df2[c] = df[c]

    # df2['WWID'] = df['WWID']
    # df2['Termination_Reason'] = df['Termination_Reason']
    # df2['Compa_Ratio'] = df['Compa_Ratio']
    # df2['Total_Base_Pay'] = df['Total_Base_Pay___US']

    df2['Termination_Category'] = df2['Termination_Category'].cat.codes
    df2.to_csv('./data_files/D2Plus/D2Plus_clean.csv', sep=',', encoding='utf-8')

    data_x_numeric = OneHotEncoder().fit_transform(df2)
    data_x_numeric.info()
    data_x_numeric = data_x_numeric.fillna(-999)
    data_x_numeric.to_pickle("./data_files/D2Plus/D2Plus_clean_x_numeric_newer.pkl")
    df3 = data_x_numeric[:100]
    df3.to_csv("./data_files/D2Plus/D2Plus_clean_x_numeric_newer.csv", sep=',', encoding='utf-8')


def clean_dataframe(year):
    df = pd.read_csv('./data_files/D2Plus/D2Plus_' + year + '.csv', sep=',')
    df_2020 = pd.read_csv('./data_files/D2Plus/D2Plus_2020.csv', sep=',')
    df = df[df['Employee_Pay_Grade'] >= 20]
    print(df.shape[0])
    df = df[df['Employee_Type__IA__Host_All_Othe'].str.contains('Fixed Term') == False]
    df = df[df['Employee_Type__IA__Host_All_Othe'].str.contains('Intern') == False]
    df['Employee_Direct_Reports'] = df['Employee_Direct_Reports'].fillna(0)
    df['Manager_Direct_Reports'] = df['Manager_Direct_Reports'].fillna(0)
    df = df[~df['Compa_Ratio'].isnull()]
    print(df.shape[0])
    df.info()

    to_drop = ['Termination_Reason', 'Legal_Name', 'First_Name', 'Last_Name']
    to_drop.append('Position__IA__Host_All_Other__Pr')
    to_drop.append('Position_ID__IA__Host_All_Other_')
    to_drop.append('Hire_Date__Most_Recent_')
    to_drop.append('Worker_Active')
    to_drop.append('Manager_Name__IA__Host_All_Other')
    to_drop.append('Manager_WWID__IA__Host_All_Other')
    to_drop.append('Supervisory_Organization___Hiera')
    to_drop.append('Original_Hire_Date')
    to_drop.append('Year_Month')
    to_drop.append('Report_Date')
    to_drop.append('Job_Profile__IA__Host_All_Other_')
    to_drop.append('Job_Sub_Function__IA__Host_All_O')
    to_drop.append('Legal_Entity_Description__IA__Ho')
    to_drop.append('MRC_Description__IA__Host_All_Ot')
    to_drop.append('Target_Job_1_Title')
    to_drop.append('Active_Status')
    to_drop.append('Promotion')
    to_drop.append('Promotion_YN')
    to_drop.append('Demotion')
    to_drop.append('Demotion_YN')
    to_drop.append('Lateral')
    to_drop.append('Lateral_YN')
    to_drop.append('Cross_Move')
    to_drop.append('Cross_Move_YN')
    to_drop.append('Mgr_Change')
    to_drop.append('Mgr_Change_YN')
    to_drop.append('SkipLevel_Mgr_Change')
    to_drop.append('SkipLevel_Mgr_Change_YN')
    to_drop.append('Working_Country__IA__Host_All_Ot')
    to_drop.append('School_Name')
    to_drop.append('School_Location')
    to_drop.append('School_Type')
    to_drop.append('Country')
    to_drop.append('Degree')
    to_drop.append('Degree_Received')
    to_drop.append('Education')
    to_drop.append('Highest_Degree_Received')
    to_drop.append('Is_Highest_Degree_Received_')
    to_drop.append('Skill_Reference_ID')
    to_drop.append('Start_End')
    to_drop.append('Worker')
    to_drop.append('Worker_Type')
    to_drop.append('Total_Base_Pay___Local___Currenc')
    to_drop.append('Total_Salary_and_Allowances___Cu')
    to_drop.append('Working_Location_Address__IA__Ho')
    to_drop.append('Working_Location_Postal_Code__IA')
    to_drop.append('Stock_Salary___Currency')
    to_drop.append('Primary_Compensation_Basis___Cur')
    to_drop.append('Home_Location_Code__Primary_')
    to_drop.append('Bonus_Salary___Local___Currency')
    to_drop.append('Compensation_Grade_Profile_Ref_I')
    to_drop.append('Home_Country__Primary_')
    to_drop.append('Flag_For_Catalano')
    to_drop.append('Primary_Position_is_Job_Exempt_')
    to_drop.append('Region__IA__Host_All_Other__Prim')
    to_drop.append('Sector__IA__Host_All_Other__Prim')
    to_drop.append('MRC_Code__IA__Host_All_Other__Pr')
    to_drop.append('Legal_Entity_Code__IA__Host_All_')

    df = df.drop(to_drop, axis=1)

    df2 = pd.DataFrame()
    year_s = str(int(year) - 1)[2:]
    for c in df.columns:
        c_c = c
        if bool(re.search(r'\d', c)):
            print('Integer=', c)
            if year_s in c:
                print(year_s, c)
                c_c = c.replace(year_s, 'XX')
            else:
                continue

        if df[~df[c].isna()][c].shape[0] < 5000:
            # print('Missing=', c, df[~df[c].isna()][c].shape[0])
            continue
        else:
            if df[c].dtype == object:
                df[c] = df[c].fillna('Missing')
                # print(c)
                df[c] = df[c].astype('category')
            else:
                df[c] = df[c].fillna(-1)
            if len(df[c].unique()) == 1:
                continue

            # print(c, len(df[c].unique()), df[c].dtype)
            df2[c_c] = df[c]

    df2['Termination_Category'] = df2['Termination_Category'].cat.codes
    df2['Working_Country_Fixed'] = df2['Working_Country_Fixed'].cat.codes

    df_filtered2 = df2[df2['Termination_Category'] == 0]
    # df_filtered_filtered.info()

    x_resigned_new = df2[df2['Termination_Category'] == 1]
    # x_resigned_new.info()

    resigs = x_resigned_new.WWID.values.tolist()
    # print('Resignations:', len(resigs), resigs)

    df_filtered2 = df_filtered2.set_index('WWID')
    for w in df_filtered2.index:
        # print(w, df_filtered_filtered.at[w, 'Status'])
        if w in resigs:
            df_filtered2.at[w, 'Termination_Category'] = 1

    df_filtered2['WWID'] = df_filtered2.index
    df_filtered2['Location_Code__IA__Host_All_Othe'] = df_filtered2['Location_Code__IA__Host_All_Othe'].cat.codes
    df_filtered2.to_csv('./data_files/D2Plus/D2Plus_' + year + '_clean.csv', sep=',', encoding='utf-8')
    df_filtered2.to_pickle('./data_files/D2Plus/D2Plus_' + year + '_clean.pkl')


def merge_files():
    df_2016 = pd.read_pickle('data_files/D2Plus/D2Plus_2016_clean.pkl')
    df_2017 = pd.read_pickle('data_files/D2Plus/D2Plus_2017_clean.pkl')
    df_2018 = pd.read_pickle('data_files/D2Plus/D2Plus_2018_clean.pkl')
    df_2019 = pd.read_pickle('data_files/D2Plus/D2Plus_2019_clean.pkl')

    df_2016['Report_Year'] = 2016
    df_2017['Report_Year'] = 2017
    df_2018['Report_Year'] = 2018
    df_2019['Report_Year'] = 2019

    df_merged = df_2016.append(df_2017, sort=True)
    df_merged = df_merged.append(df_2018, sort=True)
    df_merged = df_merged.append(df_2019, sort=True)
    df_merged.to_pickle("./data_files/D2Plus/merged_D2Plus_combined_fixed.pkl")
    df_merged.to_csv('data_files/D2Plus/merged_D2Plus_combined_fixed.csv', sep=',', encoding='utf-8')


def pickle_dataframe():
    from preprocessing import OneHotEncoder

    df_name = 'merged_D2Plus_combined_fixed'
    df = pd.read_pickle('data_files/D2Plus/' + df_name + '.pkl')
    df = df.sample(frac=1).reset_index(drop=True)
    print('size=', df.shape[0])
    to_drop = []
    to_drop.append('_0XX_Target_Bonus_Percent___WD_C')
    to_drop.append('_0XX_Target_LTI_Percent___WD_Cal')
    to_drop.append('_0XX_Target_Merit_Percent')
    to_drop.append('_0XX_LTI_Eligible')
    to_drop.append('_0XX_Planned_Bonus_Percent')
    to_drop.append('_0XX_Planned_LTI_Percent')
    to_drop.append('_0XX_Planned_Merit_Percent')
    to_drop.append('Cross_Move_20XX')


    data_x = df.drop(to_drop, axis=1)

    for c in data_x.columns:
        if str(data_x.dtypes[c]) == 'category':
            print(c)
        elif data_x[c].dtype == object:
            data_x[c] = data_x[c].fillna('Missing')
            print(c)
            data_x[c] = data_x[c].astype('category')
        else:
            data_x[c] = data_x[c].fillna(-1)

    data_x.info()

    data_x_numeric = OneHotEncoder().fit_transform(data_x)

    for c in data_x_numeric.columns:
        if 'Missing' in c:
            data_x_numeric = data_x_numeric.drop(c, axis=1)

    print(data_x_numeric.head())

    data_x_numeric.info()
    data_x_numeric = data_x_numeric.fillna(-999)
    data_x_numeric.to_pickle("./data_files/D2Plus/"+df_name+"_x_numeric_newer.pkl")
    data_x_numeric2 = data_x_numeric[:100]
    data_x_numeric2.to_csv("./data_files/D2Plus/" + df_name + "_x_numeric_newer.csv", sep=',', encoding='utf-8')


if __name__ == '__main__':
    # split_files()
    # clean_single_dataframe()

    clean_dataframe('2016')
    clean_dataframe('2017')
    clean_dataframe('2018')
    clean_dataframe('2019')

    # clean_dataframe('2020')
    #
    # # fix_moves_by_year(2016, 2017)
    # # fix_moves_by_year(2017, 2018)
    # # fix_moves_by_year(2018, 2019)
    # # fix_moves_by_year(2019, 2020)
    #
    merge_files()
    pickle_dataframe()
