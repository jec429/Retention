import pandas as pd
import numpy as np


def split_files():
    df_original = pd.read_excel('Brazil_Retention_Dataset_07192019.xlsx', sheet_name='Data')
    df_original.info()
    print(df_original['Report_Date'].str.split('/').head())

    df_2019 = df_original[df_original['Report_Date'].str.contains('2018')]
    df_2019.info()
    df_2018 = df_original[df_original['Report_Date'].str.contains('2017')]
    df_2018.info()
    df_2017 = df_original[df_original['Report_Date'].str.contains('2016')]
    df_2017.info()
    df_2016 = df_original[df_original['Report_Date'].str.contains('2015')]
    df_2016.info()

    df_2019.to_csv('Brazil_2019.csv', sep=',', encoding='utf-8')
    df_2018.to_csv('Brazil_2018.csv', sep=',', encoding='utf-8')
    df_2017.to_csv('Brazil_2017.csv', sep=',', encoding='utf-8')
    df_2016.to_csv('Brazil_2016.csv', sep=',', encoding='utf-8')
    # df_2015.to_csv('Brazil_2015.csv', sep=',', encoding='utf-8')


def clean_dataframe(year):
    from datetime import datetime
    df = pd.read_csv('Brazil_'+year+'.csv', sep=',')
    df2 = pd.DataFrame()
    df2['WWID'] = df['WWID']
    df2['Termination_Reason'] = df['Termination_Reason']

    if year == '2016':
        df2['Planned_as_a___of_Bonus_Tar'] = df['_016_Planned_as_a___of_Bonus_Tar']
        df2['Mgr_Change'] = df['Mgr_Change_2016'].map(lambda x: x > 0)
    elif year == '2017':
        df2['Planned_as_a___of_Bonus_Tar'] = df['_017_Planned_as_a___of_Bonus_Tar']
        df2['Mgr_Change'] = df['Mgr_Change_2017'].map(lambda x: x > 0)
    elif year == '2018':
        df2['Planned_as_a___of_Bonus_Tar'] = df['_018_Planned_as_a___of_Bonus_Tar']
        df2['Mgr_Change'] = df['Mgr_Change_2018'].map(lambda x: x > 0)

    for c in ['Compensation_Range___Midpoint', 'Total_Base_Pay___Local', 'Job_Sub_Function__IA__Host_All_O',
              'Length_of_Service_in_Years_inclu', 'Job_Function__IA__Host_All_Other',
              'Promotion', 'Demotion', 'Lateral',
              # 'Cross_Move', 'Trainings_Completed',
              # 'Mgr_Change_YN',  'SkipLevel_Mgr_Change',
              'Rehire_YN',
              # '_018_Planned_as_a___of_Bonus_Tar','_017_Planned_as_a___of_Bonus_Tar','_016_Planned_as_a___of_Bonus_Tar',
              'Highest_Degree_Received',
              # 'Actual_Sales_Incentive__2016', 'Actual_Sales_Incentive__2017',
              # 'Actual_Sales_Incentive__2018', 'Target_Sales_Incentive__2016',
              # 'Target_Sales_Incentive__2017', 'Target_Sales_Incentive__2018',
              'Hire_Date__Most_Recent_', 'Termination_Date']:
        df2[c] = df[c]

    for EM in ['Employee', 'Manager']:
        df2[EM+'_Rating_1'] = df[EM+'_Rating_1']

    df_filtered = df2.query('Termination_Reason != "End of Contract/Assignment Completed"')

    if True:
        df_filtered = df_filtered.assign(Compensation_Range___Midpoint=pd.Series(df_filtered['Compensation_Range___Midpoint'].replace(0, 1e9)).values)
        df_filtered['Compa_Diff_Ratio'] = (df_filtered['Total_Base_Pay___Local']-df_filtered['Compensation_Range___Midpoint'])\
                                          / df_filtered['Compensation_Range___Midpoint']
        df_filtered['Compa_Ratio'] = df_filtered['Total_Base_Pay___Local']/df_filtered['Compensation_Range___Midpoint']

        #df_filtered['Sales_Incentive_2016'] = df_filtered['Actual_Sales_Incentive__2016'] - \
        #                                      df_filtered['Target_Sales_Incentive__2016']
        #df_filtered['Sales_Incentive_2017'] = df_filtered['Actual_Sales_Incentive__2017'] - \
        #                                      df_filtered['Target_Sales_Incentive__2017']
        #df_filtered['Sales_Incentive_2018'] = df_filtered['Actual_Sales_Incentive__2018'] - \
        #                                      df_filtered['Target_Sales_Incentive__2018']

        #df_filtered = df_filtered.drop(['Actual_Sales_Incentive__2016', 'Actual_Sales_Incentive__2017',
        #                                'Actual_Sales_Incentive__2018', 'Target_Sales_Incentive__2016',
        #                                'Target_Sales_Incentive__2017', 'Target_Sales_Incentive__2018'], axis=1)

        for EM in ['Employee', 'Manager']:
            c = str(1)
            df_filtered[EM+'_Rating_'+c+'_W'] = df_filtered[EM+'_Rating_'+c].str.split('/').str.get(0).str.strip()
            df_filtered[EM+'_Rating_'+c+'_H'] = df_filtered[EM+'_Rating_'+c].str.split('/').str.get(1).str.strip()
            df_filtered = df_filtered.drop(EM+'_Rating_' + c, axis=1)

            for s in ['W', 'H']:
                print(df_filtered[EM + '_Rating_' + c + '_'+s].value_counts())
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
                    'Insufficient Data to Rate', 0)
                print('Cleaned')
                print(df_filtered[EM + '_Rating_' + c + '_' + s].value_counts())

    tenure = []
    status = []

    for r, st, et, ten in zip(df_filtered['Termination_Reason'],
                              df_filtered['Hire_Date__Most_Recent_'],
                              df_filtered['Termination_Date'],
                              df_filtered['Length_of_Service_in_Years_inclu']):
        if r == 'Resignation':
            d1 = datetime.strptime(st, "%Y-%m-%d")
            d2 = datetime.strptime(et, "%Y-%m-%d")
            tenure.append(abs((d2 - d1).days) / 365)
            status.append(1)
        else:
            tenure.append(ten)
            status.append(0)

    # df_filtered = df_filtered.assign(Tenure=pd.Series(tenure).values)
    df_filtered['Tenure'] = df_filtered['Length_of_Service_in_Years_inclu']
    df_filtered = df_filtered.assign(Status=pd.Series(status).values)

    df_filtered = df_filtered.drop(['Hire_Date__Most_Recent_', 'Termination_Date', 'Length_of_Service_in_Years_inclu',
                                    'Termination_Reason'],
                                   axis=1)
    df_filtered['Tenure_log'] = np.log(df_filtered['Tenure'] + 1)
    # df_filtered['Mgr_Change_YN'] = df_filtered['Mgr_Change_YN'].astype('category').cat.codes
    df_filtered.info()

    df_filtered_filtered = df_filtered[df_filtered['Status']==False]
    df_filtered_filtered.info()

    X_resigned_new = df_filtered[df_filtered['Status'] == True]
    X_resigned_new.info()

    resigs = X_resigned_new.WWID.values.tolist()
    print('Resignations:', len(resigs), resigs)

    df_filtered_filtered = df_filtered_filtered.set_index('WWID')
    for w in df_filtered_filtered.index:
        # print(w, df_filtered_filtered.at[w, 'Status'])
        if w in resigs:
            df_filtered_filtered.at[w, 'Status'] = 1

    df_filtered_filtered.to_csv(year + '_clean.csv', sep=',', encoding='utf-8')


def pickle_dataframe():
    from preprocessing import OneHotEncoder

    df_name = 'merged_Brazil_combined'
    df = pd.read_csv(df_name + '.csv', sep=',')
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[df.Job_Function__IA__Host_All_Other != 'Operations']
    data_x = df.drop(["Unnamed: 0"], axis=1)
    for c in data_x.columns:
        if data_x[c].dtype == object:
            data_x[c] = data_x[c].fillna('Missing')
            print(c)
            data_x[c] = data_x[c].astype('category')
        else:
            data_x[c] = data_x[c].fillna(-999)

    data_x = data_x.drop(["Job_Sub_Function__IA__Host_All_O"], axis=1)
    data_x['Rehire_YN'] = data_x['Rehire_YN'].cat.codes
    data_x.info()

    data_x_numeric = OneHotEncoder().fit_transform(data_x)

    print(data_x_numeric.head())

    for c in data_x_numeric.columns:
        if 'Missing' in c:
            print(c)
            data_x_numeric = data_x_numeric.drop(c, axis=1)

    data_x_numeric.info()
    data_x_numeric = data_x_numeric.fillna(-999)
    data_x_numeric.to_pickle("./"+df_name+"_x_numeric_new.pkl")
    # data_x_numeric.to_csv("./" + df_name + "_x_numeric_new.csv", sep=',', encoding='utf-8')


def fix_moves_by_year(y1, y2):
    from os import path
    import numpy

    year2 = str(y2) if y2 > y1 else str(y1)
    year1 = str(y1) if y2 > y1 else str(y2)

    if path.exists('Brazil_' + year1 + '_filtered.csv') and path.exists('Brazil_' + year2 + '_filtered.csv'):
        print('Moving from '+year2+' to '+year1)
        df1 = pd.read_csv('Brazil_' + year1 + '_filtered.csv', sep=',')
        df2 = pd.read_csv('Brazil_' + year2 + '_filtered.csv', sep=',')
    else:
        print('Files not available')
        return

    ws = [x for x in df2.WWID.values if x in df1.WWID.values]

    df1 = df1.set_index('WWID')
    df2 = df2.set_index('WWID')

    # df2.info()

    for w in df1.index:
        if w in ws:
            prom = df2.at[w, 'Promotion']
            demo = df2.at[w, 'Demotion']
            late = df2.at[w, 'Lateral']
            cros = df2.at[w, 'Cross_Move']
        else:
            prom = 0
            demo = 0
            late = 0
            cros = 0

        if type(prom) is numpy.ndarray:
            df1.at[w, 'Promotion'] = prom[0]
            df1.at[w, 'Demotion'] = demo[0]
            df1.at[w, 'Lateral'] = late[0]
            df1.at[w, 'Cross_Move'] = cros[0]
        else:
            df1.at[w, 'Promotion'] = prom
            df1.at[w, 'Demotion'] = demo
            df1.at[w, 'Lateral'] = late
            df1.at[w, 'Cross_Move'] = cros

    print(df1['Promotion'].head())
    print(df2['Promotion'].head())

    df1.to_csv('Brazil_' + year1 + '_filtered_shifted.csv', sep=',', encoding='utf-8')


def merge_files():
    # df_2015 = pd.read_csv('Brazil_2015_filtered_shifted.csv', sep=',')
    # df_2016 = pd.read_csv('Brazil_2016_filtered_shifted.csv', sep=',')
    # df_2017 = pd.read_csv('Brazil_2017_filtered_shifted.csv', sep=',')
    # df_2018 = pd.read_csv('Brazil_2018_filtered.csv', sep=',')

    df_2011 = pd.read_csv('2011_combined.csv', sep=',')
    df_2012 = pd.read_csv('2012_combined.csv', sep=',')
    df_2013 = pd.read_csv('2013_combined.csv', sep=',')
    df_2014 = pd.read_csv('2014_combined.csv', sep=',')
    df_2015 = pd.read_csv('2015_combined.csv', sep=',')
    df_2016 = pd.read_csv('2016_combined.csv', sep=',')
    df_2017 = pd.read_csv('2017_combined.csv', sep=',')
    df_2018 = pd.read_csv('2018_combined.csv', sep=',')
    # df_2019 = pd.read_csv('2019_combined.csv', sep=',')

    df_2011['Report_Year'] = 2011
    df_2012['Report_Year'] = 2012
    df_2013['Report_Year'] = 2013
    df_2014['Report_Year'] = 2014
    df_2015['Report_Year'] = 2015
    df_2016['Report_Year'] = 2016
    df_2017['Report_Year'] = 2017
    df_2018['Report_Year'] = 2018
    # df_2019['Report_Year'] = 2019

    df_merged = df_2011.append(df_2012, sort=True)
    df_merged = df_merged.append(df_2013, sort=True)
    df_merged = df_merged.append(df_2014, sort=True)
    df_merged = df_merged.append(df_2015, sort=True)
    df_merged = df_merged.append(df_2016, sort=True)
    df_merged = df_merged.append(df_2017, sort=True)
    df_merged = df_merged.append(df_2018, sort=True)
    # df_merged = df_merged.append(df_2019)
    df_merged.to_csv('merged_Brazil_combined.csv', sep=',', encoding='utf-8')


if __name__ == "__main__":
    # split_files()

    # clean_dataframe('2016')
    # clean_dataframe('2017')
    # clean_dataframe('2018')
    # clean_dataframe('2019')

    # clean_dataframe('Brazil_2017')
    # clean_dataframe('Brazil_2018')
    # clean_dataframe('Brazil_2019')

    # fix_moves_by_year(2015, 2016)
    # fix_moves_by_year(2016, 2017)
    # fix_moves_by_year(2017, 2018)

    # merge_files()
    pickle_dataframe()
