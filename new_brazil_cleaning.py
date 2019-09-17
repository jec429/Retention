from clean_old_data import *
from combine_clean import combine


def split_files():
    df_original = pd.read_excel('./data_files/BrazilRetentionDataset_07192019.xlsx', sheet_name='Data')
    df_original = df_original.drop('Termination_Date', axis=1)
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

    df_2019.to_csv('./data_files/BRAZIL/Brazil_2019.csv', sep=',', encoding='utf-8')
    df_2018.to_csv('./data_files/BRAZIL/Brazil_2018.csv', sep=',', encoding='utf-8')
    df_2017.to_csv('./data_files/BRAZIL/Brazil_2017.csv', sep=',', encoding='utf-8')
    df_2016.to_csv('./data_files/BRAZIL/Brazil_2016.csv', sep=',', encoding='utf-8')
    # df_2015.to_csv('Brazil_2015.csv', sep=',', encoding='utf-8')


def clean_dataframe(year):
    from datetime import datetime
    df = pd.read_csv('data_files/BRAZIL/Brazil_'+year+'.csv', sep=',')
    df = df[df['Employee_Pay_Grade'] > 20]
    df.info()
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
    elif year == '2019':
        df2['Planned_as_a___of_Bonus_Tar'] = 0
        df2['Mgr_Change'] = df['Mgr_Change_2018'].map(lambda x: x > 0)
    elif year == '2020':
        df2['Planned_as_a___of_Bonus_Tar'] = 0
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
              'Hire_Date__Most_Recent_',  # 'Termination_Date',
              'Working_Country_Fixed',
              'Location_Code__IA__Host_All_Othe',
              'Manager_WWID__IA__Host_All_Other']:
        df2[c] = df[c]

    for EM in ['Employee', 'Manager']:
        df2[EM + '_Rating_1'] = df[EM + '_Rating_1']
        df2[EM + '_Rating_2'] = df[EM + '_Rating_2']
        df2[EM + '_Rating_3'] = df[EM + '_Rating_3']

    df_filtered = df2.query('Termination_Reason != "End of Contract/Assignment Completed"')

    if True:
        df_filtered = df_filtered.assign(Compensation_Range___Midpoint=pd.Series(
            df_filtered['Compensation_Range___Midpoint'].replace(0, 1e9)).values)
        df_filtered['Compa_Diff_Ratio'] = \
            (df_filtered['Total_Base_Pay___Local'] -
             df_filtered['Compensation_Range___Midpoint']) / df_filtered['Compensation_Range___Midpoint']
        df_filtered['Compa_Ratio'] = df_filtered['Total_Base_Pay___Local']/df_filtered['Compensation_Range___Midpoint']

        # df_filtered['Sales_Incentive_2016'] = df_filtered['Actual_Sales_Incentive__2016'] - \
        #                                      df_filtered['Target_Sales_Incentive__2016']
        # df_filtered['Sales_Incentive_2017'] = df_filtered['Actual_Sales_Incentive__2017'] - \
        #                                      df_filtered['Target_Sales_Incentive__2017']
        # df_filtered['Sales_Incentive_2018'] = df_filtered['Actual_Sales_Incentive__2018'] - \
        #                                      df_filtered['Target_Sales_Incentive__2018']

        # df_filtered = df_filtered.drop(['Actual_Sales_Incentive__2016', 'Actual_Sales_Incentive__2017',
        #                                'Actual_Sales_Incentive__2018', 'Target_Sales_Incentive__2016',
        #                                'Target_Sales_Incentive__2017', 'Target_Sales_Incentive__2018'], axis=1)

        for EM in ['Employee', 'Manager']:
            for c in ['1', '2', '3']:
                # c = str(1)
                # df_filtered[EM+'_Rating_'+c+'_W'] = df_filtered[EM+'_Rating_'+c].str.split('/').str.get(0).str.strip()
                # df_filtered[EM+'_Rating_'+c+'_H'] = df_filtered[EM+'_Rating_'+c].str.split('/').str.get(1).str.strip()
                # df_filtered = df_filtered.drop(EM+'_Rating_' + c, axis=1)

                print(df_filtered[EM + '_Rating_' + c].value_counts())

                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace('1', 1)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace('2', 1)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace('3', 1)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace('4', 1)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace('5', 1)

                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace('6', 2)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace('7', 2)

                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace('8', 3)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace('9', 3)

                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Insufficient Data to Rate / Insufficient Data to Rate', 0)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Does Not Meet / Partially Meets', 1)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Does Not Meet / Fully Meets', 1)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Fully Meets / Partially Meets', 1)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Partially Meets / Partially Meets', 1)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Partially Meets / Does Not Meet', 1)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Does Not Meet / Does Not Meet', 1)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Partially Meets / Fully Meets', 1)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Fully Meets / Does Not Meet', 1)

                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Fully Meets / Fully Meets', 2)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Fully Meets / Exceeds', 2)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Exceeds / Fully Meets', 2)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Partially Meets / Exceeds', 2)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Exceeds / Partially Meets', 2)
                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Exceeds / Does Not Meet', 2)

                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Exceeds / Exceeds', 3)
                print('Cleaned')
                print(df_filtered[EM + '_Rating_' + c].value_counts())

                '''
                for s in ['W', 'H']:
                    # print(df_filtered[EM + '_Rating_' + c + '_'+s].value_counts())
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
                '''

    tenure = []
    status = []

    for r, st, ten in zip(df_filtered['Termination_Reason'],
                              df_filtered['Hire_Date__Most_Recent_'],
                              # df_filtered['Termination_Date'],
                              df_filtered['Length_of_Service_in_Years_inclu']):
        if r == 'Resignation' and year != '2019':
            d1 = datetime.strptime(st, "%Y-%m-%d")
            # d2 = datetime.strptime(et, "%Y-%m-%d")
            # tenure.append(abs((d2 - d1).days) / 365)
            status.append(1)
        else:
            tenure.append(ten)
            status.append(0)

    # df_filtered = df_filtered.assign(Tenure=pd.Series(tenure).values)
    df_filtered['Tenure'] = df_filtered['Length_of_Service_in_Years_inclu']
    df_filtered = df_filtered.assign(Status=pd.Series(status).values)

    df_filtered = df_filtered.drop(['Hire_Date__Most_Recent_', 'Length_of_Service_in_Years_inclu',
                                    'Termination_Reason'],
                                   axis=1)
    df_filtered['Tenure_log'] = np.log(df_filtered['Tenure'] + 1)
    # df_filtered['Mgr_Change_YN'] = df_filtered['Mgr_Change_YN'].astype('category').cat.codes
    df_filtered.info()

    # df_filtered['Working_Country'] = 37

    manager_manager = []
    for mw in df_filtered['Manager_WWID__IA__Host_All_Other']:
        man_man = 0
        for w, mmw in zip(df_filtered['WWID'], df_filtered['Manager_WWID__IA__Host_All_Other']):
            if mw == w:
                man_man = mmw
                break
        manager_manager.append(man_man)

    df_filtered['Manager_Manager_WWID'] = manager_manager

    df_filtered_filtered = df_filtered[df_filtered['Status'] == 0]
    df_filtered_filtered.info()

    x_resigned_new = df_filtered[df_filtered['Status'] == 1]
    x_resigned_new.info()

    resigs = x_resigned_new.WWID.values.tolist()
    print('Resignations:', len(resigs), resigs)

    df_filtered_filtered = df_filtered_filtered.set_index('WWID')
    for w in df_filtered_filtered.index:
        # print(w, df_filtered_filtered.at[w, 'Status'])
        if w in resigs:
            df_filtered_filtered.at[w, 'Status'] = 1

    print('Filtered')
    df_filtered_filtered.info()

    if year == '2020':
        df_filtered_filtered['Skip_Manager_Change'] = 0
        df_filtered_filtered.to_csv('data_files/BRAZIL/' + year + '_clean.csv', sep=',', encoding='utf-8')
    else:
        df_filtered_filtered.to_csv('data_files/BRAZIL/' + year + '_pre_clean.csv', sep=',', encoding='utf-8')


def pickle_dataframe():
    from preprocessing import OneHotEncoder

    df_name = 'merged_Brazil_combined'
    df = pd.read_csv('data_files/BRAZIL/' + df_name + '.csv', sep=',')
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[df.Job_Function__IA__Host_All_Other != 'Operations']
    print('size=', df.shape[0])
    data_x = df.drop(["Unnamed: 0", "Manager_WWID__IA__Host_All_Other"], axis=1)
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
    data_x_numeric.to_pickle("./data_files/BRAZIL/"+df_name+"_x_numeric_newer.pkl")
    data_x_numeric.to_csv("./data_files/BRAZIL/" + df_name + "_x_numeric_newer.csv", sep=',', encoding='utf-8')


def fix_moves_by_year(y1, y2):
    from os import path
    import numpy

    year2 = str(y2) if y2 > y1 else str(y1)
    year1 = str(y1) if y2 > y1 else str(y2)

    if year2 == '2020':
        if path.exists('data_files/BRAZIL/' + year1 + '_pre_clean.csv') and path.exists(
                'data_files/BRAZIL/' + year2 + '_clean.csv'):
            print('Moving from ' + year2 + ' to ' + year1)
            df1 = pd.read_csv('data_files/BRAZIL/' + year1 + '_pre_clean.csv', sep=',')
            df2 = pd.read_csv('data_files/BRAZIL/' + year2 + '_clean.csv', sep=',')
        else:
            print('Moving from ' + year2 + ' to ' + year1)
            print('Files not available')
            return
    else:
        if path.exists('data_files/BRAZIL/' + year1 + '_pre_clean.csv') and path.exists('data_files/BRAZIL/' + year2 + '_pre_clean.csv'):
            print('Moving from '+year2+' to '+year1)
            df1 = pd.read_csv('data_files/BRAZIL/' + year1 + '_pre_clean.csv', sep=',')
            df2 = pd.read_csv('data_files/BRAZIL/' + year2 + '_pre_clean.csv', sep=',')
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
            # cros = df2.at[w, 'Cross_Move']
        else:
            prom = 0
            demo = 0
            late = 0
            # cros = 0

        if type(prom) is numpy.ndarray:
            df1.at[w, 'Promotion'] = prom[0]
            df1.at[w, 'Demotion'] = demo[0]
            df1.at[w, 'Lateral'] = late[0]
            # df1.at[w, 'Cross_Move'] = cros[0]
        else:
            df1.at[w, 'Promotion'] = prom
            df1.at[w, 'Demotion'] = demo
            df1.at[w, 'Lateral'] = late
            # df1.at[w, 'Cross_Move'] = cros

    print(df1['Promotion'].head())
    print(df2['Promotion'].head())

    skip_manager_change = []
    for w1 in df1.index:
        smc = 0
        for w2 in df2.index:
            if w1 == w2:
                # print('Managers', df1.at[w1, 'Manager_Manager_WWID'], df2.at[w2, 'Manager_Manager_WWID'])
                if df1.at[w1, 'Manager_Manager_WWID'] != df2.at[w2, 'Manager_Manager_WWID'].any():
                    smc = 1
        skip_manager_change.append(smc)

    df1['Skip_Manager_Change'] = skip_manager_change
    df1 = df1.drop(['Manager_Manager_WWID'], axis=1)

    df1.to_csv('data_files/BRAZIL/' + year1 + '_clean.csv', sep=',', encoding='utf-8')
    if year2 == '2020':
        df2 = df2.drop(['Manager_Manager_WWID'], axis=1)
        df2.to_csv('data_files/BRAZIL/' + year2 + '_clean.csv', sep=',', encoding='utf-8')


def merge_files():
    # df_2015 = pd.read_csv('Brazil_2015_filtered_shifted.csv', sep=',')
    # df_2016 = pd.read_csv('Brazil_2016_filtered_shifted.csv', sep=',')
    # df_2017 = pd.read_csv('Brazil_2017_filtered_shifted.csv', sep=',')
    # df_2018 = pd.read_csv('Brazil_2018_filtered.csv', sep=',')

    df_2016 = pd.read_csv('data_files/BRAZIL/2016_clean.csv', sep=',')
    df_2017 = pd.read_csv('data_files/BRAZIL/2017_clean.csv', sep=',')
    df_2018 = pd.read_csv('data_files/BRAZIL/2018_clean.csv', sep=',')
    # df_2019 = pd.read_csv('data_files/BRAZIL/2019_combined.csv', sep=',')

    df_2016['Report_Year'] = 2016
    df_2017['Report_Year'] = 2017
    df_2018['Report_Year'] = 2018
    # df_2019['Report_Year'] = 2019

    df_merged = df_2016.append(df_2017, sort=True)
    df_merged = df_merged.append(df_2018, sort=True)
    # df_merged = df_merged.append(df_2019, sort=True)
    df_merged.to_csv('data_files/BRAZIL/merged_Brazil_combined.csv', sep=',', encoding='utf-8')


if __name__ == '__main__':
    split_files()
    # # write_to_pickle(year)
    #
    clean_dataframe('2016')
    clean_dataframe('2017')
    clean_dataframe('2018')
    clean_dataframe('2019')
    # clean_dataframe('2020')

    fix_moves_by_year(2016, 2017)
    fix_moves_by_year(2017, 2018)
    fix_moves_by_year(2018, 2019)
    fix_moves_by_year(2019, 2020)

    # combine(2016)
    # combine(2017)
    # combine(2018)
    # combine(2019)

    merge_files()
    pickle_dataframe()
