from clean_old_data import *
from combine_clean import combine
import numpy


def split_files():
    df_original = pd.read_excel('./data_files/Global_SEA_Data_Multiple_year_Demo_2015 Added.xlsx', sheet_name='Data')
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

    df_2019.to_csv('./data_files/SEA/Sea_2019.csv', sep=',', encoding='utf-8')
    df_2018.to_csv('./data_files/SEA/Sea_2018.csv', sep=',', encoding='utf-8')
    df_2017.to_csv('./data_files/SEA/Sea_2017.csv', sep=',', encoding='utf-8')
    df_2016.to_csv('./data_files/SEA/Sea_2016.csv', sep=',', encoding='utf-8')
    # df_2015.to_csv('Brazil_2015.csv', sep=',', encoding='utf-8')


def clean_dataframe(year):
    from datetime import datetime
    df = pd.read_csv('data_files/SEA/Sea_'+year+'.csv', sep=',')
    df_2020 = pd.read_excel('./data_files/SEA/SEA_Retention_Data_Jan-Sept2019_2018 changes added.xlsx', sheet_name='Data')
    df = df[df['Employee_Pay_Grade'] >= 20]
    df = df[df['Sector_Fixed'] != 'Supply Chain']
    df.info()
    df2 = pd.DataFrame()
    df2['WWID'] = df['WWID']
    df2['Termination_Reason'] = df['Termination_Reason']

    if year == '2016':
        df2['Planned_as_a___of_Bonus_Tar'] = df['_015_Planned_as_a___of_Bonus_Tar']
        df2['Planned_as_a___of_Merit_Tar'] = df['_015_Planned_as_a___of_Merit_Tar']
        df2['Mgr_Change'] = df['Mgr_Change_2015']  # .map(lambda x: x > 0)
        df2['SkipLevel_Mgr_Change'] = df['SkipLevel_Mgr_Change_2015']  # .map(lambda x: x > 0)
        df2['Promotion'] = df['Promo_2015']
        df2['Demotion'] = df['Demo_2015']
        df2['Lateral'] = df['Lateral_2015']
    elif year == '2017':
        df2['Planned_as_a___of_Bonus_Tar'] = df['_016_Planned_as_a___of_Bonus_Tar']
        df2['Planned_as_a___of_Merit_Tar'] = df['_016_Planned_as_a___of_Merit_Tar']
        df2['Mgr_Change'] = df['Mgr_Change_2016']  # .map(lambda x: x > 0)
        df2['SkipLevel_Mgr_Change'] = df['SkipLevel_Mgr_Change_2016']  # .map(lambda x: x > 0)
        df2['Promotion'] = df['Promo_2016']
        df2['Demotion'] = df['Demo_2016']
        df2['Lateral'] = df['Lateral_2016']
    elif year == '2018':
        df2['Planned_as_a___of_Bonus_Tar'] = df['_017_Planned_as_a___of_Bonus_Tar']
        df2['Planned_as_a___of_Merit_Tar'] = df['_017_Planned_as_a___of_Merit_Tar']
        df2['Mgr_Change'] = df['Mgr_Change_2017']  # .map(lambda x: x > 0)
        df2['SkipLevel_Mgr_Change'] = df['SkipLevel_Mgr_Change_2017']  # .map(lambda x: x > 0)
        df2['Promotion'] = df['Promo_2017']
        df2['Demotion'] = df['Demo_2017']
        df2['Lateral'] = df['Lateral_2017']
    elif year == '2019':
        df2['Planned_as_a___of_Bonus_Tar'] = df['_018_Planned_as_a___of_Bonus_Tar']
        df2['Planned_as_a___of_Merit_Tar'] = df['_018_Planned_as_a___of_Merit_Tar']
        df2['Mgr_Change'] = df['Mgr_Change_2018']  # .map(lambda x: x > 0)
        df2['SkipLevel_Mgr_Change'] = df['SkipLevel_Mgr_Change_2018']  # .map(lambda x: x > 0)
        df2['Promotion'] = df['Promo_2018']
        df2['Demotion'] = df['Demo_2018']
        df2['Lateral'] = df['Lateral_2018']
    elif year == '2020':
        df2['Planned_as_a___of_Bonus_Tar'] = 0
        df2['Planned_as_a___of_Merit_Tar'] = 0
        df2['Mgr_Change'] = df['Mgr_Change_2019']  # .map(lambda x: x > 0)
        df2['SkipLevel_Mgr_Change'] = df['SkipLevel_Mgr_Change_2019']  # .map(lambda x: x > 0)
        df2['Promotion'] = df['Promo_2019']
        df2['Demotion'] = df['Demo_2019']
        df2['Lateral'] = df['Lateral_2019']

    for c in ['Compensation_Range___Midpoint', 'Total_Base_Pay___Local', 'Job_Sub_Function__IA__Host_All_O',
              'Length_of_Service_in_Years_inclu', 'Job_Function__IA__Host_All_Other',
              # 'Promotion', 'Demotion', 'Lateral',
              # 'Cross_Move', 'Trainings_Completed',
              # 'Mgr_Change_YN',  'SkipLevel_Mgr_Change',
              'Rehire_YN',
              'Employee_Pay_Grade',
              # '_018_Planned_as_a___of_Bonus_Tar','_017_Planned_as_a___of_Bonus_Tar','_016_Planned_as_a___of_Bonus_Tar',
              'Highest_Degree_Received',
              # 'Actual_Sales_Incentive__2016', 'Actual_Sales_Incentive__2017',
              # 'Actual_Sales_Incentive__2018', 'Target_Sales_Incentive__2016',
              # 'Target_Sales_Incentive__2017', 'Target_Sales_Incentive__2018',
              # 'Hire_Date__Most_Recent_',  # 'Termination_Date',
              'Working_Country_Fixed',
              'Location_Code__IA__Host_All_Othe',
              'Manager_WWID__IA__Host_All_Other']:
        df2[c] = df[c]

    em1 = []
    mm1 = []
    # for EM in ['Employee', 'Manager']:
    #     df2[EM + '_Rating_1'] = df[EM + '_Rating_1']
    #     df2[EM + '_Rating_2'] = df[EM + '_Rating_2']
    #     df2[EM + '_Rating_3'] = df[EM + '_Rating_3']

    for er1, er2, er3, ery1, ery2, ery3 in zip(df['Employee_Rating_1'], df['Employee_Rating_2'],
                                               df['Employee_Rating_3'],
                                               df['Employee_Rating_Year_1'], df['Employee_Rating_Year_2'],
                                               df['Employee_Rating_Year_3']):
        if year == '2016':
            if ery1 == 2015:
                em1.append(er1)
            elif ery2 == 2015:
                em1.append(er2)
            elif ery3 == 2015:
                em1.append(er3)
            else:
                em1.append(0)
        elif year == '2017':
            if ery1 == 2016:
                em1.append(er1)
            elif ery2 == 2016:
                em1.append(er2)
            elif ery3 == 2016:
                em1.append(er3)
            else:
                em1.append(0)
        elif year == '2018':
            if ery1 == 2017:
                em1.append(er1)
            elif ery2 == 2017:
                em1.append(er2)
            elif ery3 == 2017:
                em1.append(er3)
            else:
                em1.append(0)
        elif year == '2019':
            if ery1 == 2018:
                em1.append(er1)
            elif ery2 == 2018:
                em1.append(er2)
            elif ery3 == 2018:
                em1.append(er3)
            else:
                em1.append(0)

    for er1, er2, er3, ery1, ery2, ery3 in zip(df['Manager_Rating_1'], df['Manager_Rating_2'],
                                               df['Manager_Rating_3'],
                                               df['Manager_Rating_Year_1'], df['Manager_Rating_Year_2'],
                                               df['Manager_Rating_Year_3']):
        if year == '2016':
            if ery1 == 2015:
                mm1.append(er1)
            elif ery2 == 2015:
                mm1.append(er2)
            elif ery3 == 2015:
                mm1.append(er3)
            else:
                mm1.append(0)
        elif year == '2017':
            if ery1 == 2016:
                mm1.append(er1)
            elif ery2 == 2016:
                mm1.append(er2)
            elif ery3 == 2016:
                mm1.append(er3)
            else:
                mm1.append(0)
        elif year == '2018':
            if ery1 == 2017:
                mm1.append(er1)
            elif ery2 == 2017:
                mm1.append(er2)
            elif ery3 == 2017:
                mm1.append(er3)
            else:
                mm1.append(0)
        elif year == '2019':
            if ery1 == 2018:
                mm1.append(er1)
            elif ery2 == 2018:
                mm1.append(er2)
            elif ery3 == 2018:
                mm1.append(er3)
            else:
                mm1.append(0)

    if year == '2020':
        df2['Employee_Rating_1'] = 0
        df2['Manager_Rating_1'] = 0
    else:
        df2['Employee_Rating_1'] = em1
        df2['Manager_Rating_1'] = mm1

    print(df2['Employee_Rating_1'].value_counts())

    ws = [x for x in df_2020.WWID.values if x in df2.WWID.values]
    df2 = df2.set_index('WWID')
    df_2020 = df_2020.set_index('WWID')
    for w in df2.index:
        if w in ws:
            er1 = df_2020.at[w, 'Employee_Rating_1']
            if type(er1) is numpy.ndarray:
                er1 = er1[0]
            ery1 = df_2020.at[w, 'Employee_Rating_Year_1']
            if type(ery1) is numpy.ndarray:
                ery1 = ery1[0]
            er2 = df_2020.at[w, 'Employee_Rating_2']
            if type(er2) is numpy.ndarray:
                er2 = er2[0]
            ery2 = df_2020.at[w, 'Employee_Rating_Year_2']
            if type(ery2) is numpy.ndarray:
                ery2 = ery2[0]
            er3 = df_2020.at[w, 'Employee_Rating_3']
            if type(er3) is numpy.ndarray:
                er3 = er3[0]
            ery3 = df_2020.at[w, 'Employee_Rating_Year_3']
            if type(ery3) is numpy.ndarray:
                ery3 = ery3[0]
            ##
            mr1 = df_2020.at[w, 'Manager_Rating_1']
            if type(mr1) is numpy.ndarray:
                mr1 = mr1[0]
            mry1 = df_2020.at[w, 'Manager_Rating_Year_1']
            if type(mry1) is numpy.ndarray:
                mry1 = mry1[0]
            mr2 = df_2020.at[w, 'Manager_Rating_2']
            if type(mr2) is numpy.ndarray:
                mr2 = mr2[0]
            mry2 = df_2020.at[w, 'Manager_Rating_Year_2']
            if type(mry2) is numpy.ndarray:
                mry2 = mry2[0]
            mr3 = df_2020.at[w, 'Manager_Rating_3']
            if type(mr3) is numpy.ndarray:
                mr3 = mr3[0]
            mry3 = df_2020.at[w, 'Manager_Rating_Year_3']
            if type(mry3) is numpy.ndarray:
                mry3 = mry3[0]
            ##
            if year == '2018':
                if ery1 == 2017:
                    df2.at[w, 'Employee_Rating_1'] = er1
                elif ery2 == 2017:
                    df2.at[w, 'Employee_Rating_1'] = er2
                elif ery3 == 2017:
                    df2.at[w, 'Employee_Rating_1'] = er3
                if mry1 == 2017:
                    df2.at[w, 'Manager_Rating_1'] = mr1
                elif mry2 == 2017:
                    df2.at[w, 'Manager_Rating_1'] = mr2
                elif mry3 == 2017:
                    df2.at[w, 'Manager_Rating_1'] = mr3
            elif year == '2019':
                if ery1 == 2018:
                    df2.at[w, 'Employee_Rating_1'] = er1
                elif ery2 == 2018:
                    df2.at[w, 'Employee_Rating_1'] = er2
                elif ery3 == 2018:
                    df2.at[w, 'Employee_Rating_1'] = er3

                if mry1 == 2018:
                    df2.at[w, 'Manager_Rating_1'] = mr1
                elif mry2 == 2018:
                    df2.at[w, 'Manager_Rating_1'] = mr2
                elif mry3 == 2018:
                    df2.at[w, 'Manager_Rating_1'] = mr3

    print(df2['Employee_Rating_1'].value_counts())
    df2['WWID'] = df2.index

    df_filtered = df2.query('Termination_Reason != "End of Contract/Assignment Completed"')

    if True:
        df_filtered = df_filtered.assign(Compensation_Range___Midpoint=pd.Series(
            df_filtered['Compensation_Range___Midpoint'].replace(0, 1e9)).values)

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
            for c in ['1']:
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
                    'Does Not Meet / Exceeds', 1)

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

    tenure = []
    status = []

    for r, ten in zip(df_filtered['Termination_Reason'],
                      # df_filtered['Termination_Date'],
                      df_filtered['Length_of_Service_in_Years_inclu']):
        if r == 'Resignation' and year != '2019':
            # d2 = datetime.strptime(et, "%Y-%m-%d")
            # tenure.append(abs((d2 - d1).days) / 365)
            status.append(1)
        else:
            tenure.append(ten)
            status.append(0)

    # df_filtered = df_filtered.assign(Tenure=pd.Series(tenure).values)
    df_filtered['Tenure'] = df_filtered['Length_of_Service_in_Years_inclu']
    df_filtered = df_filtered.assign(Status=pd.Series(status).values)

    df_filtered = df_filtered.drop(['Length_of_Service_in_Years_inclu',
                                    'Compensation_Range___Midpoint',
                                    'Termination_Reason'],
                                   axis=1)
    df_filtered['Tenure_log'] = np.log(df_filtered['Tenure'] + 1)
    # df_filtered['Mgr_Change_YN'] = df_filtered['Mgr_Change_YN'].astype('category').cat.codes
    # df_filtered.info()

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
    # df_filtered_filtered.info()

    x_resigned_new = df_filtered[df_filtered['Status'] == 1]
    # x_resigned_new.info()

    resigs = x_resigned_new.WWID.values.tolist()
    # print('Resignations:', len(resigs), resigs)

    df_filtered_filtered = df_filtered_filtered.set_index('WWID')
    for w in df_filtered_filtered.index:
        # print(w, df_filtered_filtered.at[w, 'Status'])
        if w in resigs:
            df_filtered_filtered.at[w, 'Status'] = 1

    # print('Filtered')
    # df_filtered_filtered.info()

    df_filtered_filtered.to_csv('data_files/SEA/' + year + '_fixed.csv', sep=',', encoding='utf-8')


def pickle_dataframe():
    from preprocessing import OneHotEncoder

    df_name = 'merged_Sea_combined_fixed2'
    df = pd.read_csv('data_files/SEA/' + df_name + '.csv', sep=',')
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[df.Job_Function__IA__Host_All_Other != 'Operations']
    df = df[df.Job_Function__IA__Host_All_Other != 'Quality']
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
    data_x = data_x.drop(["Tenure"], axis=1)
    data_x['Rehire_YN'] = data_x['Rehire_YN'].cat.codes
    data_x.info()

    data_x_numeric = OneHotEncoder().fit_transform(data_x)

    print(data_x_numeric.head())

    # for c in data_x_numeric.columns:
    #     if 'Missing' in c:
    #         print(c)
    #         data_x_numeric = data_x_numeric.drop(c, axis=1)

    data_x_numeric['Bonus_Flag'] = (df['Planned_as_a___of_Bonus_Tar'] == 0)

    data_x_numeric.info()
    data_x_numeric = data_x_numeric.fillna(-999)

    # test_1 = np.random.normal(loc=0.0, scale=0.01, size=data_x_numeric[data_x_numeric['Status'] == 1].shape[0])
    # test_2 = np.random.normal(loc=1.0, scale=0.01, size=data_x_numeric[data_x_numeric['Status'] == 0].shape[0])
    # assert data_x_numeric.shape[0] == len(test_1) + len(test_2)
    # test_l = list(test_1) + list(test_2)
    # data_x_numeric['test_gauss'] = test_l

    data_x_numeric.to_pickle("./data_files/SEA/"+df_name+"_x_numeric2.pkl")
    data_x_numeric.to_csv("./data_files/SEA/" + df_name + "_x_numeric2.csv", sep=',', encoding='utf-8')


def fix_moves_by_year(y1, y2):
    from os import path
    import numpy

    year2 = str(y2) if y2 > y1 else str(y1)
    year1 = str(y1) if y2 > y1 else str(y2)

    if path.exists('data_files/SEA/' + year1 + '_fixed.csv') and path.exists('data_files/SEA/' + year2 + '_fixed.csv'):
        print('Moving from '+year2+' to '+year1)
        df1 = pd.read_csv('data_files/SEA/' + year1 + '_fixed.csv', sep=',')
        df2 = pd.read_csv('data_files/SEA/' + year2 + '_fixed.csv', sep=',')
    else:
        print('Files not available')
        return

    ws = [x for x in df2.WWID.values if x in df1.WWID.values]

    df1 = df1.set_index('WWID')
    df2 = df2.set_index('WWID')

    # df2.info()

    # for w in df1.index:
    #     if w in ws:
    #         prom = df2.at[w, 'Promotion']
    #         demo = df2.at[w, 'Demotion']
    #         late = df2.at[w, 'Lateral']
    #         # cros = df2.at[w, 'Cross_Move']
    #     else:
    #         prom = 0
    #         demo = 0
    #         late = 0
    #         # cros = 0
    #
    #     if type(prom) is numpy.ndarray:
    #         df1.at[w, 'Promotion'] = prom[0]
    #         df1.at[w, 'Demotion'] = demo[0]
    #         df1.at[w, 'Lateral'] = late[0]
    #         # df1.at[w, 'Cross_Move'] = cros[0]
    #     else:
    #         df1.at[w, 'Promotion'] = prom
    #         df1.at[w, 'Demotion'] = demo
    #         df1.at[w, 'Lateral'] = late
    #         # df1.at[w, 'Cross_Move'] = cros
    #
    # print(df1['Promotion'].head())
    # print(df2['Promotion'].head())

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

    df1.to_csv('data_files/SEA/' + year1 + '_fixed2.csv', sep=',', encoding='utf-8')
    df2.to_csv('data_files/SEA/' + year2 + '_fixed2.csv', sep=',', encoding='utf-8')


def merge_files():
    # df_2015 = pd.read_csv('Brazil_2015_filtered_shifted.csv', sep=',')
    # df_2016 = pd.read_csv('Brazil_2016_filtered_shifted.csv', sep=',')
    # df_2017 = pd.read_csv('Brazil_2017_filtered_shifted.csv', sep=',')
    # df_2018 = pd.read_csv('Brazil_2018_filtered.csv', sep=',')

    df_2016 = pd.read_csv('data_files/SEA/2016_fixed2.csv', sep=',')
    df_2017 = pd.read_csv('data_files/SEA/2017_fixed2.csv', sep=',')
    df_2018 = pd.read_csv('data_files/SEA/2018_fixed2.csv', sep=',')
    df_2019 = pd.read_csv('data_files/SEA/2019_fixed2.csv', sep=',')

    df_2016['Report_Year'] = 2016
    df_2017['Report_Year'] = 2017
    df_2018['Report_Year'] = 2018
    df_2019['Report_Year'] = 2019

    df_merged = df_2016.append(df_2017, sort=True)
    df_merged = df_merged.append(df_2018, sort=True)
    df_merged = df_merged.append(df_2019, sort=True)
    df_merged.to_csv('data_files/SEA/merged_Sea_combined_fixed2.csv', sep=',', encoding='utf-8')


def pickle_current_file():
    from preprocessing import OneHotEncoder

    df_original = pd.read_excel('Data Request_September 16 - September 19.xlsx')
    df_original = df_original[(df_original['Working_Country_Fixed'] == 'Singapore') |
                              (df_original['Working_Country_Fixed'] == 'Thailand') |
                              (df_original['Working_Country_Fixed'] == 'Malaysia') |
                              (df_original['Working_Country_Fixed'] == 'Indonesia') |
                              (df_original['Working_Country_Fixed'] == 'Vietnam') |
                              (df_original['Working_Country_Fixed'] == 'Philippines')
    ]
    df_original.info()
    df = pd.DataFrame()
    # df['Compa_Diff_Ratio'] = (df_original['Total_Base_Pay___Local'] -
    #         df_original['Compensation_Range___Midpoint']) / df_original['Compensation_Range___Midpoint']
    df['Compa_Ratio'] = df_original['Total_Base_Pay___Local']/df_original['Compensation_Range___Midpoint']
    # df['Compensation_Range___Midpoint'] = df_original['Compensation_Range___Midpoint']
    df['Demotion'] = df_original['Demo_2019']
    df['Employee_Pay_Grade'] = df_original['Employee_Pay_Grade']
    df['Employee_Rating_1'] = df_original['Employee_Rating_1']
    # df['Employee_Rating_2'] = df_original['Employee_Rating_2']
    # df['Employee_Rating_3'] = df_original['Employee_Rating_3']
    df['Highest_Degree_Received'] = df_original['Highest_Degree_Received']
    df['Job_Function__IA__Host_All_Other'] = df_original['Job_Function__IA__Host_All_Other']
    df['Job_Sub_Function__IA__Host_All_O'] = df_original['Job_Sub_Function__IA__Host_All_O']
    df['Lateral'] = df_original['Lateral_2019']
    df['Location_Code__IA__Host_All_Othe'] = df_original['Location_Code__IA__Host_All_Othe']
    df['Manager_Rating_1'] = df_original['Manager_Rating_1']
    # df['Manager_Rating_2'] = df_original['Manager_Rating_2']
    # df['Manager_Rating_3'] = df_original['Manager_Rating_3']
    df['Manager_WWID__IA__Host_All_Other'] = df_original['Manager_WWID__IA__Host_All_Other']
    df['Mgr_Change'] = df_original['Mgr_Change_2019']
    df['Planned_as_a___of_Bonus_Tar'] = df_original['_018_Planned_as_a___of_Bonus_Tar']
    df['Planned_as_a___of_Merit_Tar'] = df_original['_018_Planned_as_a___of_Merit_Tar']
    df['Promotion'] = df_original['Promo_2019']
    df['Rehire_YN'] = df_original['Rehire_YN']
    df['Report_Year'] = 2019
    df['Skip_Manager_Change'] = df_original['SkipLevel_Mgr_Change_2019']
    df['Tenure'] = df_original['Length_of_Service_in_Years_inclu']
    df['Tenure_log'] = np.log(df['Tenure'] + 1)
    df['Total_Base_Pay___Local'] = df_original['Total_Base_Pay___Local']
    df['WWID'] = df_original['WWID']
    df['Working_Country_Fixed'] = df_original['Working_Country_Fixed']

    for EM in ['Employee', 'Manager']:
        for c in ['1']:
            # c = str(1)
            # df[EM+'_Rating_'+c+'_W'] = df[EM+'_Rating_'+c].str.split('/').str.get(0).str.strip()
            # df[EM+'_Rating_'+c+'_H'] = df[EM+'_Rating_'+c].str.split('/').str.get(1).str.strip()
            # df = df.drop(EM+'_Rating_' + c, axis=1)

            print(df[EM + '_Rating_' + c].value_counts())

            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace('1', 1)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace('2', 1)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace('3', 1)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace('4', 1)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace('5', 1)

            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace('6', 2)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace('7', 2)

            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace('8', 3)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace('9', 3)

            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Insufficient Data to Rate / Insufficient Data to Rate', 0)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Does Not Meet / Partially Meets', 1)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Does Not Meet / Fully Meets', 1)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Fully Meets / Partially Meets', 1)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Partially Meets / Partially Meets', 1)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Partially Meets / Does Not Meet', 1)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Does Not Meet / Does Not Meet', 1)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Partially Meets / Fully Meets', 1)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Fully Meets / Does Not Meet', 1)

            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Fully Meets / Fully Meets', 2)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Fully Meets / Exceeds', 2)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Exceeds / Fully Meets', 2)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Partially Meets / Exceeds', 2)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Exceeds / Partially Meets', 2)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Exceeds / Does Not Meet', 2)
            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Does Not Meet / Exceeds', 2)

            df[EM + '_Rating_' + c] = df[EM + '_Rating_' + c].replace(
                'Exceeds / Exceeds', 3)
            print('Cleaned')
            print(df[EM + '_Rating_' + c].value_counts())

    df = df.sample(frac=1).reset_index(drop=True)
    # df = df[df.Job_Function__IA__Host_All_Other != 'Operations']
    print('size=', df.shape[0])
    data_x = df.drop(["Manager_WWID__IA__Host_All_Other"], axis=1)
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

    df_name = 'merged_Sea_combined'
    data_x_numeric.to_pickle("./data_files/SEA/" + df_name + "_current_x_numeric_newer2.pkl")
    data_x_numeric.to_csv("./data_files/SEA/" + df_name + "_current_x_numeric_newer2.csv", sep=',', encoding='utf-8')


if __name__ == '__main__':
    # split_files()
    # # write_to_pickle(year)
    #
    # clean_dataframe('2016')
    # clean_dataframe('2017')
    # clean_dataframe('2018')
    # clean_dataframe('2019')
    # clean_dataframe('2020')
    #
    # fix_moves_by_year(2016, 2017)
    # fix_moves_by_year(2017, 2018)
    # fix_moves_by_year(2018, 2019)
    # fix_moves_by_year(2019, 2020)
    #
    # combine(2016)
    # combine(2017)
    # combine(2018)
    # combine(2019)

    merge_files()
    pickle_dataframe()
    # pickle_current_file()
