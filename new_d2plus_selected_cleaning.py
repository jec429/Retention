import pandas as pd
import re
import numpy as np


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

    df_2020.to_csv('./data_files/D2PlusSelected/D2Plus_2020.csv', sep=',', encoding='utf-8')
    df_2019.to_csv('./data_files/D2PlusSelected/D2Plus_2019.csv', sep=',', encoding='utf-8')
    df_2018.to_csv('./data_files/D2PlusSelected/D2Plus_2018.csv', sep=',', encoding='utf-8')
    df_2017.to_csv('./data_files/D2PlusSelected/D2Plus_2017.csv', sep=',', encoding='utf-8')
    df_2016.to_csv('./data_files/D2PlusSelected/D2Plus_2016.csv', sep=',', encoding='utf-8')


def clean_dataframe(year):
    df = pd.read_csv('./data_files/D2PlusSelected/D2Plus_' + year + '.csv', sep=',')
    df_2020 = pd.read_csv('./data_files/D2PlusSelected/D2Plus_2020.csv', sep=',')

    df = df[df['Employee_Pay_Grade'] >= 20]
    df.info()
    df2 = pd.DataFrame()
    df2['WWID'] = df['WWID']
    df2['Termination_Reason'] = df['Termination_Reason']
    df2['Compa_Ratio'] = df['Compa_Ratio']

    if year == '2016':
        df2['Planned_as_a___of_Bonus_Tar'] = df['_015_Planned_as_a___of_Bonus_Tar']
        df2['Planned_as_a___of_Merit_Tar'] = df['_015_Planned_as_a___of_Merit_Tar']
        df2['Planned_as_a___of_Merit_Tar'] = df['_015_Planned_as_a___of_LTI_Targe']
        df2['Mgr_Change'] = df['Mgr_Change_2015']  # .map(lambda x: x > 0)
        df2['SkipLevel_Mgr_Change'] = df['SkipLevel_Mgr_Change_2015']  # .map(lambda x: x > 0)
        df2['Promotion'] = df['Promo_2015']
        df2['Demotion'] = df['Demo_2015']
        df2['Lateral'] = df['Lateral_2015']
        df2['Bonus'] = df['Bonus15']
        df2['LTI'] = df['LTI15']
        df2['Merit'] = df['Merit15']
    elif year == '2017':
        df2['Planned_as_a___of_Bonus_Tar'] = df['_016_Planned_as_a___of_Bonus_Tar']
        df2['Planned_as_a___of_Merit_Tar'] = df['_016_Planned_as_a___of_Merit_Tar']
        df2['Planned_as_a___of_Merit_Tar'] = df['_016_Planned_as_a___of_LTI_Targe']
        df2['Mgr_Change'] = df['Mgr_Change_2016']  # .map(lambda x: x > 0)
        df2['SkipLevel_Mgr_Change'] = df['SkipLevel_Mgr_Change_2016']  # .map(lambda x: x > 0)
        df2['Promotion'] = df['Promo_2016']
        df2['Demotion'] = df['Demo_2016']
        df2['Lateral'] = df['Lateral_2016']
        df2['Bonus'] = df['Bonus16']
        df2['LTI'] = df['LTI16']
        df2['Merit'] = df['Merit16']
    elif year == '2018':
        df2['Planned_as_a___of_Bonus_Tar'] = df['_017_Planned_as_a___of_Bonus_Tar']
        df2['Planned_as_a___of_Merit_Tar'] = df['_017_Planned_as_a___of_Merit_Tar']
        df2['Planned_as_a___of_Merit_Tar'] = df['_017_Planned_as_a___of_LTI_Targe']
        df2['Mgr_Change'] = df['Mgr_Change_2017']  # .map(lambda x: x > 0)
        df2['SkipLevel_Mgr_Change'] = df['SkipLevel_Mgr_Change_2017']  # .map(lambda x: x > 0)
        df2['Promotion'] = df['Promo_2017']
        df2['Demotion'] = df['Demo_2017']
        df2['Lateral'] = df['Lateral_2017']
        df2['Bonus'] = df['Bonus17']
        df2['LTI'] = df['LTI17']
        df2['Merit'] = df['Merit17']
    elif year == '2019':
        df2['Planned_as_a___of_Bonus_Tar'] = df['_018_Planned_as_a___of_Bonus_Tar']
        df2['Planned_as_a___of_Merit_Tar'] = df['_018_Planned_as_a___of_Merit_Tar']
        df2['Planned_as_a___of_Merit_Tar'] = df['_018_Planned_as_a___of_LTI_Targe']
        df2['Mgr_Change'] = df['Mgr_Change_2018']  # .map(lambda x: x > 0)
        df2['SkipLevel_Mgr_Change'] = df['SkipLevel_Mgr_Change_2018']  # .map(lambda x: x > 0)
        df2['Promotion'] = df['Promo_2018']
        df2['Demotion'] = df['Demo_2018']
        df2['Lateral'] = df['Lateral_2018']
        df2['Bonus'] = df['Bonus18']
        df2['LTI'] = df['LTI18']
        df2['Merit'] = df['Merit18']
    elif year == '2020':
        df2['Planned_as_a___of_Bonus_Tar'] = df['_019_Planned_as_a___of_Bonus_Tar']
        df2['Planned_as_a___of_Merit_Tar'] = df['_019_Planned_as_a___of_Merit_Tar']
        df2['Planned_as_a___of_Merit_Tar'] = df['_019_Planned_as_a___of_LTI_Targe']
        df2['Mgr_Change'] = df['Mgr_Change_2019']  # .map(lambda x: x > 0)
        df2['SkipLevel_Mgr_Change'] = df['SkipLevel_Mgr_Change_2019']  # .map(lambda x: x > 0)
        df2['Promotion'] = df['Promo_2019']
        df2['Demotion'] = df['Demo_2019']
        df2['Lateral'] = df['Lateral_2019']
        df2['Bonus'] = df['Bonus19']
        df2['LTI'] = df['LTI19']
        df2['Merit'] = df['Merit19']

    for c in ['Compensation_Range___Midpoint', 'Total_Base_Pay___Local',
              'Job_Sub_Function__IA__Host_All_O',
              'Length_of_Service_in_Years_inclu', 'Job_Function__IA__Host_All_Other',
              # 'Promotion', 'Demotion', 'Lateral',
              # 'Cross_Move', 'Trainings_Completed',
              # 'Mgr_Change_YN',  'SkipLevel_Mgr_Change',
              'Rehire_YN',
              'Employee_Pay_Grade',
              'Highest_Degree_Received',
              # 'Actual_Sales_Incentive__2016', 'Actual_Sales_Incentive__2017',
              # 'Actual_Sales_Incentive__2018', 'Target_Sales_Incentive__2016',
              # 'Target_Sales_Incentive__2017', 'Target_Sales_Incentive__2018',
              # 'Hire_Date__Most_Recent_',  # 'Termination_Date',
              'Working_Country_Fixed',
              'Location_Code__IA__Host_All_Othe',
              'Manager_WWID__IA__Host_All_Other',
              'Stock_Salary___US', 'Bonus_FTE_Salary___US']:
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
                em1.append(-1)
        elif year == '2017':
            if ery1 == 2016:
                em1.append(er1)
            elif ery2 == 2016:
                em1.append(er2)
            elif ery3 == 2016:
                em1.append(er3)
            else:
                em1.append(-1)
        elif year == '2018':
            if ery1 == 2017:
                em1.append(er1)
            elif ery2 == 2017:
                em1.append(er2)
            elif ery3 == 2017:
                em1.append(er3)
            else:
                em1.append(-1)
        elif year == '2019':
            if ery1 == 2018:
                em1.append(er1)
            elif ery2 == 2018:
                em1.append(er2)
            elif ery3 == 2018:
                em1.append(er3)
            else:
                em1.append(-1)

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
                mm1.append(-1)
        elif year == '2017':
            if ery1 == 2016:
                mm1.append(er1)
            elif ery2 == 2016:
                mm1.append(er2)
            elif ery3 == 2016:
                mm1.append(er3)
            else:
                mm1.append(-1)
        elif year == '2018':
            if ery1 == 2017:
                mm1.append(er1)
            elif ery2 == 2017:
                mm1.append(er2)
            elif ery3 == 2017:
                mm1.append(er3)
            else:
                mm1.append(-1)
        elif year == '2019':
            if ery1 == 2018:
                mm1.append(er1)
            elif ery2 == 2018:
                mm1.append(er2)
            elif ery3 == 2018:
                mm1.append(er3)
            else:
                mm1.append(-1)

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
            if type(er1) is np.ndarray:
                er1 = er1[0]
            ery1 = df_2020.at[w, 'Employee_Rating_Year_1']
            if type(ery1) is np.ndarray:
                ery1 = ery1[0]
            er2 = df_2020.at[w, 'Employee_Rating_2']
            if type(er2) is np.ndarray:
                er2 = er2[0]
            ery2 = df_2020.at[w, 'Employee_Rating_Year_2']
            if type(ery2) is np.ndarray:
                ery2 = ery2[0]
            er3 = df_2020.at[w, 'Employee_Rating_3']
            if type(er3) is np.ndarray:
                er3 = er3[0]
            ery3 = df_2020.at[w, 'Employee_Rating_Year_3']
            if type(ery3) is np.ndarray:
                ery3 = ery3[0]
            ##
            mr1 = df_2020.at[w, 'Manager_Rating_1']
            if type(mr1) is np.ndarray:
                mr1 = mr1[0]
            mry1 = df_2020.at[w, 'Manager_Rating_Year_1']
            if type(mry1) is np.ndarray:
                mry1 = mry1[0]
            mr2 = df_2020.at[w, 'Manager_Rating_2']
            if type(mr2) is np.ndarray:
                mr2 = mr2[0]
            mry2 = df_2020.at[w, 'Manager_Rating_Year_2']
            if type(mry2) is np.ndarray:
                mry2 = mry2[0]
            mr3 = df_2020.at[w, 'Manager_Rating_3']
            if type(mr3) is np.ndarray:
                mr3 = mr3[0]
            mry3 = df_2020.at[w, 'Manager_Rating_Year_3']
            if type(mry3) is np.ndarray:
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
                    'Does Not Meet / Exceeds', 2)

                df_filtered[EM + '_Rating_' + c] = df_filtered[EM + '_Rating_' + c].replace(
                    'Exceeds / Exceeds', 3)
                print('Cleaned')
                print(df_filtered[EM + '_Rating_' + c].value_counts())

    tenure = []
    status = []

    for r, ten in zip(df_filtered['Termination_Reason'],
                      # df_filtered['Termination_Date'],
                      df_filtered['Length_of_Service_in_Years_inclu']):
        if r == 'Resignation':
            # d2 = datetime.strptime(et, "%Y-%m-%d")
            # tenure.append(abs((d2 - d1).days) / 365)
            status.append(1)
        else:
            tenure.append(ten)
            status.append(0)

    # df_filtered = df_filtered.assign(Tenure=pd.Series(tenure).values)
    df_filtered['Tenure'] = df_filtered['Length_of_Service_in_Years_inclu']
    df_filtered = df_filtered.assign(Status=pd.Series(status).values)

    df_filtered['Tenure_log'] = np.log(df_filtered['Tenure'] + 1)
    df_filtered = df_filtered.drop(['Length_of_Service_in_Years_inclu',
                                    'Compensation_Range___Midpoint',
                                    'Termination_Reason',
                                    'Job_Sub_Function__IA__Host_All_O',
                                    'Location_Code__IA__Host_All_Othe',
                                    'Tenure',
                                    'Manager_WWID__IA__Host_All_Other'],
                                   axis=1)

    # df_filtered['Mgr_Change_YN'] = df_filtered['Mgr_Change_YN'].astype('category').cat.codes
    # df_filtered.info()

    # df_filtered['Working_Country'] = 37

    # manager_manager = []
    # for mw in df_filtered['Manager_WWID__IA__Host_All_Other']:
    #     man_man = 0
    #     for w, mmw in zip(df_filtered['WWID'], df_filtered['Manager_WWID__IA__Host_All_Other']):
    #         if mw == w:
    #             man_man = mmw
    #             break
    #     manager_manager.append(man_man)
    #
    # df_filtered['Manager_Manager_WWID'] = manager_manager

    df_filtered_filtered = df_filtered[df_filtered['Status'] == 0]
    # df_filtered_filtered.info()

    x_resigned_new = df_filtered[df_filtered['Status'] == 1]
    # x_resigned_new.info()

    resigs = x_resigned_new.WWID.values.tolist()
    # print('Resignations:', len(resigs), resigs)

    df_filtered_filtered = df_filtered_filtered.set_index('WWID')
    df_filtered_filtered['WWID'] = df_filtered_filtered.index
    for w in df_filtered_filtered.index:
        # print(w, df_filtered_filtered.at[w, 'Status'])
        if w in resigs:
            df_filtered_filtered.at[w, 'Status'] = 1

    # print('Filtered')
    # df_filtered_filtered.info()
    df_filtered_filtered = df_filtered_filtered[~df_filtered_filtered['Compa_Ratio'].isnull()]

    df_filtered_filtered.to_csv('./data_files/D2PlusSelected/D2Plus_' + year + '_clean.csv', sep=',', encoding='utf-8')
    df_filtered_filtered.to_pickle('./data_files/D2PlusSelected/D2Plus_' + year + '_clean.pkl')


def merge_files():
    df_2016 = pd.read_pickle('data_files/D2PlusSelected/D2Plus_2016_clean.pkl')
    df_2017 = pd.read_pickle('data_files/D2PlusSelected/D2Plus_2017_clean.pkl')
    df_2018 = pd.read_pickle('data_files/D2PlusSelected/D2Plus_2018_clean.pkl')
    df_2019 = pd.read_pickle('data_files/D2PlusSelected/D2Plus_2019_clean.pkl')

    df_2016['Report_Year'] = 2016
    df_2017['Report_Year'] = 2017
    df_2018['Report_Year'] = 2018
    df_2019['Report_Year'] = 2019

    df_merged = df_2016.append(df_2017, sort=True)
    df_merged = df_merged.append(df_2018, sort=True)
    df_merged = df_merged.append(df_2019, sort=True)
    df_merged.to_pickle("./data_files/D2PlusSelected/merged_D2Plus_combined_fixed.pkl")
    df_merged.to_csv('data_files/D2PlusSelected/merged_D2Plus_combined_fixed.csv', sep=',', encoding='utf-8')


def pickle_dataframe():
    from preprocessing import OneHotEncoder

    df_name = 'merged_D2Plus_combined_fixed'
    df = pd.read_pickle('data_files/D2PlusSelected/' + df_name + '.pkl')
    df = df.sample(frac=1).reset_index(drop=True)
    print('size=', df.shape[0])
    to_drop = []

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
    data_x['Rehire_YN'] = data_x['Rehire_YN'].cat.codes
    data_x['Working_Country_Fixed'] = data_x['Working_Country_Fixed'].cat.codes
    data_x_numeric = OneHotEncoder().fit_transform(data_x)

    for c in data_x_numeric.columns:
        if 'Missing' in c:
            data_x_numeric = data_x_numeric.drop(c, axis=1)

    print(data_x_numeric.head())

    data_x_numeric.info()
    data_x_numeric = data_x_numeric.fillna(-999)
    data_x_numeric.to_pickle("./data_files/D2PlusSelected/"+df_name+"_x_numeric_newer.pkl")
    data_x_numeric2 = data_x_numeric[:100]
    data_x_numeric2.to_csv("./data_files/D2PlusSelected/" + df_name + "_x_numeric_newer.csv", sep=',', encoding='utf-8')


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
