import pandas as pd
import numpy as np
from preprocessing import OneHotEncoder

def split_files():
    df_original = pd.read_excel('Brazil_Retention_Dataset_07192019.xlsx', sheet_name='Data')
    df_original.info()
    print(df_original['Report_Date'].str.split('/').head())

    df_2018 = df_original[df_original['Report_Date'].str.contains('2018')]
    df_2018.info()
    df_2017 = df_original[df_original['Report_Date'].str.contains('2017')]
    df_2017.info()
    df_2016 = df_original[df_original['Report_Date'].str.contains('2016')]
    df_2016.info()
    df_2015 = df_original[df_original['Report_Date'].str.contains('2015')]
    df_2015.info()

    df_2018.to_csv('Brazil_2018.csv', sep=',', encoding='utf-8')
    df_2017.to_csv('Brazil_2017.csv', sep=',', encoding='utf-8')
    df_2016.to_csv('Brazil_2016.csv', sep=',', encoding='utf-8')
    df_2015.to_csv('Brazil_2015.csv', sep=',', encoding='utf-8')


def clean_dataframe(df_name):
    from datetime import datetime
    df = pd.read_csv(df_name+'.csv', sep=',')
    df2 = pd.DataFrame()
    df2['WWID'] = df['WWID']
    df2['Termination_Reason'] = df['Termination_Reason']

    for c in ['Compensation_Range___Midpoint', 'Total_Base_Pay___Local', 'Job_Sub_Function__IA__Host_All_O',
              'Length_of_Service_in_Years_inclu', 'Job_Function__IA__Host_All_Other', 'Promotion', 'Demotion',
              'Lateral', 'Cross_Move', 'Trainings_Completed', 'Mgr_Change', 'SkipLevel_Mgr_Change', 'Rehire_YN',
              '_018_Planned_as_a___of_Bonus_Tar', '_017_Planned_as_a___of_Bonus_Tar', '_016_Planned_as_a___of_Bonus_Tar',
              'Highest_Degree_Received', 'Actual_Sales_Incentive__2016', 'Actual_Sales_Incentive__2017',
              'Actual_Sales_Incentive__2018', 'Target_Sales_Incentive__2016', 'Target_Sales_Incentive__2017',
              'Target_Sales_Incentive__2018', 'Hire_Date__Most_Recent_', 'Termination_Date']:
        df2[c] = df[c]

    for EM in ['Employee', 'Manager']:
        for c in range(1, 4):
            c = str(c)
            df2[EM+'_Rating_'+c] = df[EM+'_Rating_'+c]

    df_filtered = df2.query('Termination_Reason != "End of Contract/Assignment Completed"')

    if True:
        df_filtered = df_filtered.assign(Compensation_Range___Midpoint=pd.Series(df_filtered['Compensation_Range___Midpoint'].replace(0, 1e9)).values)
        df_filtered['Compa_Diff_Ratio'] = (df_filtered['Total_Base_Pay___Local']-df_filtered['Compensation_Range___Midpoint'])\
                                          / df_filtered['Compensation_Range___Midpoint']
        df_filtered['Compa_Ratio'] = df_filtered['Total_Base_Pay___Local']/df_filtered['Compensation_Range___Midpoint']

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

    df_filtered = df_filtered.assign(Tenure=pd.Series(tenure).values)
    df_filtered = df_filtered.assign(Status=pd.Series(status).values)

    df_filtered = df_filtered.drop(['Hire_Date__Most_Recent_', 'Termination_Date', 'Length_of_Service_in_Years_inclu'],
                                   axis=1)
    df_filtered['Tenure_log'] = np.log(df_filtered['Tenure'] + 1)

    df_filtered.info()
    df_filtered.to_csv(df_name+'_filtered.csv', sep=',', encoding='utf-8')


def pickle_dataframe(df_name):
    df = pd.read_csv(df_name + '.csv', sep=',')
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[df.Job_Function__IA__Host_All_Other != 'Operations']
    data_x = df.drop(["Status", "Unnamed: 0"], axis=1)
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
    data_y = pd.Series()
    data_y['Status'] = df['Status']
    data_x_numeric.to_pickle("./data_x_numeric_new.pkl")
    data_y.to_pickle("./data_y_new.pkl")


if __name__ == "__main__":
    #clean_dataframe('Brazil_2015')
    pickle_dataframe('Brazil_2015_filtered')
