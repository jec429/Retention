import pandas as pd


def combine(year):
    df = pd.read_csv('data_files/' + str(year)+'_clean.csv', sep=',')
    df_pre = pd.read_csv('data_files/' + str(year-1)+'_clean.csv', sep=',')
    df_pre_pre = pd.read_csv('data_files/' + str(year-2)+'_clean.csv', sep=',')

    df['Planned_as_a___of_Bonus_Tar_1'] = df['Planned_as_a___of_Bonus_Tar']
    df.info()

    df_pre = df_pre[df_pre['WWID'].isin(list(df['WWID'].values))]
    df_pre_pre = df_pre_pre[df_pre_pre['WWID'].isin(list(df['WWID'].values))]

    df_outer = pd.merge(df, df_pre, on='WWID', how='outer')

    df['Employee_Rating_2'] = df_outer['Employee_Rating_1_y']
    df['Manager_Rating_2'] = df_outer['Manager_Rating_1_y']
    df['Planned_as_a___of_Bonus_Tar_2'] = df_outer['Planned_as_a___of_Bonus_Tar_y']

    df_outer = pd.merge(df, df_pre_pre, on='WWID', how='outer')

    df['Employee_Rating_3'] = df_outer['Employee_Rating_1_y']
    df['Manager_Rating_3'] = df_outer['Manager_Rating_1_y']
    df['Planned_as_a___of_Bonus_Tar_3'] = df_outer['Planned_as_a___of_Bonus_Tar_y']

    df = df.drop(['Planned_as_a___of_Bonus_Tar'], axis=1)
    df.info()

    df.to_csv('data_files/' + str(year) + '_combined.csv', sep=',')
