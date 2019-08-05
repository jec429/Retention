import pandas as pd


def combine(year):
    df = pd.read_csv(str(year)+'_clean.csv', sep=',')
    df_pre = pd.read_csv(str(year-1)+'_clean.csv', sep=',')
    df_pre_pre = pd.read_csv(str(year-2)+'_clean.csv', sep=',')

    df['Planned_as_a___of_Bonus_Tar_1'] = df['Planned_as_a___of_Bonus_Tar']
    df.info()

    df_pre = df_pre[df_pre['WWID'].isin(list(df['WWID'].values))]
    df_pre_pre = df_pre_pre[df_pre_pre['WWID'].isin(list(df['WWID'].values))]

    df_outer = pd.merge(df, df_pre, on='WWID', how='outer')

    df['Employee_Rating_2_W'] = df_outer['Employee_Rating_1_W_y']
    df['Employee_Rating_2_H'] = df_outer['Employee_Rating_1_H_y']
    df['Manager_Rating_2_W'] = df_outer['Manager_Rating_1_W_y']
    df['Manager_Rating_2_H'] = df_outer['Manager_Rating_1_H_y']
    df['Planned_as_a___of_Bonus_Tar_2'] = df_outer['Planned_as_a___of_Bonus_Tar_y']

    df_outer = pd.merge(df, df_pre_pre, on='WWID', how='outer')

    df['Employee_Rating_3_W'] = df_outer['Employee_Rating_1_W_y']
    df['Employee_Rating_3_H'] = df_outer['Employee_Rating_1_H_y']
    df['Manager_Rating_3_W'] = df_outer['Manager_Rating_1_W_y']
    df['Manager_Rating_3_H'] = df_outer['Manager_Rating_1_H_y']
    df['Planned_as_a___of_Bonus_Tar_3'] = df_outer['Planned_as_a___of_Bonus_Tar_y']

    df = df.drop(['Planned_as_a___of_Bonus_Tar'], axis=1)
    df.info()

    df.to_csv(str(year)+'_combined.csv', sep=',')


if __name__ == '__main__':
    combine(2019)
