import pickle
import pandas as pd


# fname = "parrot_china_fixed.pkl"


def read_probs(parrot_fname, data_fname):
    with open(parrot_fname, "rb") as fin:
        list_lists2 = pickle.load(fin)
    wwids = list(list_lists2[0])
    print('Resgs=', len(list_lists2[2]), sum(list_lists2[2]))
    prob_tf = [[x[0], y] for x, y in zip(list_lists2[1], list_lists2[2])]

    res = dict(zip(wwids, prob_tf))

    # df = pd.read_csv('data_files/CHINA/China_2018.csv', sep=',')
    df = pd.read_csv(data_fname, sep=',')
    df = df[df['WWID'].isin(res.keys())]

    probs = []
    for w in df['WWID']:
        if w in res.keys():
            probs.append(res[w][0])

    df['Probs'] = probs

    df2 = pd.DataFrame()
    df2['WWID'] = df['WWID']
    df2['Probs'] = df['Probs']
    df2['Termination_Reason'] = df['Termination_Reason']
    df2['Tenure'] = df['Tenure']
    df2['Rating'] = df['Rating']
    df2['Sector'] = df['Sector_Fixed']
    df2['Function'] = df['Job_Function__IA__Host_All_Other']
    df2['Level'] = df['Level']
    df2['MgrRating'] = df['MgrRating']
    df2['PayGrade'] = df['Employee_Pay_Grade']
    df2['Tenure_time'] = df['Length_of_Service_in_Years_inclu']
    df2['Employee_Rating'] = df['Employee_Rating_1']

    df2.drop_duplicates(subset="WWID", keep='first', inplace=True)
    return df2


def read_probs_2019(parrot_fname, data_fname, term_fname):
    with open(parrot_fname, "rb") as fin:
        list_lists2 = pickle.load(fin)
    wwids = list(list_lists2[0])
    print('Resgs=', len(list_lists2[2]), sum(list_lists2[2]))
    prob_tf = [[x[0], y] for x, y in zip(list_lists2[1], list_lists2[2])]

    res = dict(zip(wwids, prob_tf))

    df = pd.read_csv(data_fname, sep=',')
    df = df[df['WWID'].isin(res.keys())]

    probs = []
    for w in df['WWID']:
        if w in res.keys():
            probs.append(res[w][0])

    df['Probs'] = probs

    df6 = pd.DataFrame()
    df6['WWID'] = df['WWID']
    df6['Probs'] = df['Probs']
    df6['Termination_Reason'] = df['Termination_Reason']
    df6['Tenure'] = df['Tenure']
    df6['Rating'] = df['Rating']
    df6['Sector'] = df['Sector_Fixed']
    df6['Function'] = df['Job_Function__IA__Host_All_Other']
    df6['Level'] = df['Level']
    df6['MgrRating'] = df['MgrRating']
    df6['PayGrade'] = df['Employee_Pay_Grade']
    df6['Tenure_time'] = df['Length_of_Service_in_Years_inclu']
    df6['Employee_Rating'] = df['Employee_Rating_1']

    df6.drop_duplicates(subset="WWID", keep='first', inplace=True)

    df_res_2019 = pd.read_excel(term_fname, sheet_name='Data')
    res_wwids = list(df_res_2019[df_res_2019['Termination_Reason'] == 'Resignation']['WWID'])
    new_status = []
    for w in df6['WWID']:
        if w in res_wwids:
            new_status.append(1)
        else:
            new_status.append(0)
    df6['New Status'] = new_status
    return df6


def read_probs_current(parrot_fname, data_fname):
    with open(parrot_fname, "rb") as fin:
        list_lists2 = pickle.load(fin)
    wwids = list(list_lists2[0])
    prob_tf = [x[0] for x in list_lists2[1]]
    print('Resgs=', len(list_lists2[1]), len(prob_tf))

    res = dict(zip(wwids, prob_tf))

    df = pd.read_excel(data_fname, sheet_name='DataRequest_September 16-Septem')
    print(df.shape[0])
    df = df[df['WWID'].isin(res.keys())]
    df = df[df['Employee_Pay_Grade'] >= 20]
    print(df.shape[0])

    probs = []
    for w in df['WWID']:
        if w in res.keys():
            probs.append(res[w])

    df['Probs'] = probs

    df8 = pd.DataFrame()
    df8['WWID'] = df['WWID']
    df8['Probs'] = df['Probs']
    df8['Termination_Reason'] = df['Termination_Reason']
    df8['Tenure'] = df['Tenure']
    df8['Rating'] = df['Rating']
    df8['Sector'] = df['Sector_Fixed']
    df8['Function'] = df['Job_Function__IA__Host_All_Other']
    df8['Level'] = df['Level']
    df8['MgrRating'] = df['MgrRating']
    df8['PayGrade'] = df['Employee_Pay_Grade']
    df8['Tenure_time'] = df['Length_of_Service_in_Years_inclu']
    df8['Employee_Rating'] = df['Employee_Rating_1']

    df8.drop_duplicates(subset="WWID", keep='first', inplace=True)
    return df8


# df1 = read_probs('parrot_china_fixed.pkl', 'data_files/CHINA/China_2018.csv')
# df2 = read_probs_2019('parrot_china_fixed_2019.pkl', 'data_files/CHINA/China_2019.csv',
#                       'Termination data - 2019 - Jan-Sep.xlsx')
# df3 = read_probs_current('parrot_china_current_2019.pkl', 'Data Request_September 16 - September 19.xlsx')
#
# df4 = read_probs('parrot_sea_fixed.pkl', 'data_files/SEA/Sea_2018.csv')
# df5 = read_probs_2019('parrot_sea_fixed_2019.pkl', 'data_files/SEA/Sea_2019.csv',
#                       'data_files/SEA/SEA_Retention_Data_Jan-Sept2019_2018 changes added.xlsx')
# df6 = read_probs_current('parrot_sea_current_2019.pkl', 'Data Request_September 16 - September 19.xlsx')
#
# df7 = read_probs('predictions/parrot_brazil_fixed_08_Nov_2019_14_01_03.pkl', 'data_files/BRAZIL/Brazil_2018.csv')
# df8 = read_probs_2019('predictions/parrot_brazil_fixed_2019_08_Nov_2019_14_01_03.pkl', 'data_files/BRAZIL/Brazil_2019.csv',
#                       'data_files/BRAZIL/Brazil_retention_Sept2019data.xlsx')
# df9 = read_probs_current('predictions/parrot_brazil_current_2019_09_Nov_2019_13_41_47.pkl', 'Data Request_September 16 - September 19.xlsx')

df1 = pd.read_excel('output_all_countries_current_2019_6.xlsx', sheet_name='CHINA')
df2 = pd.read_excel('output_all_countries_current_2019_6.xlsx', sheet_name='CHINA_2019')
df3 = pd.read_excel('output_all_countries_current_2019_6.xlsx', sheet_name='CHINA_CURRENT')
df4 = pd.read_excel('output_all_countries_current_2019_6.xlsx', sheet_name='SEA')
df5 = pd.read_excel('output_all_countries_current_2019_6.xlsx', sheet_name='SEA_2019')
df6 = pd.read_excel('output_all_countries_current_2019_6.xlsx', sheet_name='SEA_CURRENT')
df7 = pd.read_excel('output_all_countries_current_2019_6.xlsx', sheet_name='BRAZIL')
df8 = pd.read_excel('output_all_countries_current_2019_6.xlsx', sheet_name='BRAZIL_2019')
df9 = pd.read_excel('output_all_countries_current_2019_6.xlsx', sheet_name='BRAZIL_CURRENT')


for d in [df1, df2, df3, df4, df5, df6, df7, df8]:
    d= d.drop("Unnamed: 0", axis=1)

with pd.ExcelWriter('output_all_countries_current_2019_10.xlsx') as writer:  # doctest: +SKIP
    df1.to_excel(writer, sheet_name='CHINA')
    df2.to_excel(writer, sheet_name='CHINA_2019')
    df3.to_excel(writer, sheet_name='CHINA_CURRENT')
    df4.to_excel(writer, sheet_name='SEA')
    df5.to_excel(writer, sheet_name='SEA_2019')
    df6.to_excel(writer, sheet_name='SEA_CURRENT')
    df7.to_excel(writer, sheet_name='BRAZIL')
    df8.to_excel(writer, sheet_name='BRAZIL_2019')
    df9.to_excel(writer, sheet_name='BRAZIL_CURRENT')
