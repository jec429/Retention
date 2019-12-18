import pandas as pd
import pickle


def read_probs(parrot_fname, data_fname):
    with open(parrot_fname, "rb") as fin:
        list_lists2 = pickle.load(fin)
    wwids = list(list_lists2[0])
    print('Resgs=', len(list_lists2[2]), sum(list_lists2[2]))
    prob_tf = [[x[0], y] for x, y in zip(list_lists2[1], list_lists2[2])]

    res = dict(zip(wwids, prob_tf))

    # df = pd.read_csv('data_files/CHINA/China_2018.csv', sep=',')
    df = pd.read_csv(data_fname, sep=',')
    print(df.shape[0])
    df = df[df['WWID'].isin(res.keys())]

    probs = []
    for w in df['WWID']:
        if w in res.keys():
            probs.append(res[w][0])
    df['Probs'] = probs

    df2 = pd.DataFrame()
    df2['WWID'] = df['WWID']
    df2['Probs'] = df['Probs']
    df2['Termination_Category'] = df['Termination_Category']
    df2['Tenure'] = df['Tenure']
    df2['Rating'] = df['Rating']
    df2['Sector'] = df['Sector_Fixed']
    df2['Function'] = df['Job_Function__IA__Host_All_Other']
    df2['Level'] = df['Level']
    df2['MgrRating'] = df['MgrRating']
    df2['PayGrade'] = df['Employee_Pay_Grade']
    df2['Tenure_time'] = df['Length_of_Service_in_Years_inclu']
    df2['Employee_Rating'] = df['Employee_Rating_1']
    df2['Working_Country'] = df['Working_Country_Fixed']
    df2['MC_Member'] = df['MC_Member_Fixed']
    df2['TP'] = ((df['Probs'] > 0.5) & (df['Termination_Category'].str.contains('Term'))).astype('category').cat.codes
    df2['FP'] = ((df['Probs'] > 0.5) & (df['Termination_Category'].str.contains('Active'))).astype('category').cat.codes
    df2['TN'] = ((df['Probs'] < 0.5) & (df['Termination_Category'].str.contains('Active'))).astype('category').cat.codes
    df2['FN'] = ((df['Probs'] < 0.5) & (df['Termination_Category'].str.contains('Term'))).astype('category').cat.codes

    df2.drop_duplicates(subset="WWID", keep='first', inplace=True)

    return df2


if __name__ == '__main__':
    df2 = read_probs('./predictions/parrot_D2Plus_Selected_27_Nov_2019_08_03_20.pkl', './data_files/D2Plus/D2Plus_2018.csv')
    df3 = read_probs('./predictions/parrot_D2Plus_2019_Selected_27_Nov_2019_08_03_20.pkl', './data_files/D2Plus/D2Plus_2019.csv')

    with pd.ExcelWriter('output_D2Plus_2019_5.xlsx') as writer:  # doctest: +SKIP
        df2.to_excel(writer, sheet_name='2018')
        df3.to_excel(writer, sheet_name='2019')
