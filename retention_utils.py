import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_scaler(region):
    SEA = (region == 'SEA')
    CHINA = (region == 'CHINA')
    ASIA = (region == 'ASIA')
    OURVOICE = (region == 'OURVOICE')
    print('Region 2019 = ', region)

    if SEA:
        # X_merged = pd.read_pickle("./data_files/SEA/merged_Sea_combined_x_numeric_newer.pkl")
        X_merged = pd.read_pickle("./data_files/SEA/merged_Sea_combined_fixed_x_numeric.pkl")
        raw_df = X_merged[(X_merged['Report_Year'] < 2018)]
        raw_df = raw_df.drop(['Report_Year', 'WWID', 'Total_Base_Pay___Local'], axis=1)
    elif CHINA:
        # X_merged = pd.read_pickle("./data_files/CHINA/merged_China_combined_x_numeric_newer.pkl")
        X_merged = pd.read_pickle("./data_files/CHINA/merged_China_combined_fixed_x_numeric.pkl")
        raw_df = X_merged[(X_merged['Report_Year'] < 2018)]
        raw_df = raw_df.drop(['Report_Year', 'WWID'], axis=1)
    elif ASIA:
        print('is ASIA')
        X_merged = pd.read_pickle("./data_files/merged_Asia_combined_x_numeric_newer.pkl")
        raw_df = X_merged[(X_merged['Report_Year'] < 2018)]
        raw_df = raw_df.drop(['Report_Year', 'WWID', 'Compensation_Range___Midpoint'], axis=1)
    elif OURVOICE:
        print('is OURVOICE')
        raw_df = pd.read_pickle("./data_files/OurVoice/ourvoice_merged_fixed_x_numeric_newer.pkl")
        raw_df.info()
        # Brazil = 7
        # China = 11
        # SEA = 64, 55, 72, 32, 44, 80
        # US = 72
        WORKING_COUNTRY = [11]
        # raw_df = raw_df[(raw_df['Working_Country_Fixed'].isin(WORKING_COUNTRY))]
        to_drop = ['Tenure']
        for c in raw_df.columns:
            if 'Location' in c or 'Survey_taken' in c:
                to_drop.append(c)
        raw_df = raw_df.drop('ID', axis=1)
        raw_df = raw_df.drop(to_drop, axis=1)
        raw_df.info()
        print(raw_df.shape[0])
    else:
        print('is BRAZIL')
        X_merged = pd.read_pickle("./data_files/merged_Brazil_combined_fixed_x_numeric_newer.pkl")
        X_merged[(X_merged['Report_Year'] < 2020) & (X_merged['Working_Country'] == 37)].info()
        raw_df = X_merged[(X_merged['Report_Year'] < 2018) & (X_merged['Working_Country'] == 37)]
        raw_df = raw_df.drop(['Report_Year', 'Working_Country', 'WWID', 'Compensation_Range___Midpoint'], axis=1)
        # X_merged = pd.read_pickle("./data_files/BRAZIL/merged_Brazil_combined_x_numeric_newer.pkl")
        # raw_df = X_merged[(X_merged['Report_Year'] < 2018)]
        # raw_df = raw_df.drop(['Report_Year', 'WWID', 'Compensation_Range___Midpoint'], axis=1)

    # print(raw_df.columns.to_list())
    to_drop = []  # x for x in raw_df.columns.to_list() if 'Location' in x or 'Function' in x]
    raw_df = raw_df.drop(to_drop, axis=1)
    raw_df['Status'] = raw_df['Status'].replace(0, -1)

    # Use a utility from sklearn to split and shuffle our dataset.
    train_df, test_df = train_test_split(raw_df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    # Form np arrays of labels and features.
    train_labels = np.array(train_df.pop('Status'))
    train_features = np.array(train_df)
    train_features[train_features == np.inf] = 999
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)

    return scaler

def get_data(region):
    SEA = (region == 'SEA')
    CHINA = (region == 'CHINA')
    ASIA = (region == 'ASIA')
    OURVOICE = (region == 'OURVOICE')
    print('Region = ', region)
    if SEA:
        # X_merged = pd.read_pickle("./data_files/SEA/merged_Sea_combined_x_numeric_newer.pkl")
        X_merged = pd.read_pickle("./data_files/SEA/merged_Sea_combined_fixed_x_numeric.pkl")
        raw_df = X_merged[(X_merged['Report_Year'] < 2018)]
        raw_df = raw_df.drop(['Report_Year', 'WWID', 'Total_Base_Pay___Local'], axis=1)
    elif CHINA:
        # X_merged = pd.read_pickle("./data_files/CHINA/merged_China_combined_x_numeric_newer.pkl")
        X_merged = pd.read_pickle("./data_files/CHINA/merged_China_combined_fixed_x_numeric.pkl")
        raw_df = X_merged[(X_merged['Report_Year'] < 2018)]
        raw_df = raw_df.drop(['Report_Year', 'WWID'], axis=1)
    elif ASIA:
        print('is ASIA')
        X_merged = pd.read_pickle("./data_files/merged_Asia_combined_x_numeric_newer.pkl")
        raw_df = X_merged[(X_merged['Report_Year'] < 2018)]
        raw_df = raw_df.drop(['Report_Year', 'WWID', 'Compensation_Range___Midpoint'], axis=1)
    elif OURVOICE:
        print('is OURVOICE')
        raw_df = pd.read_pickle("./data_files/OurVoice/ourvoice_merged_fixed_x_numeric_newer.pkl")
        raw_df.info()
        # Brazil = 7
        # China = 11
        # SEA = 64, 55, 72, 32, 44, 80
        # US = 72
        WORKING_COUNTRY = [11]
        # raw_df = raw_df[(raw_df['Working_Country_Fixed'].isin(WORKING_COUNTRY))]
        to_drop = ['Tenure']
        for c in raw_df.columns:
            if 'Location' in c or 'Survey_taken' in c:
                to_drop.append(c)
        raw_df = raw_df.drop('ID', axis=1)
        raw_df = raw_df.drop(to_drop, axis=1)
        raw_df.info()
        print(raw_df.shape[0])
    else:
        print('is BRAZIL')
        X_merged = pd.read_pickle("./data_files/merged_Brazil_combined_fixed_x_numeric_newer.pkl")
        X_merged[(X_merged['Report_Year'] < 2020) & (X_merged['Working_Country'] == 37)].info()
        raw_df = X_merged[(X_merged['Report_Year'] < 2018) & (X_merged['Working_Country'] == 37)]
        raw_df = raw_df.drop(['Report_Year', 'Working_Country', 'WWID', 'Compensation_Range___Midpoint'], axis=1)
        # X_merged = pd.read_pickle("./data_files/BRAZIL/merged_Brazil_combined_x_numeric_newer.pkl")
        # raw_df = X_merged[(X_merged['Report_Year'] < 2018)]
        # raw_df = raw_df.drop(['Report_Year', 'WWID', 'Compensation_Range___Midpoint'], axis=1)

    # print(raw_df.columns.to_list())
    to_drop = []  # x for x in raw_df.columns.to_list() if 'Location' in x or 'Function' in x]
    raw_df = raw_df.drop(to_drop, axis=1)
    raw_df['Status'] = raw_df['Status'].replace(0, -1)

    # Use a utility from sklearn to split and shuffle our dataset.
    train_df, test_df = train_test_split(raw_df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)

    # test_df = test_df[test_df['Working_Country_Fixed'] == 11]

    # Form np arrays of labels and features.
    train_labels = np.array(train_df.pop('Status'))
    val_labels = np.array(val_df.pop('Status'))
    test_labels = np.array(test_df.pop('Status'))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    train_features[train_features == np.inf] = 999
    val_features[val_features == np.inf] = 999
    test_features[test_features == np.inf] = 999
    # Normalize the input features using the sklearn StandardScaler.
    # This will set the mean to 0 and standard deviation to 1.
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)
    features = train_df.columns.to_list()

    print('Training labels shape:', train_labels.shape)
    print('Validation labels shape:', val_labels.shape)
    print('Test labels shape:', test_labels.shape)

    print('Training features shape:', train_features.shape)
    print('Validation features shape:', val_features.shape)
    print('Test features shape:', test_features.shape)

    neg, pos = np.bincount([1 if x == 1 else 0 for x in train_labels])
    total = neg + pos
    print('{} positive samples out of {} training samples ({:.2f}% of total)'.format(
       pos, total, 100 * pos / total))

    if SEA:
        # X_merged = pd.read_pickle("./data_files/SEA/merged_Sea_combined_x_numeric_newer.pkl")
        X_merged = pd.read_pickle("./data_files/SEA/merged_Sea_combined_fixed_x_numeric.pkl")
        x_one_jnj = pd.read_csv('data_files/SEA/Sea_2018.csv', sep=',')
        one_jnj_wwids = x_one_jnj[x_one_jnj['One JNJ Count'] == 'Yes']['WWID'].to_list()
        print(len(one_jnj_wwids))
        new_test_df = X_merged[(X_merged['Report_Year'] == 2018)]
        new_test_df = new_test_df[(X_merged['WWID'].isin(one_jnj_wwids))]
        new_test_df = new_test_df.drop(['Report_Year', 'Total_Base_Pay___Local'], axis=1)
    elif CHINA:
        # X_merged = pd.read_pickle("./data_files/CHINA/merged_China_combined_x_numeric_newer.pkl")
        X_merged = pd.read_pickle("./data_files/CHINA/merged_China_combined_fixed_x_numeric.pkl")
        new_test_df = X_merged[(X_merged['Report_Year'] == 2018)]
        new_test_df = new_test_df.drop(['Report_Year'], axis=1)
    elif ASIA:
        X_merged = pd.read_pickle("./data_files/merged_Asia_combined_x_numeric_newer.pkl")
        new_test_df = X_merged[(X_merged['Report_Year'] == 2018)]
        new_test_df = new_test_df.drop(['Report_Year', 'Compensation_Range___Midpoint'], axis=1)
    else:
        X_merged = pd.read_pickle("./data_files/merged_Brazil_combined_x_numeric_newer.pkl")
        new_test_df = X_merged[(X_merged['Report_Year'] == 2018) & (X_merged['Working_Country'] == 37)]
        new_test_df = new_test_df.drop(['Report_Year', 'Working_Country', 'Compensation_Range___Midpoint'], axis=1)
        # X_merged = pd.read_pickle("./data_files/BRAZIL/merged_Brazil_combined_x_numeric_newer.pkl")
        # new_test_df = X_merged[(X_merged['Report_Year'] == 2018)]
        # new_test_df = new_test_df.drop(['Report_Year', 'Compensation_Range___Midpoint'], axis=1)

    if OURVOICE:
        new_test_features = test_features
        new_test_labels = test_labels
        new_test_wwids = []
    else:
        new_test_df = new_test_df.drop(to_drop, axis=1)
        new_test_df['Status'] = new_test_df['Status'].replace(0, -1)
        # new_test_df.info()
        new_test_wwids = np.array(new_test_df.pop('WWID'))
        new_test_labels = np.array(new_test_df.pop('Status'))
        new_test_features = np.array(new_test_df)

        new_test_features[new_test_features == np.inf] = 999
        new_test_features = scaler.transform(new_test_features)

    return train_features, train_labels, test_features, test_labels, new_test_features, new_test_labels, features


def get_ourvoice():
    print('get OURVOICE')
    raw_df = pd.read_pickle("./data_files/OurVoice/ourvoice_merged_fixed_x_numeric_newer.pkl")
    raw_df.info()
    # Brazil = 7
    # China = 11
    # SEA = 64, 55, 72, 32, 44, 80
    # US = 72
    WORKING_COUNTRY = [11]
    # raw_df = raw_df[(raw_df['Working_Country_Fixed'].isin(WORKING_COUNTRY))]
    # raw_df = raw_df[(raw_df['Status'] != 1)]
    # raw_df['Status'] = raw_df['Status'].replace(2, 1)
    # raw_df = raw_df.drop(['Sector_Fixed=No Leader', 'Employee_Type__IA__Host_All_Othe=Fixed Term Employee'], axis=1)
    to_drop = ['Tenure']
    for c in raw_df.columns:
        if 'Location' in c or 'Survey_taken' in c:
            to_drop.append(c)
    raw_df = raw_df.drop('ID', axis=1)
    raw_df = raw_df.drop(to_drop, axis=1)
    raw_df.info()
    print(raw_df.shape[0], len(raw_df.columns.to_list()))
    train_df, test_df = train_test_split(raw_df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)

    train_labels = np.array(train_df.pop('Status'))

    train_features = np.array(train_df)
    scaler = StandardScaler()
    scaler.fit_transform(train_features)
    if True:
        print('is OURVOICE 2017')
        test_df_2017 = pd.read_pickle("./data_files/OurVoice/ourvoice_merged_fixed_x_numeric_newer.pkl")
        test_df_2017.info()
        # Brazil = 7
        # China = 11
        # SEA = 64, 55, 72, 32, 44, 80
        # US = 72
        WORKING_COUNTRY = [11]
        # raw_df = raw_df[(raw_df['Working_Country_Fixed'].isin(WORKING_COUNTRY))]
        # raw_df = raw_df[(raw_df['Status'] != 1)]
        # raw_df['Status'] = raw_df['Status'].replace(2, 1)
        # raw_df = raw_df.drop(['Sector_Fixed=No Leader', 'Employee_Type__IA__Host_All_Othe=Fixed Term Employee'], axis=1)
        to_drop = ['Tenure']
        for c in test_df_2017.columns:
            if 'Location' in c or 'Survey_taken' in c:
                to_drop.append(c)
        # test_df_2017 = test_df_2017.drop('ID', axis=1)
        id_2017 = np.array(test_df_2017.pop('ID'))
        test_df_2017 = test_df_2017.drop(to_drop, axis=1)
        train_labels = np.array(test_df_2017.pop('Status'))
        test_df_2017.info()
        print(test_df_2017.shape[0], len(test_df_2017.columns.to_list()))
        test_features_2017 = np.array(test_df_2017)
        test_features_2017 = scaler.transform(test_features_2017)

        print('is OURVOICE 2019')
        pre_test_df_2019 = pd.read_pickle("./data_files/OurVoice/ourvoice_merged_2019_fixed_x_numeric_newer.pkl")
        pre_test_df_2019.info()
        # Brazil = 7
        # China = 11
        # SEA = 64, 55, 72, 32, 44, 80
        # US = 72
        WORKING_COUNTRY = [11]
        # raw_df = raw_df[(raw_df['Working_Country_Fixed'].isin(WORKING_COUNTRY))]
        # raw_df = raw_df[(raw_df['Status'] != 1)]
        # raw_df['Status'] = raw_df['Status'].replace(2, 1)
        # raw_df = raw_df.drop(['Sector_Fixed=No Leader', 'Employee_Type__IA__Host_All_Othe=Fixed Term Employee'], axis=1)
        id_2019 = np.array(pre_test_df_2019.pop('ID'))
        test_df_2019 = pd.DataFrame()
        for c in test_df_2017.columns:
            if c in pre_test_df_2019.columns.to_list():
                test_df_2019[c] = pre_test_df_2019[c]
            else:
                test_df_2019[c] = 0
        test_df_2019.info()
        print(test_df_2017.shape[0], len(test_df_2019.columns.to_list()))
        test_features_2019 = np.array(test_df_2019)
        test_features_2019 = scaler.transform(test_features_2019)

    return test_features_2017, test_features_2019, id_2017, id_2019


def get_2019_data(region):
    SEA = (region == 'SEA')
    CHINA = (region == 'CHINA')
    ASIA = (region == 'ASIA')
    OURVOICE = (region == 'OURVOICE')
    print('Region 2019 = ', region)

    scaler = get_scaler(region)

    # 2019 test dataset
    if CHINA:
        # X_merged = pd.read_pickle("./data_files/CHINA/merged_China_combined_x_numeric_newer.pkl")
        X_merged = pd.read_pickle("./data_files/CHINA/merged_China_combined_fixed_x_numeric.pkl")
    elif SEA:
        print('SEA')
        # X_merged = pd.read_pickle("./data_files/SEA/merged_Sea_combined_x_numeric_newer.pkl")
        X_merged = pd.read_pickle("./data_files/SEA/merged_Sea_combined_fixed_x_numeric.pkl")
        x_one_jnj = pd.read_csv('data_files/SEA/Sea_2019.csv', sep=',')
        one_jnj_wwids = x_one_jnj[x_one_jnj['One JNJ Count'] == 'Yes']['WWID'].to_list()
        print(len(one_jnj_wwids))
        # X_merged = X_merged[(X_merged['WWID'].isin(one_jnj_wwids))]
    new_test_df = X_merged[(X_merged['Report_Year'] == 2019)]
    new_test_df = new_test_df.drop(['Report_Year', 'Total_Base_Pay___Local'], axis=1)
    # new_test_df = new_test_df.drop(to_drop, axis=1)
    new_test_df.info()
    if CHINA:
        df_res_2019 = pd.read_excel('./data_files/CHINA/ChinaData_Jan-June 2019.xlsx', sheet_name='Data')
    elif SEA:
        df_res_2019 = pd.read_excel('./data_files/SEA/SEA_Retention_Data_Jan-Sept2019_2018 changes added.xlsx',
                                    sheet_name='Data')
    res_wwids = list(df_res_2019[df_res_2019['Termination_Reason'] == 'Resignation']['WWID'])
    # print('res=', res_wwids)
    new_test_wwids = np.array(new_test_df.pop('WWID'))
    new_test_labels = np.array(new_test_df.pop('Status'))
    new_test_labels2 = [1 if x in res_wwids else 0 for x in new_test_wwids]
    new_test_features = np.array(new_test_df)
    new_test_features[new_test_features == np.inf] = 999
    new_test_features = scaler.transform(new_test_features)

    return new_test_features, new_test_wwids, new_test_labels2


def get_current_data(region):
    SEA = (region == 'SEA')
    CHINA = (region == 'CHINA')
    ASIA = (region == 'ASIA')
    OURVOICE = (region == 'OURVOICE')
    print('Region Current = ', region)

    scaler = get_scaler(region)
    # 2019 current test dataset
    if CHINA:
        print('China current')
        X_merged = pd.read_pickle("./data_files/CHINA/merged_China_combined_current_x_numeric_newer.pkl")
        X_merged_original = pd.read_pickle("./data_files/CHINA/merged_China_combined_x_numeric_newer.pkl")
        merged_original_features = X_merged_original.columns.to_list()
        df_current = pd.DataFrame()
        for c in merged_original_features:
            if c in X_merged.columns.to_list():
                df_current[c] = X_merged[c]
            else:
                df_current[c] = 0
    elif SEA:
        print('Sea current')
        X_merged = pd.read_pickle("./data_files/SEA/merged_Sea_combined_current_x_numeric_newer.pkl")
        X_merged_original = pd.read_pickle("./data_files/SEA/merged_Sea_combined_fixed_x_numeric.pkl")
        merged_original_features = X_merged_original.columns.to_list()
        df_current = pd.DataFrame()
        for c in merged_original_features:
            if c in X_merged.columns.to_list():
                df_current[c] = X_merged[c]
            else:
                df_current[c] = 0
    new_test_df = df_current[(df_current['Report_Year'] == 2019)]
    new_test_df = new_test_df.drop(['Report_Year', 'Status', 'Total_Base_Pay___Local'], axis=1)
    # new_test_df = new_test_df.drop(to_drop, axis=1)
    new_test_df.info()

    new_test_wwids = np.array(new_test_df.pop('WWID'))

    new_test_features = np.array(new_test_df)
    new_test_features[new_test_features == np.inf] = 999
    new_test_features = scaler.transform(new_test_features)

    return new_test_features, new_test_wwids
