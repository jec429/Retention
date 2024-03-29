from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import range


def testing(x_merged):
    df_x = x_merged[(x_merged['Report_Year'] == 2018) & (x_merged['Working_Country'] == 37)]
    df_x = df_x.drop(['Report_Year', 'Working_Country', 'Status'], axis=1)
    wwid = 1021037
    df_x = df_x.reset_index(drop=True)
    i = df_x.index[df_x['WWID'] == wwid].tolist()
    df_x = df_x.drop(['WWID'], axis=1)
    df_x = np.array(df_x.values)

    means = df_x.mean(0)
    stds = df_x.std(0)
    sel = df_x[i]
    new_sels = abs(((sel - means) / stds)[0])
    a = list(new_sels)
    top6 = [0, 0, 0, 0, 0, 0]

    for im in range(6):
        maxpos = a.index(max(a))
        top6[im] = maxpos
        a[maxpos] = 0

    print(top6)
    x = np.linspace(0, 2 * np.pi, 400)
    y = np.sin(x ** 2)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(df_x[:, top6[0]])
    axs[0, 0].set_title(x_merged.columns[top6[0]])
    axs[0, 1].plot(x, y, 'tab:orange')
    axs[0, 1].set_title('Axis [0, 1]')
    axs[1, 0].plot(x, -y, 'tab:green')
    axs[1, 0].set_title('Axis [1, 0]')
    axs[1, 1].plot(x, -y, 'tab:red')
    axs[1, 1].set_title('Axis [1, 1]')


def calculate_probabilities():
    import pickle
    from sklearn.preprocessing import StandardScaler
    import tensorflow.keras as keras
    x_merged = pd.read_pickle("./data_files/merged_Brazil_combined_x_numeric_newer.pkl")
    x = x_merged[(x_merged['Report_Year'] == 2018) & (x_merged['Working_Country'] == 37)]
    x = x.drop(['Report_Year', 'Working_Country', 'Compensation_Range___Midpoint'], axis=1)
    x = x.drop(['Status'], axis=1)
    x = x.reset_index(drop=True)
    wwids = x.WWID
    x = x.drop(['WWID'], axis=1)
    x = np.array(x.values)
    x2 = StandardScaler().fit_transform(x)
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    # prob_mlp = loaded_model.predict_proba(x2)
    # prob_1 = prob_mlp[:, 1]
    prob_2 = 0.0  # prob_1.reshape(3831, 1)
    new_model = keras.models.load_model('my_model_weighted.h5')
    prob_tf = new_model.predict(x2)

    return [wwids, prob_2, prob_tf]


def calculate_probability(wwid):
    import pickle
    fname = "parrot.pkl"
    with open(fname, "rb") as fin:
        list_lists2 = pickle.load(fin)
    wwids = list_lists2[0].to_list()
    prob_mlp = list_lists2[1]
    prob_tf = list_lists2[2]
    if wwid in wwids:
        index = wwids.index(wwid)
    else:
        return 'N/A'
    prob = prob_tf[index]
    if prob < 0.5:
        return 'Low Risk'
    elif prob < 0.9:
        return 'Medium Risk'
    else:
        return 'High Risk'
    # return '%.2f' % prob


def histo_feature(wwid, df):
    print('WWID=', wwid)
    df_x = df[(df['Report_Year'] == 2018) & (df['Working_Country'] == 37)]
    df_x = df_x.drop(['Report_Year', 'Working_Country', 'Status'], axis=1)
    df_x = df_x.reset_index(drop=True)
    df_x = df_x.replace(-999, np.nan)
    i = df_x.index[df_x['WWID'] == wwid].tolist()
    df_x = df_x.drop(['WWID'], axis=1)
    df_names = df_x.columns.str.replace('Job_Function__IA__Host_All_Other=', 'Function=')
    df_names = df_names.str.replace('Highest_Degree_Received=', 'Degree=')
    df_names = df_names.str.replace('_', ' ')
    df_x = np.array(df_x.values)

    means = np.nanmean(df_x, axis=0)
    stds = df_x.std(0)
    sel = df_x[i]
    new_sels = abs(((sel - means) / stds)[0])
    a = list(new_sels)
    top6 = [0, 0, 0, 0, 0, 0]

    for im in range(6):
        maxpos = a.index(max(a))
        top6[im] = maxpos
        a[maxpos] = 0

    print(top6)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(df_x[:, top6[0]], label='Value = %.2f' % sel[0][top6[0]])
    axs[0, 0].set_title(df_names[top6[0]])
    axs[0, 1].hist(df_x[:, top6[1]], label='Value = %.2f' % sel[0][top6[1]])
    axs[0, 1].set_title(df_names[top6[1]])
    axs[1, 0].hist(df_x[:, top6[2]], label='Value = %.2f' % sel[0][top6[2]])
    axs[1, 0].set_title(df_names[top6[2]])
    axs[1, 1].hist(df_x[:, top6[3]], label='Value = %.2f' % sel[0][top6[3]])
    axs[1, 1].set_title(df_names[top6[3]])

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()

    return fig


def get_pictures_and_links():
    from time import sleep
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    import urllib
    import pickle

    linkedin_username = 'jorge.chaves37@gmail.com'
    linkedin_password = ""

    # driver = webdriver.Chrome('/usr/local/bin/chromedriver')
    driver = webdriver.Chrome(r'C:\Users\jchaves6\PycharmProjects\Retention\chromedriver')
    driver.get('https://www.linkedin.com')

    username = driver.find_element_by_name('session_key')
    username.send_keys(linkedin_username)
    sleep(0.5)

    password = driver.find_element_by_name('session_password')
    password.send_keys(linkedin_password)
    sleep(0.5)

    sign_in_button = driver.find_element_by_class_name('sign-in-form__submit-btn')
    sign_in_button.click()
    sleep(0.5)

    employees = []

    df = pd.read_csv(r'C:\Users\jchaves6\PycharmProjects\Retention\data_files\Brazil_2018.csv', sep=',')
    ie = 0
    for w, p in zip(df['WWID'], df['Position__IA__Host_All_Other__Pr']):
        if ie > 20:
            break
        e = p.split('(')[0].strip().lower().replace('-', '')
        print(w, e)
        employees.append([w, e+' Johnson'])
        ie += 1

    websites = {}
    for e in employees:
        search = driver.find_element_by_class_name('search-global-typeahead__input')
        search.clear()
        search.send_keys(e[1])
        search.send_keys(Keys.RETURN)
        sleep(4)
        try:
            a = driver.find_elements_by_class_name("search-result")
            b = a[0].find_element_by_class_name('search-result__image')
            c = a[0].find_element_by_class_name('search-result__info')
            src = ''
            try:
                src = b.find_element_by_class_name('lazy-image').get_attribute('src')
            except:
                print('No photo')

            if len(src) > 0:
                urllib.request.urlretrieve(src,
                                           r'C:\Users\jchaves6\PycharmProjects\Retention\pics\ ' + str(e[0]) + ".png")
            else:
                from PIL import Image

                img = Image.new('RGB', (100, 100), color='gray')
                img.save(r'C:\Users\jchaves6\PycharmProjects\Retention\pics\ ' + str(e[0]) + '.png')

            print(c.find_element_by_css_selector('a').get_attribute('href'))
            website = c.find_element_by_css_selector('a').get_attribute('href')
            websites[e[0]] = website
        except:
            websites[e[0]] = 'N/A'

    with open(r'C:\Users\jchaves6\PycharmProjects\Retention\websites.pkl', 'wb') as handle:
        pickle.dump(websites, handle, protocol=pickle.HIGHEST_PROTOCOL)

    driver.quit()


def determine_regret():
    df_br = pd.read_csv('data_files/Brazil_2018.csv', sep=',')
    x_merged = pd.read_pickle("./data_files/merged_Brazil_combined_x_numeric_new.pkl")
    df1 = x_merged[x_merged['Report_Year']==2018].set_index('WWID')
    df2 = df_br.set_index('WWID')

    df2['Regrettable'] = 'No'
    regret = {}
    for w1 in df1.index:
        er1 = df1.at[w1, 'Employee_Rating_1']
        er2 = df1.at[w1, 'Employee_Rating_2']
        er3 = df1.at[w1, 'Employee_Rating_3']
        if er1 < 0:
            er1 = 0
        if er2 < 0:
            er2 = 0
        if er3 < 0:
            er3 = 0
        print(w1, er1, er2, er3)

        ten = df1.at[w1, 'Tenure']
        cpr = df1.at[w1, 'Compa_Ratio']
        if ten < 0:
            ten = 0
        if cpr < 0:
            cpr = 0
        rg = False
        if er1+er2+er3 > 7:
            rg = True
            print('rating')
        if ten < 5 and cpr > 1.5:
            rg = True
        if rg:
            regret[w1] = 'Yes'

    for k in regret.keys():
        # print(k)
        df2.at[k, 'Regrettable'] = 'Yes'

    print(df2['Regrettable'].value_counts())


if __name__ == '__main__':
    determine_regret()
