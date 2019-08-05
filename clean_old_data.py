import pandas as pd
import numpy as np

# df = pd.read_csv('Brazil_2015_filtered.csv', sep=',')


def write_to_pickle(year):
    df = pd.read_excel('./'+year+'.xlsx')
    print('Read old')
    df_comp = pd.read_excel('./'+year+'_Final_Compensation_Data.xlsx')
    df_comp['Wwid'] = df_comp['World-Wide ID']
    print('Read old comp')
    df_term = pd.read_excel('./Terminations_'+year+'.xlsx')
    df_term['Wwid'] = df_term['Employee_WWID']
    print('Read old terms')
    df.to_pickle('df_'+year+'.pkl')
    df_comp.to_pickle('df_comp_'+year+'.pkl')
    df_term.to_pickle('df_term_'+year+'.pkl')


def clean_dataframe(year):
    df = pd.read_pickle('df_'+year+'.pkl')
    df_comp = pd.read_pickle('df_comp_'+year+'.pkl')
    df_term = pd.read_pickle('df_term_'+year+'.pkl')

    print(df['Wwid'].head())
    print(df_comp['Wwid'].head())
    print(df_term['Wwid'].head())

    df.set_index('Wwid')
    df_comp.set_index('Wwid')
    df_term.set_index('Wwid')

    #df_term = df_term.drop('Job_Sub_Function', axis=1)

    df_comb = df.merge(df_comp, on='Wwid', how='left')
    df_comb = df_comb.merge(df_term, on='Wwid', how='left')

    for c in df_comb.columns:
        print(c)

    features = ['WWID',
                'Compensation_Range___Midpoint', 'Total_Base_Pay___Local',
                'Job_Sub_Function__IA__Host_All_O', 'Job_Function__IA__Host_All_Other',
                'Promotion', 'Demotion', 'Lateral', 'Cross_Move', 'Trainings_Completed',
                'Mgr_Change', 'SkipLevel_Mgr_Change',
                'Rehire_YN',
                '_018_Planned_as_a___of_Bonus_Tar', '_017_Planned_as_a___of_Bonus_Tar',
                '_016_Planned_as_a___of_Bonus_Tar',
                'Highest_Degree_Received',
                'Compa_Diff_Ratio', 'Compa_Ratio',
                'Sales_Incentive_2016', 'Sales_Incentive_2017', 'Sales_Incentive_2018',
                'Employee_Rating_1_W', 'Employee_Rating_1_H',
                'Employee_Rating_2_W', 'Employee_Rating_2_H',
                'Employee_Rating_3_W', 'Employee_Rating_3_H',
                'Manager_Rating_1_W', 'Manager_Rating_1_H',
                'Manager_Rating_2_W', 'Manager_Rating_2_H',
                'Manager_Rating_3_W', 'Manager_Rating_3_H',
                'Tenure',
                'Status',
                'Tenure_log']

    new = pd.DataFrame()
    new['WWID'] = df_comb['Wwid']
    new['Tenure'] = df_comb['Tenure']
    new['Tenure_log'] = np.log(new['Tenure'] + 1)
    new['Total_Base_Pay___Local'] = df_comb['Total_Comp']*df['Exchange_Rate']
    new['Rehire_YN'] = df_comb['Rehire_Flag']
    new['Compensation_Range___Midpoint'] = df_comb['Pay Grade Mid/Ref']
    new['Compa_Ratio'] = new['Total_Base_Pay___Local']/new['Compensation_Range___Midpoint']
    new['Compa_Diff_Ratio'] = (new['Total_Base_Pay___Local']-new['Compensation_Range___Midpoint'])\
                                              / new['Compensation_Range___Midpoint']

    new['Employee_Rating_1'] = df_comb['Performance_Code'].replace(7, 34).replace(6, 42).replace(5, 33)\
        .replace(0, 11).replace(1, 12).replace(2, 23).replace(3, 31).replace(4, 32)

    new['Employee_Rating_1_W'] = (new['Employee_Rating_1']/10).fillna(0.0).astype(str).str.extract('\.(.*)')
    new['Employee_Rating_1_H'] = (new['Employee_Rating_1']/10).fillna(0.0).astype(int)
    new = new.drop('Employee_Rating_1', axis=1)

    new['Manager_Rating_1'] = df_comb['Perf_Code_Mgr'].replace(7, 34).replace(6, 42).replace(5, 33)\
        .replace(0, 11).replace(1, 12).replace(2, 23).replace(3, 31).replace(4, 32)

    new['Manager_Rating_1_W'] = (new['Manager_Rating_1']/10).fillna(0.0).astype(str).str.extract('\.(.*)')
    new['Manager_Rating_1_H'] = (new['Manager_Rating_1']/10).fillna(0.0).astype(int)
    new = new.drop('Manager_Rating_1', axis=1)

    func_dic = {1: 'Engineering',
                2: 'Facilities',
                3: 'Finance',
                4: 'General Administration',
                5: 'General Management',
                6: 'Human Resources',
                7: 'Info Technology',
                8: 'Legal',
                9: 'Marketing',
                10: 'Non-Employee',
                11: 'Not assigned',
                12: 'Operations',
                13: 'Public Relations',
                14: 'Quality',
                15: 'R&D',
                16: 'Regulatory Affairs',
                17: 'Sales',
                18: 'Strategic Planning'}


    f_dics = []
    for f in df_comb['Function']:
        f_dics.append(func_dic[f])

    new['Job_Function__IA__Host_All_Other'] = f_dics

    sub_func_dic = {1:'Accounts Payable (1)',
    2:'Accounts Receivable (2)',
    3:'Administration (3)',
    4:'Advertising (4)',
    5:'Approval Liaison (5)',
    6:'Architecture (6)',
    7:'Aviation (7)',
    8:'Benefits (8)',
    9:'Biology (9)',
    10:'Biomedical Engineering (10)',
    11:'Biostatistics (11)',
    12:'Business Development (12)',
    13:'Business Relationship Management (13)',
    14:'Business Solutions (14)',
    15:'Chemistry (15)',
    16:'Clinical Data Management (16)',
    17:'Clinical Research MD (17)',
    18:'Clinical Research non-MD (18)',
    19:'Clinical Supplies (19)',
    20:'Clinical Trial Administration (20)',
    21:'Clinical Trial Coordination (21)',
    22:'Communications (22)',
    23:'Community (23)',
    24:'Compensation (24)',
    25:'Compliance (25)',
    26:'Compliance Security (26)',
    27:'Construction (27)',
    28:'Contract Administration (28)',
    29:'Contract Manufacturing (29)',
    30:'Corporate Law (30)',
    31:'Creative Design (31)',
    32:'Customer Education (32)',
    33:'Customer Info (33)',
    34:'Customer Service (34)',
    35:'Customer Technical Service (35)',
    36:'Database (36)',
    37:'Development (37)',
    38:'Diversity (38)',
    39:'Drug & Product Safety Operations (39)',
    40:'Drug & Product Safety Science (40)',
    41:'Duplicating (41)',
    42:'Electrical (42)',
    43:'Employee Relations (43)',
    44:'Employment (44)',
    45:'Engineering (Generalist) (45)',
    46:'Engineering (IT) (46)',
    47:'Environmental (47)',
    48:'Environmental Health & Safety (48)',
    49:'Epidemiology (49)',
    50:'Events Planning (50)',
    51:'Facilities (Eng) (51)',
    52:'Facilities (Generalist) (52)',
    53:'Finance (53)',
    54:'Food Services (54)',
    55:'General Accounting (55)',
    56:'General Management (56)',
    57:'General Services (57)',
    58:'Global Mobility (58)',
    59:'Government Affairs (59)',
    60:'Graphics/Technical Writing (60)',
    61:'HR Operations (61)',
    62:'Health & Safety (62)',
    63:'Health Care Compliance (63)',
    64:'Health Economics Mkt (64)',
    65:'Health Economics R&D (65)',
    66:'Human Resources (66)',
    67:'IT Compliance (67)',
    68:'IT Management (68)',
    69:'IT Process Mgmt and Control (69)',
    70:'IT Project Management (70)',
    71:'Industrial Manufacturing (71)',
    72:'Info Technology (72)',
    73:'Informatics (73)',
    74:'Information Security (74)',
    75:'Internal Audit (75)',
    76:'Internal Audit IT (76)',
    77:'Inventory Planning/Control (77)',
    78:'Investor Relations (78)',
    79:'Laboratory Animal Medicine (79)',
    80:'Legal (Non Attorney) (80)',
    81:'Library Services (81)',
    82:'Local Government (82)',
    83:'Mail Services (83)',
    84:'Market Research (84)',
    85:'Marketing (85)',
    86:'Marketing & Sales (86)',
    87:'Materials Management & Distributi (87)',
    88:'Materials Management & Distribution (88)',
    89:'Medical Affairs (89)',
    90:'Medical Science Liaison (90)',
    91:'Medical Writing (91)',
    92:'Mergers (FIN) (92)',
    93:'Mergers (SP) (93)',
    94:'National Accounts (94)',
    95:'New Product Intro-Life Cycle Mgmt (95)',
    96:'Non-Employee (96)',
    97:'Not assigned (97)',
    98:'Occupational Health (98)',
    99:'Office Space (99)',
    100:'Operations (Generalist) (100)',
    101:'Operations (IT) (101)',
    102:'Organizational Development (102)',
    103:'Packaging (103)',
    104:'Paralegal (104)',
    105:'Patent Liaison (105)',
    106:'Patents (106)',
    107:'Payroll (107)',
    108:'Pharmacokinetics (108)',
    109:'Pharmacovigilance (109)',
    110:'Planning & Analysis (110)',
    111:'Plant Management (111)',
    112:'Process Engineering (112)',
    113:'Process Excellence (113)',
    114:'Procurement (114)',
    115:'Product Development (115)',
    116:'Product Management (116)',
    117:'Product Research (117)',
    118:'Production (118)',
    119:'Production Maintenance (119)',
    120:'Production Planning (120)',
    121:'Quality (Eng) (121)',
    122:'Quality (Generalist) (122)',
    123:'Quality Assurance (123)',
    124:'Quality Control (124)',
    125:'Quality Systems (125)',
    126:'R&D (126)',
    127:'R&D Engineering (R&D) (127)',
    128:'Real Estate (128)',
    129:'Regulatory Affairs (129)',
    130:'Regulatory Compliance (130)',
    131:'Risk Management (131)',
    132:'Sales Administration (132)',
    133:'Sales Training (133)',
    134:'Security (134)',
    135:'Selling (135)',
    136:'Selling Consumer (136)',
    137:'Selling MD&D (137)',
    138:'Selling Pharmaceutical (138)',
    139:'Sourcing (139)',
    140:'Strategic Planning (140)',
    141:'Strategic Sourcing/Logistics (141)',
    142:'Submissions (142)',
    143:'Tax (143)',
    144:'Technical Assurance (144)',
    145:'Technology Engineering (145)',
    146:'Technology Operations (146)',
    147:'Telephone Services (147)',
    148:'Total Rewards (148)',
    149:'Toxicology Research (149)',
    150:'Trade Relations (150)',
    151:'Training & Development (151)',
    152:'Transportation Services (152)',
    153:'Treasury (153)',
    154:'Validation (154)',
    155:'Vendor Management (155)',
    156:'Video/Voice Communication (156)',
    157:'Warehousing (157)'}

    sf_dics = []
    for sf in df_comb['Job_Sub_Function']:
        sf_dics.append(sub_func_dic[sf].split('(')[0])

    new['Job_Sub_Function__IA__Host_All_O'] = sf_dics

    new['Planned_as_a___of_Bonus_Tar'] = df_comb['Planned_Bonus_Perc']/df_comb['Bonus_Target']

    new['Highest_Degree_Received'] = df_comb['EdLevel'].replace(1, 'High School').\
        replace(2, 'Vocational, Certificate, Technical or Associates').replace(3, 'University/Bachelors Degree or Equivalent').\
        replace(4, 'Masters Degree or Equivalent').replace(5, 'Doctorate (PHD) or Equivalent')

    #new['Status'] = [1 if x == 'Resignation' else 0 for x in df_comb['Action_Reason']]
    new['Status'] = df_comb['Vol_Flag']
    new['Promotion'] = df_comb['Promo_Flag']
    new['Demotion'] = df_comb['PG_Demote']
    new['Lateral'] = df_comb['Lateral_Flag']
    new['Mgr_Change'] = df_comb['Term_Flag_Mgr']

    new.info()
    print(new.head(20))

    new.to_csv(year+'_clean.csv', sep=',', encoding='utf-8')


if __name__ == '__main__':
    # write_to_pickle(year)
    #clean_dataframe('2009')
    #clean_dataframe('2010')
    #clean_dataframe('2011')
    #clean_dataframe('2012')
    #clean_dataframe('2013')
    #clean_dataframe('2014')
    clean_dataframe('2015')