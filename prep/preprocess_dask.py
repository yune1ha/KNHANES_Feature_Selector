import numpy as np
from matplotlib import pyplot as plt
import scipy
import os

import dask.dataframe as dd
from dask.distributed import Client
from dask.diagnostics import ProgressBar

# global
# paths
datapath = '/home/r3406/SharedFiles/Data/PublicData/KNHANES'
jar = 'jar'
NUM_CORES = os.cpu_count()

# feats
basic_mask = [ 
    #'year', 
    'age', 'sex',
]

nutri_useable_feats = [
    'n_wat_c',
    'nf_carot',
    'n_fm_wt',
    'nf_va',
    'nf_k',
    'n_kindg1',
    'n_kindg2',
    'nf_retin',
    'nf_na',
    'nf_phos',
    'nf_ca',
    'nf_intk',
    'nf_cho',
    'nf_en',
    'nf_water',
    'nf_fe',
    'nf_prot',
    'nf_fat',
    'nf_b1',
    'n_diet',
    'nf_vitc',
    'n_meal',
    'n_meal_t',
    'n_fcode',
    'n_mtype',
    'n_meal_p',
    'n_dcode',
    'nf_niac',
    'nf_b2',
    'n_dname',
    'n_ap',
    'n_fname',
    'n_day',
    'n_dusual',
    'n_cd_vol',
    'n_diet_why',
    'n_fname3',
    'n_fcode3',
    'nf_intk3',
    'n_td_vol',
    'n_fname2',
    'n_fcode2',
]

health_status_feats = [
    'DC1_dg',
    'DC11_dg',
    'DC12_dg',
    'DC2_dg',
    'DC3_dg',
    'DC4_dg',
    'DC5_dg',
    'DC6_dg',
    'DC7_dg',
    'DE1_dg',
    'DF2_dg',
    'DF2_dg',
    'DI1_dg',
    'DI3_dg',
    'DI4_dg',
    'DI5_dg',
    'DI6_dg',
    'DJ2_dg',
    'DJ4_dg',
    'DM1_dg',
    'DM2_dg',
    'DM3_dg',
    'DN1_dg',
    'DK4_dg',
]

white_list_prefix = [
    'D', 'H', 
]

def get_df_allYears(df_paths, prefix_feats, fullname_feats, 
                    year_range=(5,18), age_range=(19,70), missing_rate_th=0.3):
    ''' 
    concatenate all tailored dfs after dropping cols above missing_rate_th
    '''
    
    # perform on all dfs
    min_year, max_year = year_range
    min_age, max_age = age_range
    df_allYears_ls = []
    
    # initialize dask
    client = Client(asynchronous=True, n_workers=NUM_CORES, dashboard_address=':9988')

    with ProgressBar():
        for df_path in df_paths:
            year = int(df_path.split('_')[1]) 
            if year == min_year: #>= min_year and year <=max_year: 
                try:
                    # read in csv with desired cols only
                    print('Reading in {}...'.format(df_path[:5]))
                    df = dd.read_csv(df_paths[df_path]).compute()
                                    # usecols=lambda c: c.str.lower() in set(fullname_feats) or \
                                    # c.str.lower().startswith(tuple(prefix_feats)) )
                    print(df.shape)
                    # print('renaming columns...')
                    df.columns = df.columns.str.lower().compute()
                    break

                    # lowercase all feats
                    for ls in (prefix_feats, fullname_feats):
                        ls = [it.lower() for it in ls]
                    keep = [col for col in df.compute() if col.split('_')[0].startswith(tuple(prefix_feats))] + \
                        [col for col in fullname_feats if col in df]
                    df = df[keep].compute()

                    # get healthy status df
                    print('dropping unhealthy status rows...')
                    df = df[~df[[col for col in health_status_feats if col in df
                                ]].isin([1]).any(axis=1)].compute()
                    
                    # get df nutri
                    print('getting nutri and core dfs')
                    df_nutri = df[[col for col in nutri_useable_feats if col in df]]
                    df_core = df.drop(df_nutri.columns, axis=1).compute()

                    # #mem clear
                    # del df, df_nutri#temp 
                    # print('mem cleared...')

                    # drop cols w/ nanrate gt missing_rate_th
                    print('dropping unwanted cols on df core...')
                    df_core.dropna(axis=1, inplace=True, 
                        thresh=round(df_core.shape[0]*(1-missing_rate_th))).compute()

                    # drop non number cols
                    df_core = df_core.select_dtypes(include=['number']).compute()
                    # filter age
                    df_core = df_core[(df_core.age >= min_age) & (df_core.age <= max_age)].compute()
                    
                    # skip ['age','sex'] only dfs
                    if df_core.shape[1] <= 2:
                        print('Not adding {} with shape of only {}...'.format(
                            df_path[:5], df_core.shape))
                        continue
                    df_allYears_ls.append(df_core)
                    print('Added {} with final shape of {}!'.format(
                        df_path[:5], df_core.shape))
                
                except:
                    print('Unknown error!')

        # concat and save 
        if df_allYears_ls:
            df_allYears = dd.multi.concat(df_allYears_ls)
            print('df_allYears shape:', df_allYears.shape)
            
            fname = 'knhanes_05_18_healthy_all_ltmr{}_feats.pkl'.format(
                str(missing_rate_th*10))
            fpath = os.path.join(jar, fname)
            if not os.path.exists(fpath):
                df_allYears.to_csv(fpath)
                print('Saved df_allYears as', fpath)
            return df_allYears
        return None

# run

# get paths to all dfs # change split keyword to your needs
df_paths = {'df_' + filename.split('_')[1] + '_path':os.path.join(jar, filename) \
    for filename in os.listdir(jar) if filename.split('_')[2]=='all'}


prefix_feats = white_list_prefix
fullname_feats = basic_mask + nutri_useable_feats + health_status_feats

get_df_allYears(df_paths, prefix_feats, fullname_feats)
