import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from prep.combineYearDfs import concatYearDfs

# def get_healthy_status_df(df, health_status_feats):
#     ''' get healthy status df '''
#     print('len rows', df.shape[0])
#     print('dropping unhealthy status rows...')
#     df = df[~df[[c for c in health_status_feats if c in df
#                 ]].isin([1]).any(axis=1)] # if 1 is in any, drop entire row
#     print('len rows', df.shape[0])

#     if df.shape[0] == 0:
#         print('0 rows after drop, skipping df...')
#         return None
#     return df

def get_nutri_df(df, feats):
    print('keeping only nutri and basic feats...')
    df = df[[c for c in feats if c in df]]
    return df if df.shape[0] > 1 else None

# Main
def makeDfAllYears(year_csv_fnames, input_dir, feats_list,
                    year_range=(5,18), age_range=(19,70), missing_rate_th=0.3):
    ''' 
    concatenate all tailored dfs after dropping cols above missing_rate_th
    '''
    nutri_useable_feats, basic_feats, health_status_feats = feats_list
    min_year, max_year = year_range
    min_age, max_age = age_range
    fname = 'knhanes_{}_{}_healthy_nutri_ltmr_{}_feats.csv'.format(
        str(int(min_year)).zfill(2), str(int(max_year)).zfill(2),
        str(int(missing_rate_th*10)).zfill(2))
    fpath = os.path.join(input_dir, fname)
    if os.path.exists(fpath): # safety net
        return

    print('Initiating preprocessing procedures:')
    df_allYears_ls = []
    for year_csv_fname in year_csv_fnames:
        year = int(year_csv_fname.split('_')[1]) 
        if year >= min_year and year <=max_year: # safety net
            year_csv_fpath = os.path.join(input_dir, year_csv_fname)
            print('---Starting on year', year)
            try:
                # read in csv with desired cols only
                print('Reading in {}...'.format(year_csv_fname))
                df = pd.read_csv(year_csv_fpath, usecols= lambda c: c in set(
                    nutri_useable_feats + basic_feats + health_status_feats))
                
                '''Do not exclude unhealthy rows: 
                    training on healthy only gives wrong results irl
                '''
                # # get healthy status rows
                # df = get_healthy_status_df(df, health_status_feats)
     
                # get nutri cols
                df = get_nutri_df(df, nutri_useable_feats + basic_feats)
     
                # drop rows w/ nanrate gt missing_rate_th
                print('dropping high nanrate rows...')
                df.dropna(axis=0, inplace=True, 
                    thresh=round(df.shape[1]*(1-missing_rate_th)))
                # print(df.shape) 

                if df.shape[0] == 0:
                    print('Null df with dropped rows. Skipping...')
                    continue

                # drop non number cols
                # TODO: double check if all numbers are not strings first ###
                df = df.select_dtypes(include=['number'])
                # filter age
                df = df[(df.age >= min_age) & (df.age <= max_age)]
                
                # skip virtually empty dfs
                if df.shape[1] <= 2 or df.shape[0]==0:
                    print('Not appending with shape of only {}...'.format(
                        year, df.shape))
                    continue
                df_allYears_ls.append(df)
                print('Appended {} rows with final shape of {}'.format(
                    year, df.shape))
            
            except:
                print('Unknown error!')

    # concat and save 
    if df_allYears_ls:
        df_allYears = pd.concat(df_allYears_ls)
        print('df_allYears shape:', df_allYears.shape)
        df_allYears.to_csv(fpath)
        print('Saved to', fpath)

# penultimate line
