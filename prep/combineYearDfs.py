import pandas as pd
import os, re

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def concatYearDfs(input_dir, output_dir, year_range=(5,18), extension='.sas7bdat'):
    ''' concatenate all KNHANES dfs for one particular year'''
    fnames = []
    start_year, end_year = year_range
    for year in range(start_year, end_year+1):
        year_str = str(year).zfill(2)
        fname = 'knhanes_{}_all_feats.csv'.format(year_str)
        fpath = os.path.join(output_dir, fname)
        fnames.append(fname)
        if os.path.exists(fpath):
            print('Skipping preexisting file', fpath)
            continue

        year_dfs = []
        for it in os.listdir(input_dir):
            if os.path.splitext(it)[-1] == extension and \
                int(re.findall("\d+", it)[0]) == year:
                    sas_path = os.path.join(input_dir, it)
                    print('reading in {}...'.format(it))
                    df = pd.read_sas(sas_path)
                    year_dfs.append(df)
        if year_dfs:
            print('--- concatenating dfs for year', year_str)
            print('concatenating...')
            year_df = pd.concat(year_dfs)
            print('converting to csv...')
            year_df.to_csv(fpath) #, chunksize=size)
            print('saved to', fpath)
    
    return fnames

# penultimate line        
