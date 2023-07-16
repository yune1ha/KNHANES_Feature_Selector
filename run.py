'''
Performs concatenation and preprocessing on KNHANES raw data 
If preprocessed df already exists, then skips to perform feature selection

HIRCLab @ SNUBH, 2023-2-8
'''
import os, pathlib, json

from prep.combineYearDfs import concatYearDfs
from prep.preprocess import makeDfAllYears 
from featSelectors.factorAnalysis import factorAnalysis

## paths
curr_dir = pathlib.Path(__file__).parent.resolve()
# KNHANES raw data path of sas files. # needed only if starting from raw
raw_dir =  '/home/SharedFiles/Data/PublicData/KNHANES' 

interim_dir = os.path.join(curr_dir, 'interim')
reports_dir = os.path.join(curr_dir, 'reports')
features_dir = os.path.join(curr_dir, 'features')
[os.makedirs(p) for p in (interim_dir, reports_dir) if not os.path.exists(p)]

if raw_dir is None or not raw_dir:
    raw_dir = input('Enter path to raw KNHANES directory (top):')

# it's okay if this path doesn't exist yet:
preprocessed_data_path = os.path.join(interim_dir, 'knhanes_05_18_healthy_nutri_ltmr_03_feats.csv') 

# get features.json 
json_fpath = os.path.join(features_dir, 'features.json')
with open(json_fpath) as json_file:
    ft = json.load(json_file)

## Settings
YEAR_RANGE = (5, 18) # 2005 to 2018 inclusive # not designed for data from 20th cent.
AGE_RANGE = (19, 70) # 19 to 17 inclusive
MISSING_RATE = 0.3 # keep rows with missing rate below this amount

## Main
if not os.path.exists(preprocessed_data_path):
    # read in raw KNHANES sas files and combine into one csv for each year
    year_csv_fnames = concatYearDfs(raw_dir, interim_dir, YEAR_RANGE, extension='.sas7bdat')
    # print('--- year_csv_fnames', year_csv_fnames)

    # combine all years after only extracting desired features from each year
    makeDfAllYears(year_csv_fnames, interim_dir, 
                    feats_list = [
                        ft['nutri_useable_feats'], ft['basic_feats'], ft['health_status_feats'] ],
                    year_range=YEAR_RANGE, age_range=AGE_RANGE, missing_rate_th=MISSING_RATE)

# run feature selection

# white_list = ft['nutri_quantitative_feats'] + ft['nutri_nonquantitative_feats'] + ft['basic_feats'] 
white_list = ft['easily_obtainable_nutri_quantitative_feats'] + \
    ft['easily_obtainable_nutri_nonquantitative_feats'] + ft['basic_feats'] 

# pos_list = ft['easily_obtainable_nutri_quantitative_feats'] + ft['basic_feats'] 
# pos_list = ft['nutri_nonquantitative_feats'] + ft['basic_feats'] 
black_list = ['N_MEAL_P','N_MTYPE'] # 장소 has too high of a corr with age
feats = [it for it in white_list if it not in black_list]
factorAnalysis(preprocessed_data_path, reports_dir, 
               feats = feats,
               methods=['famd'],  # , 'mca'], # pca, mca, famd
               categoricals = ft['easily_obtainable_nutri_nonquantitative_feats']
              )

# Done
print('All processes complete.')

# penultimate line


'''
TODO:
* Correlation plot
* try famd with just easily obtainable quant feats (and non quant feats)
'''