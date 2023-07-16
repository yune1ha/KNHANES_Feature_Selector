''' *** contains various factor analysis methods from prince library
* All your variables are numeric: use principal component analysis (prince.PCA)
* You have a contingency table: use correspondence analysis (prince.CA)
* You have more than 2 variables and they are all categorical: use multiple correspondence analysis (prince.MCA)
* You have groups of categorical or numerical variables: use multiple factor analysis (prince.MFA)
* You have both categorical and numerical variables: use factor analysis of mixed data (prince.FAMD)
'''

import prince
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *

from joblib import parallel_backend
import os

# settings
class G:
    NUM_CORES = os.cpu_count()
    RANDOM_STATE = 42

# methods
def getMethodOutcomes(method, n_components, X_train, X_test, 
                      n_iter=3, random_state=G.RANDOM_STATE):
    '''
    ### Citations ###
    @software{Halford_Prince,
        author = {Halford, Max},
        license = {MIT},
        title = {{Prince}},
        url = {https://github.com/MaxHalford/prince}
    }
    '''
    if method == 'pca': # use of non categorical
        analysis = prince.PCA(
            n_components=n_components,
            n_iter=n_iter,
            rescale_with_mean=True,
            rescale_with_std=True,
            copy=True,
            check_input=True,
            engine='fbpca', # pip install fbpca
            random_state=random_state)
        
    elif method == 'mca': # use if categorical
        analysis = prince.MCA(
            n_components=n_components,
            n_iter=n_iter,
            copy=True,
            check_input=True,
            engine='auto',
            random_state=random_state)
            
    elif method == 'famd': # use if mixed, but drop categoricals before fitting
        analysis = prince.FAMD(
            n_components=n_components,
            n_iter=n_iter,
            copy=True,
            check_input=True,
            engine='auto',
            random_state=random_state)
    
    if method == 'famd':
        # TODO: r_ merge with numpy, resplit at saved index
        # X_train = pd.DataFrame(X_train)
        # idx = len(X_train)
        X_test = pd.DataFrame(X_test)
        # X_df = pd.concat([X_train, X_test])
        # print('---Old df shape:', X_df.shape)
        analysis = analysis.fit(X_test)
        X_test_new = analysis.transform(X_test)
        # print('---New df shape:', X_new_df.shape)

        X_train_new = X_train
        # X_train_new = X_new_df.iloc[:idx].values
        # X_test_new = X_new_df.iloc[idx:].values
        
        # print('---TEST: old new length eq train/test:', 
        #       len(X_df) == len(X_new_df) )
        # print(len(X_df), len(X_new_df))

    else:
        analysis = analysis.fit(X_train)
        X_train_new = analysis.transform(X_train)
        X_test_new = analysis.transform(X_test)

    V_ = analysis.V_

    print('---V_.shape', V_.shape) # TEST
    print('---V_', V_) # TEST
        

    return V_, X_train_new, X_test_new

def getScore(X_train, y_train, X_test, y_test):
    # LogReg Pred Scores
    lr = LinearRegression(n_jobs=G.NUM_CORES)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    # evaluate the model and collect the score
    # report the model performance
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # cnf_matrix = confusion_matrix(y_test, y_pred)
    # print(cnf_matrix)
    # target_names = [str(n) for n in range(19, 70+1)]
    # report = classification_report(y_test, y_pred, target_names=target_names)
    # print(report)

    return round(r2,4), round(rmse,4)

def getRVdf(method, initial_feature_names, X_train, X_test, y_train, y_test):
    r2_ls, rmse_ls, n_components_ls, n_unique_components_ls, components_ls = [],[],[],[],[]
    num_features = len(initial_feature_names)
    for n_components in range(num_features, 0, -1): # start with all features
        V_, X_train_new, X_test_new = getMethodOutcomes(method, n_components, X_train, X_test)
        r2, rmse = getScore(X_train_new, y_train, X_test_new, y_test)
        
        most_important = np.abs(V_).argmax(axis=1)
        print('--- most imp', most_important)
        most_important_names = initial_feature_names[most_important]
        # print('--- most imp names', most_important_names)
        
        # get num components excluding duplicates
        n_unique_components = len(np.unique(np.array(most_important_names)))
        # append as dict entry
        components = { 
            'PC_{}'.format(i+1):name for i, name in enumerate(most_important_names)} 
        # append all entries to corresponding lists
        for val, val_ls in zip(
            [r2, rmse, n_components, n_unique_components, components], 
            [r2_ls, rmse_ls, n_components_ls, n_unique_components_ls, components_ls]):
                val_ls.append(val)
                
    # create final dataframe
    out_df = pd.DataFrame(
        list(zip(n_components_ls, n_unique_components_ls, r2_ls, rmse_ls)), 
        columns=['N_Components', 'N_Unique_Components', 'R2', 'RMSE'])
    out_df = out_df.join(pd.DataFrame(components_ls))

    return out_df.round(4)

def get_factor_analysis_df(df, method, categoricals, target='age'):
    print('input shape:', df.shape)
    print('Performing factor analysis...')

    y = np.ravel(df[[target]])#.values
    X_df = df.drop([target], axis=1).astype(float)
    del df

    if categoricals:
        categoricals = [col for col in categoricals if col in X_df.columns]
        X_df[categoricals] = X_df[categoricals].astype('object', errors='ignore')
        print('---TEST: num cat:{}, num tot:{}'.format(len(categoricals), X_df.shape[1]))

    print('Inputted num features:', X_df.shape[1])

    if method == 'famd':
        X = X_df
    else:
        X = X_df.values

    print('---TEST dtypes:')
    print(X_df.dtypes)

    X_train, X_test, y_train, y_test = train_test_split(X,y ,
                                    random_state=G.RANDOM_STATE, 
                                    test_size=0.2, 
                                    shuffle=True)
    print('Shapes of X_train,', 'X_test,', 'y_train,', 'y_test:')
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # components for all retained_ths
    initial_feature_names = X_df.columns
    print('---initial_feature_names', initial_feature_names)
    rv_df = getRVdf(method, initial_feature_names, X_train, X_test, y_train, y_test) 

    return rv_df

        
# Main
def factorAnalysis(input_fpath, output_dir, feats, methods, categoricals=None):
    ''' 
    Data from 'knhanes_05_18_healthy_nutri_ltmr03_feats.csv'
    --explanation:
    * 05_18: from 2005 to 2018 inclusive
    * healthy: healthy status only (excluded all rows with any unhealthy status (morbidity == 1))
    * nutri_ltmr03: nutrition features with missing rate less than 30%
    * data must have 'sex' column with 1 as Male and 2 as Female
    '''
    print('Initiating feature selection...')
    df = pd.read_csv(input_fpath)

    # drop na rows 
    df = df.dropna(axis=0, how='any')

    # check if desired features exist in df
    avail_feats = [c for c in feats if c in df.columns]
    print('num available feats:', len(avail_feats))
    df= df[avail_feats]

    ## separate sex dfs
    # df_m = df[df.sex == 1].drop(['sex'], axis=1)
    # df_f = df[df.sex == 2].drop(['sex'], axis=1)
    df_all = df.drop(['sex'], axis=1)
    # print('m: {}, f: {}, all: {}'.format(df_m.shape, df_f.shape, df_all.shape) )

    # get pca and save df
    for df, cohort in zip(
        [df_all, 
            # df_m, df_f
        ], 
        ['All', 
            # 'M', 'F'
        ]):
        print('--- Gender:', cohort)
        with parallel_backend('threading', n_jobs=G.NUM_CORES):
            for method in methods:
                print('--- Method:', method)
                out_df = get_factor_analysis_df(df, method, categoricals)
                fname = 'nutri_{}_{}.csv'.format(method, cohort)
                fpath = os.path.join(output_dir, fname)
                out_df.to_csv(fpath)
                print('Saved to', fpath)

# penultimate line
