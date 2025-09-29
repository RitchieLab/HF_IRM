# load some packages
import sys
import subprocess

# install packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "imblearn"])

# load more packages
import pandas as pd
import argparse as ap
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from collections import Counter

# parse arguments
def make_arg_parser():
    parser = ap.ArgumentParser(description = ".")
    
    parser.add_argument('--input', required = True, help = 'input filename')
    
    parser.add_argument('--sig', required = True, help = 'significant filename')
    
    parser.add_argument('--important', required = True, help = 'important filename')
    
    parser.add_argument('--beta', required = True, help = 'beta filename')
    
    parser.add_argument('--iter', required = True, help = 'iteration number')
    
    return parser

args = make_arg_parser().parse_args()

# parse arguments
input_filename = args.input
sig_filename = args.sig
important_filename = args.important
beta_filename = args.beta
iter = int(args.iter)

# create iteration column name
print(iter)
colname = 'ITER_' + str(iter)

# read in input files
hf_balanced = pd.read_csv(input_filename)
significant_95 = pd.read_csv(sig_filename, index_col = 0)
important_95 = pd.read_csv(important_filename, index_col = 0)
beta = pd.read_csv(beta_filename, index_col = 0, usecols = ['Feature', colname])

# set significance threshold
corrected_sig = 0.05

# make CRS column list
crs_cols = ['T2D',
            'TRIG_INV_NORMAL_SCALE',
            'LDL_INV_NORMAL_SCALE',
            'HDL_INV_NORMAL_SCALE',
            'GLUCOSE_INV_NORMAL_SCALE',
            'HbA1c_INV_NORMAL_SCALE',
            'SBP_INV_NORMAL_SCALE',
            'DBP_INV_NORMAL_SCALE']

# filter CRS columns to those that are important and significant
crs_cols = [item for item in crs_cols if item in significant_95.index]
crs_cols = [item for item in crs_cols if item in important_95.index]

# create weighted column list
crs_weighted_cols = [col.replace(col, (col + '_WEIGHTED')) for col in crs_cols]

# make PXS column list
pxs_cols = ['BMI_INV_NORMAL_SCALE',
            'SMOKING_SCALE',
            'PA_dur_walk_INV_NORMAL_SCALE',
            'PA_dur_vig_activity_INV_NORMAL_SCALE',
            'PA_dur_mod_activity_INV_NORMAL_SCALE',
            'PA_n_days_walk_10+_mins_SCALE',
            'PA_n_days_mod_activity_10+_mins_SCALE',
            'PA_n_days_vig_activity_10+_mins_SCALE',
            'PA_freq_walk_pleasure_SCALE',
            'PA_freq_stair_SCALE',
            'PA_freq_other_exercises_SCALE',
            'PA_freq_heavy_diy_SCALE',
            'PA_freq_light_diy_SCALE',
            'PA_dur_heavy_diy_SCALE',
            'PA_dur_light_diy_SCALE',
            'PA_dur_walk_pleasure_SCALE',
            'PA_dur_oth_exercises_SCALE',
            'PA_sum_min_activity_INV_NORMAL_SCALE',
            'PA_sum_days_activity_INV_NORMAL_SCALE',
            'PA_sum_met_mins_week_INV_NORMAL_SCALE',
            'PA_met_mins_week_walk_INV_NORMAL_SCALE',
            'PA_met_mins_week_mod_INV_NORMAL_SCALE',
            'PA_met_mins_week_vig_INV_NORMAL_SCALE',
            'PA_at_above_mod_vig_rec',
            'PA_at_above_mod_vig_walk_rec',
            'INCOME_SCALE',
            'EDUCATION_SCALE',
            'HOUSING_SCALE',
            'TOWNSEND_DEP_INDEX_INV_NORMAL_SCALE']

# filter PXS columns to those that are important and significant
pxs_cols = [item for item in pxs_cols if item in significant_95.index]
pxs_cols = [item for item in pxs_cols if item in important_95.index]

# create weighted column list
pxs_weighted_cols = [col.replace(col, (col + '_WEIGHTED')) for col in pxs_cols]

# create eval column list
eval_col_list = ['PGS',
                 'CRS_SUM',
                 'CRS_WEIGHTED_SUM',
                 'PXS_SUM',
                 'PXS_WEIGHTED_SUM',
                 ['PGS', 'CRS_SUM', 'PXS_SUM'],
                 ['CRS_SUM', 'PXS_SUM'],
                 ['PGS', 'CRS_SUM'],
                 ['PGS', 'PXS_SUM'],
                 ['PGS', 'CRS_WEIGHTED_SUM', 'PXS_WEIGHTED_SUM'],
                 ['CRS_WEIGHTED_SUM', 'PXS_WEIGHTED_SUM'],
                 ['PGS', 'CRS_WEIGHTED_SUM'],
                 ['PGS', 'PXS_WEIGHTED_SUM'],
                 (['PGS'] + crs_cols + pxs_cols),
                 crs_cols,
                 pxs_cols,
                 (crs_cols + pxs_cols)]
    
# split dataset
hf_train = hf_balanced.sample(frac = 0.7, random_state = iter)
hf_reg = hf_train.sample(frac = 0.5, random_state = iter)
hf_lasso = hf_train.drop(hf_reg.index)
hf_no_train = hf_balanced.drop(hf_train.index)

# compute weighted columns
for col in (crs_cols + pxs_cols):
    weighted_colname = col + '_WEIGHTED'
    beta_val = beta.loc[col, colname]
    hf_no_train[weighted_colname] = hf_no_train[col] * beta_val

# compute integrated scores
hf_no_train['CRS_SUM'] = hf_no_train[crs_cols].sum(axis = 1, min_count = 1)
hf_no_train['CRS_WEIGHTED_SUM'] = hf_no_train[crs_weighted_cols].sum(axis = 1, min_count = 1)

hf_no_train['PXS_SUM'] = hf_no_train[pxs_cols].sum(axis = 1, min_count = 1)
hf_no_train['PXS_WEIGHTED_SUM'] = hf_no_train[pxs_weighted_cols].sum(axis = 1, min_count = 1)

# split testing dataset
hf_val = hf_no_train.sample(frac = 0.5, random_state = iter)
hf_test = hf_no_train.drop(hf_val.index)

# apply SMOTE
x = hf_val.drop(columns = ['HF'])
y = hf_val[['HF']]
x_resampled, y_resampled = SMOTE(random_state = iter).fit_resample(x, y)
hf_val = pd.concat([x_resampled, y_resampled], axis = 1)

# create empty dictionaries
auroc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in eval_col_list}
auprc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in eval_col_list}
f1_dict = {tuple(item) if isinstance(item, list) else item: [] for item in eval_col_list}
balanced_acc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in eval_col_list}

# evaluate models
for col in eval_col_list:
    if isinstance(col, str):
        model_df = hf_val[['HF', 'AGE', 'SEX'] + [col]].dropna()
        predictors = ['AGE', 'SEX'] + [col]
    elif isinstance(col, list):
        model_df = hf_val[['HF', 'AGE', 'SEX'] + col].dropna()
        predictors = ['AGE', 'SEX'] + col
    else:
        raise ValueError("conditions not met")
        
    all_cols = ['HF'] + predictors
        
    if len(model_df.index) == 0:
        print('skipping, all values are zero')
        if isinstance(col, str):  
            auroc_dict[col].append(np.nan)
            auprc_dict[col].append(np.nan)
            f1_dict[col].append(np.nan)
            balanced_acc_dict[col].append(np.nan)
        elif isinstance(col, list):
            auroc_dict[tuple(col)].append(np.nan)
            auprc_dict[tuple(col)].append(np.nan)
            f1_dict[tuple(col)].append(np.nan)
            balanced_acc_dict[tuple(col)].append(np.nan)
        else:
            raise ValueError("conditions not met")
        
    else:
        model = sm.Logit(model_df['HF'], model_df[predictors]).fit()
        test_df = hf_test[all_cols].dropna()
        y_prob_cont = model.predict(test_df[predictors])
        y_prob_bin = (y_prob_cont >= 0.5).astype(int)
        auroc = roc_auc_score(test_df['HF'], y_prob_cont)
        auprc = average_precision_score(test_df['HF'], y_prob_cont)
        f1 = f1_score(test_df['HF'], y_prob_bin)
        balanced_acc = balanced_accuracy_score(test_df['HF'], y_prob_bin)
        if isinstance(col, str):
            auroc_dict[col].append(auroc)
            auprc_dict[col].append(auprc)
            f1_dict[col].append(f1)
            balanced_acc_dict[col].append(balanced_acc)
        elif isinstance(col, list):
            auroc_dict[tuple(col)].append(auroc)
            auprc_dict[tuple(col)].append(auprc)
            f1_dict[tuple(col)].append(f1)
            balanced_acc_dict[tuple(col)].append(balanced_acc)
        else:
            raise ValueError("conditions not met")

# make output dfs
auroc_df = pd.DataFrame.from_dict(auroc_dict, orient = 'index', columns = [colname])
auprc_df = pd.DataFrame.from_dict(auprc_dict, orient = 'index', columns = [colname])
f1_df = pd.DataFrame.from_dict(f1_dict, orient = 'index', columns = [colname])
balanced_acc_df = pd.DataFrame.from_dict(balanced_acc_dict, orient = 'index', columns = [colname])
    
# export dfs
auroc_df.to_csv(('AUROC_' + str(iter) + '.txt'), sep = '\t')
auprc_df.to_csv(('AUPRC_' + str(iter) + '.txt'), sep = '\t')
f1_df.to_csv(('F1_SCORE_' + str(iter) + '.txt'), sep = '\t')
balanced_acc_df.to_csv(('BALANCED_ACCURACY_' + str(iter) + '.txt'), sep = '\t')
