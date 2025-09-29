# load some packages
import subprocess
import sys

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
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from collections import Counter

# parse arguments
def make_arg_parser():
    parser = ap.ArgumentParser(description = ".")

    parser.add_argument('--input', required = True, help = 'input filename')
    
    parser.add_argument('--iter', required = True, help = 'iteration number')
    
    return parser

args = make_arg_parser().parse_args()

# parse arguments
input_filename = args.input
iter = int(args.iter)

# read in input file
hf_balanced = pd.read_csv(input_filename)

# create column list for regressions
col_list = ['TRIG_INV_NORMAL_SCALE',
            'LDL_INV_NORMAL_SCALE',
            'HDL_INV_NORMAL_SCALE',
            'GLUCOSE_INV_NORMAL_SCALE',
            'HbA1c_INV_NORMAL_SCALE',
            'SBP_INV_NORMAL_SCALE',
            'DBP_INV_NORMAL_SCALE',
            'T2D',
            'BMI_INV_NORMAL_SCALE',
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

# create iteration column name
print(iter)
colname = 'ITER_' + str(iter)

# set significance threshold
corrected_sig = 0.05

# create df lists
#pval_dfs = []
#beta_dfs = []
#insig_dfs = []
#sig_dfs = []
#coef_dfs = []
#important_dfs = []
#unimportant_dfs = []

# loop through iterations
#for iter in list(range(1, 1001)):
# split dataset into training and testing
hf_train = hf_balanced.sample(frac = 0.7, random_state = iter)
hf_reg = hf_train.sample(frac = 0.5, random_state = iter)
hf_lasso = hf_train.drop(hf_reg.index)
hf_no_train = hf_balanced.drop(hf_train.index)

# apply SMOTE
x_reg = hf_reg.drop(columns = ['HF'])
y_reg = hf_reg[['HF']]
x_reg_resampled, y_reg_resampled = SMOTE(random_state = iter).fit_resample(x_reg, y_reg)
hf_reg = pd.concat([x_reg_resampled, y_reg_resampled], axis = 1)

x_lasso = hf_lasso.drop(columns = ['HF'])
y_lasso = hf_lasso[['HF']]
x_lasso_resampled, y_lasso_resampled = SMOTE(random_state = iter).fit_resample(x_lasso, y_lasso)
hf_lasso = pd.concat([x_lasso_resampled, y_lasso_resampled], axis = 1)

# initial regressions
train_metric_list = []
for col in col_list:
    model_df = hf_reg.dropna(subset = ['HF', col, 'AGE', 'SEX'])
    
    model = sm.Logit(model_df['HF'], model_df[[col, 'AGE', 'SEX']]).fit()
    pval = f"{model.pvalues.iloc[0]:.2e}"
    beta = model.params.iloc[0]
    df = pd.DataFrame(data = {'Feature' : [col], 'BETA' : beta, 'PVAL' : pval})
    train_metric_list.append(df)

train_metric_df = pd.concat(train_metric_list, axis = 0)

# clean df
train_metric_df['PVAL'] = train_metric_df['PVAL'].astype(float)
train_metric_df.set_index('Feature', inplace = True)

# filter to betas
train_metric_df_beta = train_metric_df.drop(columns = ['PVAL'])
train_metric_df_beta = train_metric_df_beta.rename(columns = {'BETA' : colname})
#beta_dfs.append(train_metric_df_beta)

# filter to pvals
train_metric_df_pval = train_metric_df.drop(columns = ['BETA'])
train_metric_df_pval = train_metric_df_pval.rename(columns = {'PVAL' : colname})
#pval_dfs.append(train_metric_df_pval)

# filter to insignificant regressions
train_metric_df_insig = train_metric_df[train_metric_df['PVAL'] > corrected_sig]
train_metric_df_insig = train_metric_df_insig.drop(columns = ['BETA'])
train_metric_df_insig = train_metric_df_insig.rename(columns = {'PVAL' : colname})
#insig_dfs.append(train_metric_df_insig)

# filter to significant regressions
train_metric_df_sig = train_metric_df[train_metric_df['PVAL'] <= corrected_sig]
train_metric_df_sig = train_metric_df_sig.drop(columns = ['BETA'])
train_metric_df_sig = train_metric_df_sig.rename(columns = {'PVAL' : colname})
#sig_dfs.append(train_metric_df_sig)

# lasso regressions
hf_lasso_x  = hf_lasso[['AGE', 'SEX'] + col_list]
hf_lasso_y = hf_lasso[['HF']]

# Fit Lasso Logistic Regression
lasso_logit = LogisticRegressionCV(
penalty = 'l1',
solver = 'liblinear',
Cs = 10,
cv = 5)
lasso_logit.fit(hf_lasso_x, hf_lasso_y)

# clean results
coefs = lasso_logit.coef_.flatten()
feature_names = hf_lasso_x.columns
coef_df = pd.DataFrame({'Feature' : feature_names,
                        'Coefficient' : coefs})
coef_df.set_index('Feature', inplace = True)

# filter to coefficients
coef_df_export = coef_df.rename(columns = {'Coefficient' : colname})
#coef_dfs.append(coef_df_export)

# filter to important features
important_df = coef_df[coef_df['Coefficient'] != 0].sort_values(by = 'Coefficient', key = abs, ascending = False)
important_df = important_df.rename(columns = {'Coefficient' : colname})
#important_dfs.append(important_df)

# filter to unimportant features
unimportant_df = coef_df[coef_df['Coefficient'] == 0]
unimportant_df = unimportant_df.rename(columns = {'Coefficient' : colname})
#unimportant_dfs.append(unimportant_df)

# concatenate dfs
#pval_df_cat = pd.concat(pval_dfs, axis = 1)
#beta_df_cat = pd.concat(beta_dfs, axis = 1)
#insig_df_cat = pd.concat(insig_dfs, axis = 1)
#sig_df_cat = pd.concat(sig_dfs, axis = 1)
#coef_df_cat = pd.concat(coef_dfs, axis = 1)
#important_df_cat = pd.concat(important_dfs, axis = 1)
#unimportant_df_cat = pd.concat(unimportant_dfs, axis = 1)

# export dfs
train_metric_df_pval.to_csv(('LR_pval_' + str(iter) + '.txt'), sep = '\t')
train_metric_df_beta.to_csv(('LR_beta_' + str(iter) + '.txt'), sep = '\t')
train_metric_df_insig.to_csv(('LR_insignificant_' + str(iter) + '.txt'), sep = '\t')
train_metric_df_sig.to_csv(('LR_significant_' + str(iter) + '.txt'), sep = '\t')
coef_df_export.to_csv(('LASSO_coef_' + str(iter) + '.txt'), sep = '\t')
important_df.to_csv(('LASSO_important_' + str(iter) + '.txt'), sep = '\t')
unimportant_df.to_csv(('LASSO_unimportant_' + str(iter) + '.txt'), sep = '\t')
