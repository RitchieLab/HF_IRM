# load packages
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

    parser.add_argument('--iter', required = True, help = 'random seed for iteration')
    
    parser.add_argument('--input', required = True, help = 'input filename')
    
    parser.add_argument('--output_dir', required = True, help = 'output filename')
    
    return parser

args = make_arg_parser().parse_args()

# parse arguments
iter = int(args.iter)
input_filename = args.input
output_dir = args.output_dir

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
            'BMI_INV_NORMAL_SCALE',
            'PA_EVERYDAY_SCALE',
            'NEIGHBORHOOD_TRUST_SCALE',
            'NEIGHBORHOOD_HOUSING_SCALE',
            'NEIGHBORHOOD_GET_ALONG_SCALE',
            'INCOME_SCALE',
            'CENSUS_MEDIAN_INCOME_INV_NORMAL_SCALE',
            'SOCIAL_DEPRIVATION_INDEX_INV_NORMAL_SCALE',
            'EDUCATION_HIGHEST_SCALE',
            'NEIGHBORHOOD_DRUG_USE_SCALE',
            'NEIGHBORHOOD_SAFE_CRIME_SCALE',
            'NEIGHBORHOOD_BUILDING_SCALE',
            'NEIGHBORHOOD_ALCOHOL_SCALE',
            'NEIGHBORHOOD_VANDALISM_SCALE',
            'NEIGHBORHOOD_SIDEWALK_SCALE',
            'NEIGHBORHOOD_BIKE_SCALE',
            'NEIGHBORHOOD_CLEAN_SCALE',
            'NEIGHBORHOOD_WATCH_SCALE',
            'NEIGHBORHOOD_UNSAFE_WALK_SCALE',
            'NEIGHBORHOOD_CARE_SCALE',
            'NEIGHBORHOOD_ALOT_CRIME_SCALE',
            'NEIGHBORHOOD_CRIME_WALK_SCALE',
            'NEIGHBORHOOD_SAME_VALUES_SCALE',
            'NEIGHBORHOOD_GRAFFITI_SCALE',
            'NEIGHBORHOOD_NOISE_SCALE',
            'NEIGHBORHOOD_FREE_AMENITIES_SCALE',
            'NEIGHBORHOOD_PPL_HANGING_AROUND_SCALE',
            'NEIGHBORHOOD_TROUBLE_SCALE',
            'NEIGHBORHOOD_STORES_SCALE',
            'NEIGHBORHOOD_TRANSIT_SCALE',
            'SMOKING',
            'T2D']

# create iteration column name
print(iter)
colname = 'ITER_' + str(iter)

# set significance threshold
corrected_sig = 0.05

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
    df = pd.DataFrame(data = {'COLUMN' : [col], 'BETA' : beta, 'PVAL' : pval})
    train_metric_list.append(df)

train_metric_df = pd.concat(train_metric_list, axis = 0)

# clean df
train_metric_df['PVAL'] = train_metric_df['PVAL'].astype(float)
train_metric_df.set_index('COLUMN', inplace = True)

# filter to betas
train_metric_df_beta = train_metric_df.drop(columns = ['PVAL'])
train_metric_df_beta = train_metric_df_beta.rename(columns = {'BETA' : colname})

# filter to pvals
train_metric_df_pval = train_metric_df.drop(columns = ['BETA'])
train_metric_df_pval = train_metric_df_pval.rename(columns = {'PVAL' : colname})

# filter to insignificant regressions
train_metric_df_insig = train_metric_df[train_metric_df['PVAL'] > corrected_sig]
train_metric_df_insig = train_metric_df_insig.drop(columns = ['BETA'])
train_metric_df_insig = train_metric_df_insig.rename(columns = {'PVAL' : colname})

# filter to significant regressions
train_metric_df_sig = train_metric_df[train_metric_df['PVAL'] <= corrected_sig]
train_metric_df_sig = train_metric_df_sig.drop(columns = ['BETA'])
train_metric_df_sig = train_metric_df_sig.rename(columns = {'PVAL' : colname})

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

# filter to important features
important_df = coef_df[coef_df['Coefficient'] != 0].sort_values(by = 'Coefficient', key = abs, ascending = False)
important_df = important_df.rename(columns = {'Coefficient' : colname})

# filter to unimportant features
unimportant_df = coef_df[coef_df['Coefficient'] == 0]
unimportant_df = unimportant_df.rename(columns = {'Coefficient' : colname})

# export dfs
train_metric_df_pval.to_csv((output_dir + 'LR_pval_' + colname + '.txt'), sep = '\t')
train_metric_df_beta.to_csv((output_dir + 'LR_beta_' + colname + '.txt'), sep = '\t')
train_metric_df_insig.to_csv((output_dir + 'LR_insignificant_' + colname + '.txt'), sep = '\t')
train_metric_df_sig.to_csv((output_dir + 'LR_significant_' + colname + '.txt'), sep = '\t')
coef_df_export.to_csv((output_dir + 'LASSO_coef_' + colname + '.txt'), sep = '\t')
important_df.to_csv((output_dir + 'LASSO_important_' + colname + '.txt'), sep = '\t')
unimportant_df.to_csv((output_dir + 'LASSO_unimportant_' + colname + '.txt'), sep = '\t')
