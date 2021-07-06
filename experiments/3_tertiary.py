from pycaret.regression import *
from pprint import pprint

N_TRAIN = 800

EXPERIMENTS = '/Users/petercotton/github/rumsfeld/experiments/'

tertiary = pd.read_csv(EXPERIMENTS + '2_secondary.csv')
tertiary['delta'] = tertiary['secondary_probability']-tertiary['primary_probability']

train = tertiary[:N_TRAIN]

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)

REGRESSORS = ['WeekofPurchase', 'StoreID', 'PriceCH',
       'PriceMM', 'DiscCH', 'DiscMM', 'SpecialCH', 'SpecialMM', 'LoyalCH',
       'SalePriceMM', 'SalePriceCH', 'PriceDiff', 'Store7', 'PctDiscMM',
       'PctDiscCH', 'ListPriceDiff', 'STORE']
train = train[REGRESSORS+['delta','primary_brier','secondary_brier','Purchase']]
setup(train, target = 'delta', session_id=109, train_size=0.9, silent=True,ignore_features=['primary_brier','secondary_brier','Purchase'])

CHOICES = ['lr','lasso','ridge','en','omp','br','ransac',
                     'huber','kr','rf','et','gbr','mlp','lightgbm']

tertiary_copy = tertiary.copy()

report = dict()
for model_choice in CHOICES:
       print('---------')
       print(model_choice)
       reg_model = create_model(model_choice, fold = 10, verbose=False)
       tuned_reg_model = tune_model(reg_model, tuner_verbose=False)
       reg_model_final = finalize_model(tuned_reg_model)
       tertiary_out = predict_model(reg_model_final,tertiary_copy)
       tertiary_out.rename(inplace=True, columns={'Label': 'delta_hat'})


       gain = 0.3
       thresholds = [0.005,0.01,0.03,0.05,0.08,0.13]
       for threshold in thresholds:
              tertiary_out['hypocratic_'+str(threshold)] = [ prim+gain*delta_hat if abs(delta_hat)>threshold else prim for prim, delta_hat in zip( tertiary_out['primary_probability'],tertiary_out['delta_hat']) ]
              tertiary_out['brier_'+str(threshold)]  = [(1 - p) ** 2 if lab == pur else (0 - p) ** 2 for lab, pur, p in zip(tertiary_out['Primary Label'], tertiary_out['Purchase'], tertiary_out['hypocratic_'+str(threshold)])]

       brier_cols = [ 'brier_'+str(threshold) for threshold in thresholds ]
       summary = tertiary_out[['primary_brier', 'secondary_brier', 'in_sample']+brier_cols].groupby('in_sample').mean().reset_index()
       report[model_choice] = summary

for mc,summ in report.items():
       print(" ")
       print(mc)
       print(summ.transpose())


# Save the best ... you need to have the best one listed last!






