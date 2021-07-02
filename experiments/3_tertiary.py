import pandas as pd
import numpy as np
import math
from pycaret.classification import *
from pprint import pprint


N_TRAIN = 800

EXPERIMENTS = '/Users/petercotton/github/rumsfeld/experiments/'

tertiary = pd.read_csv(EXPERIMENTS + '2_secondary.csv')

train = tertiary[:N_TRAIN]
REGRESSORS = ['WeekofPurchase', 'StoreID', 'PriceCH',
       'PriceMM', 'DiscCH', 'DiscMM', 'SpecialCH', 'SpecialMM', 'LoyalCH',
       'SalePriceMM', 'SalePriceCH', 'PriceDiff', 'Store7', 'PctDiscMM',
       'PctDiscCH', 'ListPriceDiff', 'STORE','secondary_probability','primary_probability']
train = train[REGRESSORS+['Purchase','primary_brier','secondary_brier']]
setup(train, target = 'Purchase', session_id=105, train_size=0.9, silent=True,ignore_features=['primary_brier','secondary_brier'])

CHOICES = ['lr','knn','nb','dt','rbfsvm','gpc','mlp','rf',
       'qda','ada','gbc','lda','et','et','lightgbm']

CHOICES = ['lr','mlp','rf',
       'qda','lightgbm']

tertiary_copy = tertiary.copy()

report = dict()
for model_choice in CHOICES:
       print('---------')
       print(model_choice)
       reg_model = create_model(model_choice, fold = 10, verbose=False)
       tuned_reg_model = tune_model(reg_model, tuner_verbose=False)
       reg_model_final = finalize_model(tuned_reg_model)
       tertiary_out = predict_model(reg_model_final,tertiary_copy)
       if 'Score' not in tertiary_out.columns:
              raise Exception("Cannot use "+model_choice)
       tertiary_out.rename(inplace=True, columns={'Score': 'tertiary_probability','Label':'Tertiary Label'})

       tertiary_out['tertiary_brier']  = [(1 - p) ** 2 if lab == pur else (0 - p) ** 2 for lab, pur, p in zip(tertiary_out['Tertiary Label'], tertiary_out['Purchase'], tertiary_out['tertiary_probability'])]

       summary = tertiary_out[['primary_brier', 'secondary_brier', 'tertiary_brier','in_sample']].groupby('in_sample').mean()
       report[model_choice] = summary
       pprint(report)

for mc,summ in report.items():
       print(" ")
       print(mc)
       print(summ)


# Save the best ... you need to have the best one listed last!






