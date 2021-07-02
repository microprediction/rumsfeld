import pandas as pd
import numpy as np
import math
from pycaret.regression import *


N_TRAIN = 100

EXPERIMENTS = '/Users/petercotton/github/rumsfeld/experiments/'

contests = pd.read_csv(EXPERIMENTS+'2_posterior.csv')
primary = pd.read_csv(EXPERIMENTS + '1_predictions.csv')
primary = primary.merge(contests[['contest_id', 'contestant_id', 'posterior']], on=['contest_id', 'contestant_id'], how='left')
primary.rename(inplace=True, columns={'Score': 'primary_probability', 'Label': 'Primary Label'})
primary['in_sample'] = [ 1 for _ in range(N_TRAIN)] + [0 for _ in range(len(primary)-N_TRAIN)]


train = primary[:N_TRAIN]
REGRESSORS = ['WeekofPurchase', 'StoreID', 'PriceCH',
       'PriceMM', 'DiscCH', 'DiscMM', 'SpecialCH', 'SpecialMM', 'LoyalCH',
       'SalePriceMM', 'SalePriceCH', 'PriceDiff', 'Store7', 'PctDiscMM',
       'PctDiscCH', 'ListPriceDiff', 'STORE','probability']
train = train[REGRESSORS+['posterior','Purchase']]
setup(train, target = 'posterior', session_id=105, train_size=0.9, silent=True, ignore_features=['Purchase'])

blurb = """
        * 'lr' - Linear Regression                   
        * 'lasso' - Lasso Regression                
        * 'ridge' - Ridge Regression                
        * 'en' - Elastic Net                   
        * 'lar' - Least Angle Regression                  
        * 'llar' - Lasso Least Angle Regression                   
        * 'omp' - Orthogonal Matching Pursuit                     
        * 'br' - Bayesian Ridge                   
        * 'ard' - Automatic Relevance Determination                  
        * 'par' - Passive Aggressive Regressor                    
        * 'ransac' - Random Sample Consensus       
        * 'tr' - TheilSen Regressor                   
        * 'huber' - Huber Regressor                               
        * 'kr' - Kernel Ridge                                     
        * 'svm' - Support Vector Regression                           
        * 'knn' - K Neighbors Regressor                           
        * 'dt' - Decision Tree Regressor                                   
        * 'rf' - Random Forest Regressor                                   
        * 'et' - Extra Trees Regressor                            
        * 'ada' - AdaBoost Regressor                              
        * 'gbr' - Gradient Boosting Regressor                               
        * 'mlp' - MLP Regressor
        * 'xgboost' - Extreme Gradient Boosting                   
        * 'lightgbm' - Light Gradient Boosting Machine                    
        * 'catboost' - CatBoost Regressor     1
"""

CHOICES = ['lr','lasso','ridge','en','omp','br','ransac',
                     'huber','kr','rf','et','gbr','mlp','lightgbm']


report = dict()
for model_choice in CHOICES:
       reg_model = create_model(model_choice, fold = 10)
       tuned_reg_model = tune_model(reg_model, optimize='MSE')
       reg_model_final = finalize_model(tuned_reg_model)
       secondary = predict_model(reg_model_final, primary)
       secondary.rename(inplace=True, columns={'Label': 'secondary_probability'})
       mirror = primary.copy()

       secondary['primary_brier'] = [(1 - p) ** 2 if lab == pur else (0 - p) ** 2 for lab, pur, p in zip(secondary['Primary Label'], secondary['Purchase'], secondary['primary_probability'])]
       secondary['secondary_brier'] = [(1 - p) ** 2 if lab == pur else (0 - p) ** 2 for lab, pur, p in zip(secondary['Primary Label'], secondary['Purchase'], secondary['secondary_probability'])]

       summary = secondary[['primary_brier','secondary_brier','in_sample']].groupby('in_sample').mean()
       report[model_choice] = summary

for mc,summ in report.items():
       print(" ")
       print(mc)
       print(summ)


# Save the best ... you need to have the best one listed last!
secondary.to_csv(EXPERIMENTS+'2_secondary.csv')





