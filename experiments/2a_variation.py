import pandas as pd
import numpy as np

EXPERIMENTS = '/Users/petercotton/github/rumsfeld/experiments/'
classifier_names = ['lr','lightgbm','lda','gbc']

contest0 = None
for k, classifier_name in enumerate(classifier_names):
    contestk = pd.read_csv(EXPERIMENTS +'1_contests_' + classifier_name + '.csv')
    contestk.rename(inplace=True, columns={'ability': 'ability_'+classifier_name})
    if contest0 is None:
        contest0 = contestk.copy()
    else:
        ck = contestk[['contest_id', 'contestant_id', 'ability_'+classifier_name]]
        contest0 = contest0.merge(ck, on=['contest_id','contestant_id'])

cols = ['ability_'+cn for cn in classifier_names]
variation = np.nanmean( np.nanstd( contest0[cols].values, axis=1 ) )
print(variation)
import math
print(math.sqrt(1-variation*variation))

contest0.to_csv(EXPERIMENTS+'1_contests_variation.csv')






