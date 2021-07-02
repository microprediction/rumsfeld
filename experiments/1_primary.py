from pycaret.datasets import get_data
from pycaret.classification import *
from pprint import pprint

N_TRAIN = 800
EXPERIMENTS = '/Users/petercotton/github/rumsfeld/experiments/'
classifier_names = ['lr','lightgbm','lda','gbc','lightgbm']
for classifier_name in classifier_names:

    data = get_data('juice')
    small_data = data
    training_data = small_data[:N_TRAIN]

    clf1 = setup(training_data, target = 'Purchase', session_id=100, silent=True )
    classifier_model = create_model(classifier_name)
    classifier_tuned = tune_model(classifier_model)
    predictions = predict_model(classifier_tuned, data=data )


    # Massage
    predictions['contest_id'] = list(range(len(predictions)))
    predictions['contestant_id'] = predictions['Label'].apply( lambda s: 1 if s=='CH' else 0 )
    predictions['won'] = (predictions['Purchase']==predictions['Label']).apply( lambda b: 1 if b else 0)
    predictions['probability'] = [ score if label=='CH' else 1-score for score, label in zip(predictions['Score'],predictions['Label'])]
    duplicate_train = predictions.copy()
    duplicate_train['Label'] = duplicate_train['Label'].apply( lambda s: 'CH' if s=='MM' else 'CH' )
    duplicate_train['probability'] = 1-duplicate_train['probability']
    duplicate_train['contestant_id'] = 1-duplicate_train['contestant_id']
    duplicate_train['won'] = 1-duplicate_train['won']
    contests = pd.concat([predictions, duplicate_train],ignore_index=True).sort_values('contest_id')
    contests['market_dividend'] = 1/contests['probability']

    # Sample
    from winning.std_calibration import std_dividend_implied_ability

    def center(xs):
        x_mean = np.nanmean(xs)
        return [xi-x_mean for xi in xs]

    def add_ability(df_contest):
        dividends = df_contest['market_dividend'].values
        abilities = center( std_dividend_implied_ability(dividends) )
        df_contest['ability'] = abilities
        return df_contest

    print('')
    print('Computing abilities ..')
    contests = contests.groupby('contest_id').apply(add_ability)
    print('Saving...'+classifier_name)
    contests.to_csv(EXPERIMENTS + '1_contests_'+classifier_name+'.csv')
    predictions.to_csv(EXPERIMENTS + '1_predictions_'+classifier_name+'.csv')



