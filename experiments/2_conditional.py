import pandas as pd
import numpy as np
import math

EXPERIMENTS = '/Users/petercotton/github/rumsfeld/experiments/'

contests = pd.read_csv(EXPERIMENTS+'1_contests.csv')


def add_posterior(df_contest):
    first_guy_won = df_contest['won'].values[0]
    abilities = df_contest['ability'].values
    a0, a1 = abilities[0], abilities[1]
    n = 5000
    rho = 0.975  # <-- uncertainty in ability. Close to 1 means close to empirical. Close to 0 we try to mimic primary model.
    tau = math.sqrt(1-rho*rho)   # <--- uncertainty in performance
    ra0 = [ a0 + rho*np.random.randn() for _ in range(n) ] # Ability
    ra1 = [ a1 + rho*np.random.randn() for _ in range(n) ]
    p0s = [ (a+tau*np.random.randn(), a+tau*np.random.randn() ) for a in ra0 ]  # Performances of first guy
    p1s = [ (a+tau*np.random.randn(), a+tau*np.random.randn() ) for a in ra1 ]  # Performances of second guy
    prior = np.mean( [ p01<p11 for (p00,p01),(p10,p11) in zip(p0s,p1s) ]  )
    if first_guy_won:
        posterior = np.mean( [ p01<p11 for (p00,p01),(p10,p11) in zip(p0s,p1s) if p00<p10] )
    else:
        posterior = np.mean( [ p01<p11 for (p00,p01),(p10,p11) in zip(p0s,p1s) if p00>p10] )
    df_contest['posterior'] = [ posterior, 1-posterior ]
    df_contest['prior'] = [ prior, 1-prior ]

    return df_contest

contests = contests.groupby('contest_id').apply(add_posterior)

contests.to_csv(EXPERIMENTS+'2_posterior.csv')

