import numpy as np

tags = {
    'IND': {'name': 'Independence', 'parameter': 'alpha'},
    'SEP': {'name': 'Separation', 'parameter': 'alpha_beta_gamma'},
    'SUF': {'name': 'Sufficiency', 'parameter': 'beta_gamma'},
    'BAL': {'name': 'Balance', 'parameter': 'beta'},
    '-ACC': {'name': 'Negative accuracy', 'parameter': 'gamma'},
    'N-IND': {'name': 'Normalized independence'},
    'N-SEP': {'name': 'Normalized separation'},
    'N-SUF': {'name': 'Normalized sufficiency'}
}

# Colors should at least have a contrast ratio of 4.5:1 --> https://webaim.org/resources/contrastchecker/
style = {
    'IND': {
        'color': '#397A89'
    },
    'SEP': {
        'color': '#601A4A'
    },
    'SUF': {
        'color': '#B3220F'
    },
    'N-IND': {
        'color': '#397A89'
    },
    'N-SEP': {
        'color': '#601A4A'
    },
    'N-SUF': {
        'color': '#B3220F'
    },
    '-ACC': {
        'color': '#382119'
    },
    'cross_entropy': {
        'color': '#382119'
    },
    'BAL': {
        'color': '#9C6507'
    },
    'LEG': {
        'color': '#4B7C5B'
    },
    'german': {
        'linestyle': 'solid',
        'marker': 'o'
    },
    'compas': {
        'linestyle': (0, (5, 10)),
        'marker': '^'
    },
    'adult': {
        'linestyle': 'dotted',
        'marker': 'x'
    }
}


# Datasets
compas = {
    'tag': 'compas',
    'name': 'ProPublica recidivism',
    'filename': 'propublica-recidivism_numerical-binsensitive_slim.csv',
    'sensitive_attribute': 'race',
    'target': 'two_year_recid',
    'numerical_attributes': ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
}
german = {
    'tag': 'german',
    'name': 'German credit',
    'filename': 'german_numerical-binsensitive.csv',
    'sensitive_attribute': 'sex',
    'target': 'credit',
    'numerical_attributes': ['month', 'credit_amount', 'investment_as_income_percentage', 'residence_since', 'number_of_credits', 'people_liable_for']
}

adult = {
    'tag': 'adult',
    'name': 'Adult income',
    'filename': 'adult_numerical-binsensitive.csv',
    'sensitive_attribute': 'race', # could also be sex
    'target': 'income-per-year',
    'numerical_attributes': ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
}

datasets = [compas, german, adult]

method = 'L-BFGS-B'
parameters = {
    'list': list(np.arange(0,1,0.2)) + list(np.arange(1,10,2)) + list(np.arange(10,40,5)) + list(np.arange(40,201,30)),
    'name': 'complete'
}
number_of_folds = 5
regularization_terms = [['alpha'], ['beta'], ['gamma'], ['alpha', 'beta'], ['alpha', 'gamma'], ['beta', 'gamma'], ['alpha', 'beta', 'gamma']]

# lambdas over which to search for the optimal lambda (for the L2 regularization)
lambdas = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0]
number_of_lambda_folds = 5