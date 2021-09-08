import numpy as np
import pickle
import data
import lr
import it
import config
import utils
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

def regularize(dataset_info):
    folds = data.get_kfold_data(dataset_info, config.number_of_folds)
    l2_lambda = utils.get_lambda(dataset_info)
    for term_names in config.regularization_terms:
        parameter_names = '_'.join(term_names)
        print(parameter_names)
        regularization_parameters = create_regularization_parameters(term_names, config.parameters['list'], l2_lambda)
        cross_evaluate_parameter(dataset_info['tag'], config.parameters['name'], folds, regularization_parameters, parameter_names)

def create_regularization_parameters(term_names, parameters, l2_lambda):
    regularization_parameters = dict(alpha=np.zeros(len(parameters)), beta=np.zeros(len(parameters)), gamma=np.zeros(len(parameters)), l2_lambda=np.repeat(l2_lambda, len(parameters)))
    for term_name in term_names:
        regularization_parameters[term_name] = parameters
    return regularization_parameters

def calculate_dataset_metrics(folds):
    metrics_train_folds = []
    metrics_test_folds = []
    for fold in folds:
        y_train, y_test, A_train, A_test = fold['y_train'], fold['y_test'], fold['A_train'], fold['A_test']
        metrics_train = it.calculate_relevant_metrics(A_train, list(y_train))
        metrics_train_folds.append(metrics_train)
        metrics_test = it.calculate_relevant_metrics(A_test, list(y_test))
        metrics_test_folds.append(metrics_test)
    return metrics_train_folds, metrics_test_folds

def summarize_dicts_from_folds(dicts_folds):
    summarized_dict = {}
    averaged_dict = {}
    for key in dicts_folds[0].keys():
        summarized_dict[key] = []
        for train_dict_fold in dicts_folds:
            summarized_dict[key].append(train_dict_fold[key])
        averaged_dict[key] = np.mean(summarized_dict[key])
    return summarized_dict, averaged_dict

def cross_evaluate_parameter(dataset_tag, parameter_description, folds, regularization_parameters, term_name):
    train_dicts_alpha = []
    test_dicts_alpha = []
    train_dicts_alpha_folds = []
    test_dicts_alpha_folds = []
    
    metrics_train_folds, metrics_test_folds = calculate_dataset_metrics(folds)
    print('Calculated entropies for fold')
    for alpha, beta, gamma, l2_lambda in zip(regularization_parameters['alpha'], regularization_parameters['beta'], regularization_parameters['gamma'], regularization_parameters['l2_lambda']):
        train_dicts_folds = []
        test_dicts_folds = []
        # iterate through k folds
        for f in range(len(folds)):
            print('Fold', str(f))
            fold = folds[f]
            metrics_train = metrics_train_folds[f]
            metrics_test = metrics_test_folds[f]
            model = train_lr(fold, alpha, beta, gamma, l2_lambda, metrics_train, metrics_test)
            train_dicts_folds.append(model[1])
            test_dicts_folds.append(model[2])
        summarized_train_dict, averaged_train_dict = summarize_dicts_from_folds(train_dicts_folds)
        summarized_test_dict, averaged_test_dict = summarize_dicts_from_folds(test_dicts_folds)
        train_dicts_alpha.append(averaged_train_dict)
        test_dicts_alpha.append(averaged_test_dict)
        train_dicts_alpha_folds.append({'full': summarized_train_dict, 'averaged': averaged_train_dict})
        test_dicts_alpha_folds.append({'full': summarized_test_dict, 'averaged': averaged_test_dict})
    save_evaluation(dataset_tag, parameter_description, term_name, config.parameters['list'], train_dicts_alpha_folds, test_dicts_alpha_folds)

def save_evaluation(dataset_name, parameter_description, parameter_name, parameters, train_dicts, test_dicts):
    evaluation = {
        'training': train_dicts,
        'testing': test_dicts,
        'parameters': parameters
    }
    pickle.dump(evaluation, open("evaluations/" + dataset_name + "_" + parameter_description + "_" + parameter_name + ".p", "wb"))

def train_lr(data, alpha, beta, gamma, _lambda, metrics_train, metrics_test):
    X_train, X_test, y_train, y_test, A_train, A_test = data['X_train'], data['X_test'], data['y_train'], data['y_test'], data['A_train'], data['A_test']
    print('alpha', alpha, 'beta', beta, 'gamma', gamma)
    
    model = lr.CustomLogisticRegression(
        X=X_train, Y=y_train, A=A_train, alpha=alpha, beta=beta, gamma=gamma, _lambda=_lambda
    )
    
    print('Training model...')
    model.fit()
    print('Model trained')
    
    train_dict = model_evaluation(model, X_train, y_train, A_train, metrics_train)
    test_dict = model_evaluation(model, X_test, y_test, A_test, metrics_test)
    print(test_dict)

    return model, train_dict, test_dict

def model_evaluation(model, X, y, A, model_metrics):
    y_prob = model.predict_prob(X)
    y_pred = y_prob >= 0.5
    y_pred = y_pred.astype(int)
    
    evaluation_dict = {}
    evaluation_dict['IND'] = it.independence(list(y), y_pred, A)
    evaluation_dict['-ACC'] = it.neg_accuracy(list(y), y_pred, A)
    evaluation_dict['BAL'] = it.balance(list(y), y_pred, A)
    evaluation_dict['LEG'] = model_metrics['legacy']
    evaluation_dict['SEP'] = evaluation_dict['-ACC'] + evaluation_dict['BAL'] + evaluation_dict['IND']
    evaluation_dict['SUF'] = evaluation_dict['-ACC'] + evaluation_dict['BAL'] + evaluation_dict['LEG']
    evaluation_dict['N-IND'] = evaluation_dict['IND'] / model_metrics['entropy_A']
    evaluation_dict['N-SEP'] = evaluation_dict['SEP'] / model_metrics['entropy_A_Y']
    evaluation_dict['N-SUF'] = evaluation_dict['SUF'] / it.entropy_A_R(A, y_pred)
    
    cross_entropy = log_loss(y, y_prob)
    evaluation_dict['cross_entropy'] = cross_entropy
    
    return evaluation_dict