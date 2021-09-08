import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

def get_kfold_data(dataset_info, number_of_folds=5):
    X, y = get_X_y(dataset_info)

    # Split data into training/holdout sets
    kf = KFold(n_splits=number_of_folds, shuffle=True, random_state=0)
    kf.get_n_splits(X)
    
    # Keep track of the data for the folds
    folds = []

    # Iterate over folds, using k-1 folds for training
    # and the k-th fold for validation
    for train_index, test_index in kf.split(X):
        # Training data
        X_train = X.iloc[train_index]
        y_train = y[train_index]
        
        # Holdout data
        X_test = X.iloc[test_index]
        y_test = y[test_index]
        
        scale_data(X_train, X_test, dataset_info['numerical_attributes'])
        
        A_train = list(X_train[dataset_info['sensitive_attribute']])
        A_test = list(X_test[dataset_info['sensitive_attribute']])

        fold = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'A_train': A_train,
            'A_test': A_test
        }

        folds.append(fold)
        
    return folds

def get_X_y(dataset_info):
    filename, target = dataset_info['filename'], dataset_info['target']
    df = pd.read_csv('data/' + filename)
    y = df[target]
    y = y.replace(to_replace=2, value=0, inplace=False)
    X = df.drop([target], axis=1)
    return X, y

def scale_data(X_train, X_test, numerical):
    scaler = StandardScaler()
    
    scaler.fit(X_train[numerical])
    
    X_train[numerical] = scaler.transform(X_train[numerical])
    X_test[numerical] = scaler.transform(X_test[numerical])