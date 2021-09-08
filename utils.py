import pickle

def save_lambda(dataset_info, l2_lambda):
    outfile = open('evaluations/l2_lambda_' + dataset_info['tag'], 'wb')
    pickle.dump(l2_lambda, outfile)
    outfile.close()

def get_lambda(dataset_info):
    infile = open('evaluations/l2_lambda_' + dataset_info['tag'], 'rb')
    l2_lambda = pickle.load(infile)
    infile.close()
    return l2_lambda