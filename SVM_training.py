import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from block_functions import approximation_error as ape
from block_functions import lambda_calculus as lc
from joblib import dump, load

#constant declaration
LABEL_SC = 0
LABEL_RC = 1

#Base_Path = "./../../dataset/complete_dataset"
Base_Path = "./complete_dataset"

def initialize():
    data = {
        'img_path' : [],
        'label' : [],
        'lambda_avg' : [],
        'approx_error' : []
    } 
    return data

def extracting_feature(single_captured_train, recaptured_train):
    data = initialize()
    Dsc_T = np.loadtxt(Base_Path +"/dictionaries/Dsc_T.txt")
    Drc_T = np.loadtxt(Base_Path +"/dictionaries/Drc_T.txt")

    #orthogonal matching pursuit
    for x in single_captured_train:
        Qsc = np.loadtxt(x)
        if np.any(Qsc): 
            Esc_on_sc = ape.approximation_error(Qsc, Dsc_T)
            Esc_on_rc = ape.approximation_error(Qsc, Drc_T)
            Ed_sc = Esc_on_sc - Esc_on_rc
            lambda_avg_sc = lc.find_lambda_average(Qsc)
            data["img_path"].append(x)
            data['label'].append(LABEL_SC)
            data['approx_error'].append(Ed_sc[0,0])
            data['lambda_avg'].append(lambda_avg_sc)
        else:
            print("No Qsc found for: ", x)

    for x in recaptured_train:
        Qrc = np.loadtxt(x)
        if np.any(Qrc): 
            Erc_on_sc = ape.approximation_error(Qrc, Dsc_T)
            Erc_on_rc = ape.approximation_error(Qrc, Drc_T)
            Ed_rc = Erc_on_sc - Erc_on_rc
            lambda_avg_rc = lc.find_lambda_average(Qrc)
            data["img_path"].append(x)
            data['label'].append(LABEL_RC)
            data['approx_error'].append(Ed_rc[0,0])
            data['lambda_avg'].append(lambda_avg_rc)
        else:
            print("No Qrc found for: ", x)
    return data

def main():
    start_time = time.time()
    with open(Base_Path + "/training_set_paths/single_captured_train.txt", 'r') as f:
        single_captured_train = [line.rstrip('\n') for line in f]
    with open(Base_Path + "/training_set_paths/recaptured_train.txt", 'r') as f:
        recaptured_train = [line.rstrip('\n') for line in f]
    data = extracting_feature(single_captured_train, recaptured_train)
    dataset = pd.DataFrame.from_dict(data, orient = 'index').transpose()
    time1 = time.time()
    print("Tempo impiegato per preparazione features di training: --- %s seconds ---" % (time1 - start_time)) 
    dataset.to_csv(Base_Path + "/training_set_paths/table.csv")
    dataset = pd.read_csv(Base_Path + "/training_set_paths/table.csv")
    X = dataset[['lambda_avg','approx_error']].values
    Y = dataset[['label']].values
    
    svm = SVC(kernel='linear')
    model = svm.fit(X, Y.ravel())
    dump(model, Base_Path + "/svm/model.joblib")
    w = svm.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 20)
    yy = a * xx - (svm.intercept_[0]) / w[1]
    time2 = time.time()
    print("Tempo impiegato per training SVM: --- %s seconds ---" % (time2 - time1))
    plt.scatter(dataset[['lambda_avg']].values, dataset[['approx_error']].values, c = dataset[['label']].values, marker='+')
    plt.plot(xx, yy)
    plt.show()
    print("fine")
    return 1


if __name__ == '__main__':
	output = main()