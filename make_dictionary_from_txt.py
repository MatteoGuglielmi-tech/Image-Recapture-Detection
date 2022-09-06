import numpy as np
import time
from sklearn.decomposition import DictionaryLearning
from block_functions import selection_blocks as sb

#constant declaration
L = 3 #number of non zero coef.
K = 50 #number of atoms
Alpha = 1
Max_Iter = 50 #max iteration for dictionary fitting
Tollerance = 1e-6 #tollerance for the omp algorithm

#Base_Path = "./../../dataset/complete_dataset"
Base_Path = "./complete_dataset"

W = 16

def main():
    #acquiring training set paths
    start_time = time.time()
    with open(Base_Path + "/training_set_paths/single_captured_train.txt", 'r') as f:
        single_captured_train = [line.rstrip('\n') for line in f]
    #find the training feature matrix for single captured img
    Ssc = np.array([])
    flag = True
    i = 0
    while flag:
        Qi = np.loadtxt(single_captured_train[i])
        if np.any(Qi):
            Ssc = Qi
            flag = False
        else:
            print("No Qi found for: ", single_captured_train[i])
        i += 1
    for x in single_captured_train[i :]:
        Qi = np.loadtxt(x)
        if np.any(Qi):
            Ssc = np.c_[Ssc, Qi]
        else:
            print("No Qi found for: ", single_captured_train[i])
    Ssc_red = sb.reduce_S(Ssc, 4) 
    Ssc_red_T = np.transpose(Ssc_red)
    time2 = time.time()
    print("Ssc shape 0:", Ssc.shape[0])
    print("Ssc shape 1:", Ssc.shape[1])
    print("Ssc_red shape 0:", Ssc_red.shape[0])
    print("Ssc_red shape 1:", Ssc_red.shape[1])
    print("Ssc_red_T shape 0:", Ssc_red_T.shape[0])
    print("Ssc_red_T shape 1:", Ssc_red_T.shape[1])
    print("Tempo impiegato per assemblaggio Ssc, Ssc_red ed Ssc_red_T--- %s seconds ---" % (time2 - start_time))

    #find the training feature matrix for  re-captured img
    time3 = time.time()
    with open(Base_Path + "/training_set_paths/recaptured_train.txt", 'r') as f:
        recaptured_train = [line.rstrip('\n') for line in f]
    #find the training feature matrix for single captured img
    Src = np.array([])
    flag = True
    i = 0
    while flag:
        Qi = np.loadtxt(recaptured_train[i])
        if np.any(Qi):
            Src = Qi
            flag = False
        else:
            print("No Qi found for: ", recaptured_train[i])
        i += 1
    for x in recaptured_train[i :]:
        Qi = np.loadtxt(x)
        if np.any(Qi):
            Src = np.c_[Src, Qi]
        else:
            print("No Qi found for: ", recaptured_train[i])
    Src_red = sb.reduce_S(Src, 4) 
    Src_red_T = np.transpose(Src_red)
    time4 = time.time()
    print("Src shape 0:", Src.shape[0])
    print("Src shape 1:", Src.shape[1])
    print("Src_red shape 0:", Src_red.shape[0])
    print("Src_red shape 1:", Src_red.shape[1])
    print("Src_red_T shape 0:", Src_red_T.shape[0])
    print("Src_red_T shape 1:", Src_red_T.shape[1])
    print("Tempo impiegato per assemblaggio Src, Src_red ed Src_red_T--- %s seconds ---" % (time3 - time3))
    
    #calculate dictionary for single capturedDsc
    dict_learner_sc = DictionaryLearning(n_components = K, alpha = 1, max_iter = Max_Iter, tol = Tollerance, transform_algorithm = 'omp', transform_n_nonzero_coefs = L)
    dict_learner_sc.fit(Ssc_red_T)
    Dsc = dict_learner_sc.components_
    Dsc_T = np.transpose(Dsc)
    time5 = time.time()
    print("Dsc_T shape 0:", Dsc_T.shape[0])
    print("Dsc_T shape 1:", Dsc_T.shape[1])
    print("Tempo impiegato per calcolo dizionario Dsc--- %s seconds ---" % (time5 - time4))

    #calculate dictionary for recapturedDsc
    dict_learner_rc = DictionaryLearning(n_components = K, alpha = 1, max_iter = Max_Iter, tol = Tollerance, transform_algorithm = 'omp', transform_n_nonzero_coefs = L)
    dict_learner_rc.fit(Src_red_T)
    Drc = dict_learner_rc.components_
    Drc_T = np.transpose(Drc)
    time6 = time.time()
    print("Drc_T shape 0:", Drc_T.shape[0])
    print("Drc_T shape 1:", Drc_T.shape[1])
    print("Tempo impiegato per calcolo dizionario Drc--- %s seconds ---" % (time6 - time5))

    #save dictionary
    np.savetxt(Base_Path + "/dictionaries/Dsc_T.txt", Dsc_T)
    np.savetxt(Base_Path + "/dictionaries/Drc_T.txt", Drc_T)
    
    return 1

if __name__ == '__main__':
	output = main()