import numpy as np
import glob 
import time
from block_functions import findQi as fqi

#This script is used for the feature exstraction phase of every image in the dataset
#Since this phase is the longest one it's better to do it just one time and than store the results 

#constant declaration
W = 16

#Base_Path = "./../../dataset/complete_dataset"
Base_Path = "./complete_dataset"


def main():
    start_time = time.time()
    single_captured_path_list = glob.glob(Base_Path + "/single_captured/SingleCaptureImages/V610/*.JPG")
    print("single captured found: ", len(single_captured_path_list))
    recaptured_path_list = glob.glob(Base_Path + "/recaptured/RecapturedImages/*.png")
    print("re-captured found: ", len(recaptured_path_list))
    i = 0
    for x in single_captured_path_list[0 : ]:
        time1 = time.time()
        Qi = fqi.findQi(x)
        if i == 0:
            print("Qi shape 0:", Qi.shape[0])
            print("Qi shape 1:", Qi.shape[1])
        np.savetxt(x + ".txt", Qi)
        time2 = time.time()
        i=i+1
        print("Tempo impiegato per calcolo single captured Qi di ", i ," --- %s seconds ---" % (time2 - time1))
    time3 = time.time()
    


    i = 0
    for x in recaptured_path_list[0 : ]:
        time1 = time.time()
        Qi = fqi.findQi(x)
        if i == 0:
            print("Qi shape 0:", Qi.shape[0])
            print("Qi shape 1:", Qi.shape[1])
        np.savetxt(x + ".txt", Qi)
        time2 = time.time()
        i=i+1
        print("Tempo impiegato per calcolo recaptured Qi di ", i ," --- %s seconds ---" % (time2 - time1))
    time4 = time.time()
    print("Tempo impiegato per calcolo di tutte le single captured Qi --- %s seconds ---" % (time3 - start_time))
    print("Tempo impiegato per calcolo di tutte le recaptured Qi --- %s seconds ---" % (time4 - time3))
    return 1

if __name__ == '__main__':
	output = main()