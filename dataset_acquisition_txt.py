import glob 
from sklearn.model_selection import train_test_split

#Constant declaration
#Base_Path = "./../../dataset/complete_dataset"
Base_Path = "./complete_dataset"


def main():
    #aquiring single captures paths for every camera
    single_captured_path_list_D40 = glob.glob(Base_Path + "/single_captured/SingleCaptureImages/D40/*.txt")
    single_captured_path_list_D70S = glob.glob(Base_Path + "/single_captured/SingleCaptureImages/D70S/*.txt")
    single_captured_path_list_EOS600D = glob.glob(Base_Path + "/single_captured/SingleCaptureImages/EOS600D/*.txt")
    single_captured_path_list_EPM2 = glob.glob(Base_Path + "/single_captured/SingleCaptureImages/EPM2/*.txt")
    single_captured_path_list_RX100 = glob.glob(Base_Path + "/single_captured/SingleCaptureImages/RX100/*.txt")
    single_captured_path_list_TZ7 = glob.glob(Base_Path + "/single_captured/SingleCaptureImages/TZ7/*.txt")
    single_captured_path_list_V550B = glob.glob(Base_Path + "/single_captured/SingleCaptureImages/V550B/*.txt")
    single_captured_path_list_V550S = glob.glob(Base_Path + "/single_captured/SingleCaptureImages/V550S/*.txt")
    single_captured_path_list_V610 = glob.glob(Base_Path + "/single_captured/SingleCaptureImages/V610/*.txt")
    print("single captured found for D40: ", len(single_captured_path_list_D40))
    print("single captured found for D70S: ", len(single_captured_path_list_D70S))
    print("single captured found for EOS600D: ", len(single_captured_path_list_EOS600D))
    print("single captured found for EPM2: ", len(single_captured_path_list_EPM2))
    print("single captured found for RX100: ", len(single_captured_path_list_RX100))
    print("single captured found for TZ7: ", len(single_captured_path_list_TZ7))
    print("single captured found for V550B: ", len(single_captured_path_list_V550B))
    print("single captured found for V550S: ", len(single_captured_path_list_V550S))
    print("single captured found for V610: ", len(single_captured_path_list_V610))
    #split into training and validation set (according to paper we use 15 images for avery camera as training)
    single_captured_D40_train, single_captured_D40_test = train_test_split(single_captured_path_list_D40, train_size=15, test_size=None, shuffle= True)
    single_captured_D70S_train, single_captured_D70S_test = train_test_split(single_captured_path_list_D70S, train_size=15, test_size=None, shuffle= True)
    single_captured_EOS600D_train, single_captured_EOS600D_test = train_test_split(single_captured_path_list_EOS600D, train_size=15, test_size=None, shuffle= True)
    single_captured_EPM2_train, single_captured_EPM2_test = train_test_split(single_captured_path_list_EPM2, train_size=15, test_size=None, shuffle= True)
    single_captured_RX100_train, single_captured_RX100_test = train_test_split(single_captured_path_list_RX100, train_size=15, test_size=None, shuffle= True)
    single_captured_TZ7_train, single_captured_TZ7_test = train_test_split(single_captured_path_list_TZ7, train_size=15, test_size=None, shuffle= True)
    single_captured_V550B_train, single_captured_V550B_test = train_test_split(single_captured_path_list_V550B, train_size=15, test_size=None, shuffle= True)
    single_captured_V550S_train, single_captured_V550S_test = train_test_split(single_captured_path_list_V550S, train_size=15, test_size=None, shuffle= True)
    single_captured_V610_train, single_captured_V610_test = train_test_split(single_captured_path_list_V610, train_size=15, test_size=None, shuffle= True)

    single_captured_train = single_captured_D40_train + single_captured_D70S_train + single_captured_EOS600D_train + single_captured_EPM2_train + single_captured_RX100_train + single_captured_TZ7_train + single_captured_V550B_train + single_captured_V550S_train + single_captured_V610_train
    single_captured_test = single_captured_D40_test + single_captured_D70S_test + single_captured_EOS600D_test + single_captured_EPM2_test + single_captured_RX100_test + single_captured_TZ7_test + single_captured_V550B_test + single_captured_V550S_test + single_captured_V610_test
        
    #save on text file all the paths
    with open(Base_Path + "/training_set_paths/single_captured_train.txt", 'w') as f:
        for p in single_captured_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/single_captured_D40_train.txt", 'w') as f:
        for p in single_captured_D40_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/single_captured_D70S_train.txt", 'w') as f:
        for p in single_captured_D70S_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/single_captured_EOS600D_train.txt", 'w') as f:
        for p in single_captured_EOS600D_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/single_captured_EPM2_train.txt", 'w') as f:
        for p in single_captured_EPM2_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/single_captured_RX100_train.txt", 'w') as f:
        for p in single_captured_RX100_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/single_captured_TZ7_train.txt", 'w') as f:
        for p in single_captured_TZ7_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/single_captured_V550B_train.txt", 'w') as f:
        for p in single_captured_V550B_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/single_captured_V550S_train.txt", 'w') as f:
        for p in single_captured_V550S_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/single_captured_V610_train.txt", 'w') as f:
        for p in single_captured_V610_train:
            f.write(str(p) + '\n')
    
    with open(Base_Path + "/test_set_paths/single_captured_test.txt", 'w') as f:
        for p in single_captured_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/single_captured_D40_test.txt", 'w') as f:
        for p in single_captured_D40_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/single_captured_D70S_test.txt", 'w') as f:
        for p in single_captured_D70S_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/single_captured_EOS600D_test.txt", 'w') as f:
        for p in single_captured_EOS600D_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/single_captured_EPM2_test.txt", 'w') as f:
        for p in single_captured_EPM2_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/single_captured_RX100_test.txt", 'w') as f:
        for p in single_captured_RX100_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/single_captured_TZ7_test.txt", 'w') as f:
        for p in single_captured_TZ7_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/single_captured_V550B_test.txt", 'w') as f:
        for p in single_captured_V550B_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/single_captured_V550S_test.txt", 'w') as f:
        for p in single_captured_V550S_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/single_captured_V610_test.txt", 'w') as f:
        for p in single_captured_V610_test:
            f.write(str(p) + '\n')
    print("all single captured path saved")


    #acquiring recaptured paths for every camera pair
    #camera 60D
    recaptured_path_list_60D_D40 = glob.glob(Base_Path + "/recaptured/RecapturedImages/60D/*D40*.txt")
    recaptured_path_list_60D_D70S = glob.glob(Base_Path + "/recaptured/RecapturedImages/60D/*D70S*.txt")
    recaptured_path_list_60D_EOS600D = glob.glob(Base_Path + "/recaptured/RecapturedImages/60D/*EOS600D*.txt")
    recaptured_path_list_60D_EPM2 = glob.glob(Base_Path + "/recaptured/RecapturedImages/60D/*EPM2*.txt")
    recaptured_path_list_60D_RX100 = glob.glob(Base_Path + "/recaptured/RecapturedImages/60D/*RX100*.txt")
    recaptured_path_list_60D_TZ7 = glob.glob(Base_Path + "/recaptured/RecapturedImages/60D/*TZ7*.txt")
    recaptured_path_list_60D_V550B = glob.glob(Base_Path + "/recaptured/RecapturedImages/60D/*V550B*.txt")
    recaptured_path_list_60D_V550S = glob.glob(Base_Path + "/recaptured/RecapturedImages/60D/*V550S*.txt")
    recaptured_path_list_60D_V610 = glob.glob(Base_Path + "/recaptured/RecapturedImages/60D/*V610*.txt")
    print("single captured found for 60D with D40: ", len(recaptured_path_list_60D_D40))
    print("single captured found for 60D with D70S: ", len(recaptured_path_list_60D_D70S))
    print("single captured found for 60D with EOS600D: ", len(recaptured_path_list_60D_EOS600D))
    print("single captured found for 60D with EPM2: ", len(recaptured_path_list_60D_EPM2))
    print("single captured found for 60D with RX100: ", len(recaptured_path_list_60D_RX100))
    print("single captured found for 60D with TZ7: ", len(recaptured_path_list_60D_TZ7))
    print("single captured found for 60D with V550B: ", len(recaptured_path_list_60D_V550B))
    print("single captured found for 60D with V550S: ", len(recaptured_path_list_60D_V550S))
    print("single captured found for 60D with V610: ", len(recaptured_path_list_60D_V610))
    recaptured_60D_D40_train, recaptured_60D_D40_test = train_test_split(recaptured_path_list_60D_D40, train_size=3, test_size=None, shuffle= True)
    recaptured_60D_D70S_train, recaptured_60D_D70S_test = train_test_split(recaptured_path_list_60D_D70S, train_size=3, test_size=None, shuffle= True)
    recaptured_60D_EOS600D_train, recaptured_60D_EOS600D_test = train_test_split(recaptured_path_list_60D_EOS600D, train_size=3, test_size=None, shuffle= True)
    recaptured_60D_EPM2_train, recaptured_60D_EPM2_test = train_test_split(recaptured_path_list_60D_EPM2, train_size=3, test_size=None, shuffle= True)
    recaptured_60D_RX100_train, recaptured_60D_RX100_test = train_test_split(recaptured_path_list_60D_RX100, train_size=3, test_size=None, shuffle= True)
    recaptured_60D_TZ7_train, recaptured_60D_TZ7_test = train_test_split(recaptured_path_list_60D_TZ7, train_size=3, test_size=None, shuffle= True)
    recaptured_60D_V550B_train, recaptured_60D_V550B_test = train_test_split(recaptured_path_list_60D_V550B, train_size=3, test_size=None, shuffle= True)
    recaptured_60D_V550S_train, recaptured_60D_V550S_test = train_test_split(recaptured_path_list_60D_V550S, train_size=3, test_size=None, shuffle= True)
    recaptured_60D_V610_train, recaptured_60D_V610_test = train_test_split(recaptured_path_list_60D_V610, train_size=3, test_size=None, shuffle= True)

    recaptured_60D_train = recaptured_60D_D40_train + recaptured_60D_D70S_train + recaptured_60D_EOS600D_train + recaptured_60D_EPM2_train + recaptured_60D_RX100_train + recaptured_60D_TZ7_train + recaptured_60D_V550B_train + recaptured_60D_V550S_train + recaptured_60D_V610_train
    recaptured_60D_test = recaptured_60D_D40_test + recaptured_60D_D70S_test + recaptured_60D_EOS600D_test + recaptured_60D_EPM2_test + recaptured_60D_RX100_test + recaptured_60D_TZ7_test + recaptured_60D_V550B_test + recaptured_60D_V550S_test + recaptured_60D_V610_test 

    #camera 600D
    recaptured_path_list_600D_D40 = glob.glob(Base_Path + "/recaptured/RecapturedImages/600D/*D40*.txt")
    recaptured_path_list_600D_D70S = glob.glob(Base_Path + "/recaptured/RecapturedImages/600D/*D70S*.txt")
    recaptured_path_list_600D_EOS600D = glob.glob(Base_Path + "/recaptured/RecapturedImages/600D/*600D*EOS600D*.txt")
    recaptured_path_list_600D_EPM2 = glob.glob(Base_Path + "/recaptured/RecapturedImages/600D/*EPM2*.txt")
    recaptured_path_list_600D_RX100 = glob.glob(Base_Path + "/recaptured/RecapturedImages/600D/*RX100*.txt")
    recaptured_path_list_600D_TZ7 = glob.glob(Base_Path + "/recaptured/RecapturedImages/600D/*TZ7*.txt")
    recaptured_path_list_600D_V550B = glob.glob(Base_Path + "/recaptured/RecapturedImages/600D/*V550B*.txt")
    recaptured_path_list_600D_V550S = glob.glob(Base_Path + "/recaptured/RecapturedImages/600D/*V550S*.txt")
    recaptured_path_list_600D_V610 = glob.glob(Base_Path + "/recaptured/RecapturedImages/600D/*V610*.txt")
    print("single captured found for 600D with D40: ", len(recaptured_path_list_600D_D40))
    print("single captured found for 600D with D70S: ", len(recaptured_path_list_600D_D70S))
    print("single captured found for 600D with EOS600D: ", len(recaptured_path_list_600D_EOS600D))
    print("single captured found for 600D with EPM2: ", len(recaptured_path_list_600D_EPM2))
    print("single captured found for 600D with RX100: ", len(recaptured_path_list_600D_RX100))
    print("single captured found for 600D with TZ7: ", len(recaptured_path_list_600D_TZ7))
    print("single captured found for 600D with V550B: ", len(recaptured_path_list_600D_V550B))
    print("single captured found for 600D with V550S: ", len(recaptured_path_list_600D_V550S))
    print("single captured found for 600D with V610: ", len(recaptured_path_list_600D_V610))
    recaptured_600D_D40_train, recaptured_600D_D40_test = train_test_split(recaptured_path_list_600D_D40, train_size=3, test_size=None, shuffle= True)
    recaptured_600D_D70S_train, recaptured_600D_D70S_test = train_test_split(recaptured_path_list_600D_D70S, train_size=3, test_size=None, shuffle= True)
    recaptured_600D_EOS600D_train, recaptured_600D_EOS600D_test = train_test_split(recaptured_path_list_600D_EOS600D, train_size=3, test_size=None, shuffle= True)
    recaptured_600D_EPM2_train, recaptured_600D_EPM2_test = train_test_split(recaptured_path_list_600D_EPM2, train_size=3, test_size=None, shuffle= True)
    recaptured_600D_RX100_train, recaptured_600D_RX100_test = train_test_split(recaptured_path_list_600D_RX100, train_size=3, test_size=None, shuffle= True)
    recaptured_600D_TZ7_train, recaptured_600D_TZ7_test = train_test_split(recaptured_path_list_600D_TZ7, train_size=3, test_size=None, shuffle= True)
    recaptured_600D_V550B_train, recaptured_600D_V550B_test = train_test_split(recaptured_path_list_600D_V550B, train_size=3, test_size=None, shuffle= True)
    recaptured_600D_V550S_train, recaptured_600D_V550S_test = train_test_split(recaptured_path_list_600D_V550S, train_size=3, test_size=None, shuffle= True)
    recaptured_600D_V610_train, recaptured_600D_V610_test = train_test_split(recaptured_path_list_600D_V610, train_size=3, test_size=None, shuffle= True)

    recaptured_600D_train = recaptured_600D_D40_train + recaptured_600D_D70S_train + recaptured_600D_EOS600D_train + recaptured_600D_EPM2_train + recaptured_600D_RX100_train + recaptured_600D_TZ7_train + recaptured_600D_V550B_train + recaptured_600D_V550S_train + recaptured_600D_V610_train
    recaptured_600D_test = recaptured_600D_D40_test + recaptured_600D_D70S_test + recaptured_600D_EOS600D_test + recaptured_600D_EPM2_test + recaptured_600D_RX100_test + recaptured_600D_TZ7_test + recaptured_600D_V550B_test + recaptured_600D_V550S_test + recaptured_600D_V610_test 

    #camera D70s
    recaptured_path_list_D70S_D40 = glob.glob(Base_Path + "/recaptured/RecapturedImages/D70S/*D40*.txt")
    recaptured_path_list_D70S_D70S = glob.glob(Base_Path + "/recaptured/RecapturedImages/D70S/*D70S*D70S*.txt")
    recaptured_path_list_D70S_EOS600D = glob.glob(Base_Path + "/recaptured/RecapturedImages/D70S/*EOS600D*.txt")
    recaptured_path_list_D70S_EPM2 = glob.glob(Base_Path + "/recaptured/RecapturedImages/D70S/*EPM2*.txt")
    recaptured_path_list_D70S_RX100 = glob.glob(Base_Path + "/recaptured/RecapturedImages/D70S/*RX100*.txt")
    recaptured_path_list_D70S_TZ7 = glob.glob(Base_Path + "/recaptured/RecapturedImages/D70S/*TZ7*.txt")
    recaptured_path_list_D70S_V550B = glob.glob(Base_Path + "/recaptured/RecapturedImages/D70S/*V550B*.txt")
    recaptured_path_list_D70S_V550S = glob.glob(Base_Path + "/recaptured/RecapturedImages/D70S/*V550S*.txt")
    recaptured_path_list_D70S_V610 = glob.glob(Base_Path + "/recaptured/RecapturedImages/D70S/*V610*.txt")
    print("single captured found for D70S with D40: ", len(recaptured_path_list_D70S_D40))
    print("single captured found for D70S with D70S: ", len(recaptured_path_list_D70S_D70S))
    print("single captured found for D70S with EOS600D: ", len(recaptured_path_list_D70S_EOS600D))
    print("single captured found for D70S with EPM2: ", len(recaptured_path_list_D70S_EPM2))
    print("single captured found for D70S with RX100: ", len(recaptured_path_list_D70S_RX100))
    print("single captured found for D70S with TZ7: ", len(recaptured_path_list_D70S_TZ7))
    print("single captured found for D70S with V550B: ", len(recaptured_path_list_D70S_V550B))
    print("single captured found for D70S with V550S: ", len(recaptured_path_list_D70S_V550S))
    print("single captured found for D70S with V610: ", len(recaptured_path_list_D70S_V610))
    recaptured_D70S_D40_train, recaptured_D70S_D40_test = train_test_split(recaptured_path_list_D70S_D40, train_size=3, test_size=None, shuffle= True)
    recaptured_D70S_D70S_train, recaptured_D70S_D70S_test = train_test_split(recaptured_path_list_D70S_D70S, train_size=3, test_size=None, shuffle= True)
    recaptured_D70S_EOS600D_train, recaptured_D70S_EOS600D_test = train_test_split(recaptured_path_list_D70S_EOS600D, train_size=3, test_size=None, shuffle= True)
    recaptured_D70S_EPM2_train, recaptured_D70S_EPM2_test = train_test_split(recaptured_path_list_D70S_EPM2, train_size=3, test_size=None, shuffle= True)
    recaptured_D70S_RX100_train, recaptured_D70S_RX100_test = train_test_split(recaptured_path_list_D70S_RX100, train_size=3, test_size=None, shuffle= True)
    recaptured_D70S_TZ7_train, recaptured_D70S_TZ7_test = train_test_split(recaptured_path_list_D70S_TZ7, train_size=3, test_size=None, shuffle= True)
    recaptured_D70S_V550B_train, recaptured_D70S_V550B_test = train_test_split(recaptured_path_list_D70S_V550B, train_size=3, test_size=None, shuffle= True)
    recaptured_D70S_V550S_train, recaptured_D70S_V550S_test = train_test_split(recaptured_path_list_D70S_V550S, train_size=3, test_size=None, shuffle= True)
    recaptured_D70S_V610_train, recaptured_D70S_V610_test = train_test_split(recaptured_path_list_D70S_V610, train_size=3, test_size=None, shuffle= True)

    recaptured_D70S_train = recaptured_D70S_D40_train + recaptured_D70S_D70S_train + recaptured_D70S_EOS600D_train + recaptured_D70S_EPM2_train + recaptured_D70S_RX100_train + recaptured_D70S_TZ7_train + recaptured_D70S_V550B_train + recaptured_D70S_V550S_train + recaptured_D70S_V610_train
    recaptured_D70S_test = recaptured_D70S_D40_test + recaptured_D70S_D70S_test + recaptured_D70S_EOS600D_test + recaptured_D70S_EPM2_test + recaptured_D70S_RX100_test + recaptured_D70S_TZ7_test + recaptured_D70S_V550B_test + recaptured_D70S_V550S_test + recaptured_D70S_V610_test 

    #camera D3200
    recaptured_path_list_D3200_D40 = glob.glob(Base_Path + "/recaptured/RecapturedImages/D3200/*D40*.txt")
    recaptured_path_list_D3200_D70S = glob.glob(Base_Path + "/recaptured/RecapturedImages/D3200/*D70S*.txt")
    recaptured_path_list_D3200_EOS600D = glob.glob(Base_Path + "/recaptured/RecapturedImages/D3200/*EOS600D*.txt")
    recaptured_path_list_D3200_EPM2 = glob.glob(Base_Path + "/recaptured/RecapturedImages/D3200/*EPM2*.txt")
    recaptured_path_list_D3200_RX100 = glob.glob(Base_Path + "/recaptured/RecapturedImages/D3200/*RX100*.txt")
    recaptured_path_list_D3200_TZ7 = glob.glob(Base_Path + "/recaptured/RecapturedImages/D3200/*TZ7*.txt")
    recaptured_path_list_D3200_V550B = glob.glob(Base_Path + "/recaptured/RecapturedImages/D3200/*V550B*.txt")
    recaptured_path_list_D3200_V550S = glob.glob(Base_Path + "/recaptured/RecapturedImages/D3200/*V550S*.txt")
    recaptured_path_list_D3200_V610 = glob.glob(Base_Path + "/recaptured/RecapturedImages/D3200/*V610*.txt")
    print("single captured found for D3200 with D40: ", len(recaptured_path_list_D3200_D40))
    print("single captured found for D3200 with D70S: ", len(recaptured_path_list_D3200_D70S))
    print("single captured found for D3200 with EOS600D: ", len(recaptured_path_list_D3200_EOS600D))
    print("single captured found for D3200 with EPM2: ", len(recaptured_path_list_D3200_EPM2))
    print("single captured found for D3200 with RX100: ", len(recaptured_path_list_D3200_RX100))
    print("single captured found for D3200 with TZ7: ", len(recaptured_path_list_D3200_TZ7))
    print("single captured found for D3200 with V550B: ", len(recaptured_path_list_D3200_V550B))
    print("single captured found for D3200 with V550S: ", len(recaptured_path_list_D3200_V550S))
    print("single captured found for D3200 with V610: ", len(recaptured_path_list_D3200_V610))
    recaptured_D3200_D40_train, recaptured_D3200_D40_test = train_test_split(recaptured_path_list_D3200_D40, train_size=3, test_size=None, shuffle= True)
    recaptured_D3200_D70S_train, recaptured_D3200_D70S_test = train_test_split(recaptured_path_list_D3200_D70S, train_size=3, test_size=None, shuffle= True)
    recaptured_D3200_EOS600D_train, recaptured_D3200_EOS600D_test = train_test_split(recaptured_path_list_D3200_EOS600D, train_size=3, test_size=None, shuffle= True)
    recaptured_D3200_EPM2_train, recaptured_D3200_EPM2_test = train_test_split(recaptured_path_list_D3200_EPM2, train_size=3, test_size=None, shuffle= True)
    recaptured_D3200_RX100_train, recaptured_D3200_RX100_test = train_test_split(recaptured_path_list_D3200_RX100, train_size=3, test_size=None, shuffle= True)
    recaptured_D3200_TZ7_train, recaptured_D3200_TZ7_test = train_test_split(recaptured_path_list_D3200_TZ7, train_size=3, test_size=None, shuffle= True)
    recaptured_D3200_V550B_train, recaptured_D3200_V550B_test = train_test_split(recaptured_path_list_D3200_V550B, train_size=3, test_size=None, shuffle= True)
    recaptured_D3200_V550S_train, recaptured_D3200_V550S_test = train_test_split(recaptured_path_list_D3200_V550S, train_size=3, test_size=None, shuffle= True)
    recaptured_D3200_V610_train, recaptured_D3200_V610_test = train_test_split(recaptured_path_list_D3200_V610, train_size=3, test_size=None, shuffle= True)

    recaptured_D3200_train = recaptured_D3200_D40_train + recaptured_D3200_D70S_train + recaptured_D3200_EOS600D_train + recaptured_D3200_EPM2_train + recaptured_D3200_RX100_train + recaptured_D3200_TZ7_train + recaptured_D3200_V550B_train + recaptured_D3200_V550S_train + recaptured_D3200_V610_train
    recaptured_D3200_test = recaptured_D3200_D40_test + recaptured_D3200_D70S_test + recaptured_D3200_EOS600D_test + recaptured_D3200_EPM2_test + recaptured_D3200_RX100_test + recaptured_D3200_TZ7_test + recaptured_D3200_V550B_test + recaptured_D3200_V550S_test + recaptured_D3200_V610_test 

    #camera EPM2
    recaptured_path_list_EPM2_D40 = glob.glob(Base_Path + "/recaptured/RecapturedImages/EPM2/*D40*.txt")
    recaptured_path_list_EPM2_D70S = glob.glob(Base_Path + "/recaptured/RecapturedImages/EPM2/*D70S*.txt")
    recaptured_path_list_EPM2_EOS600D = glob.glob(Base_Path + "/recaptured/RecapturedImages/EPM2/*EOS600D*.txt")
    recaptured_path_list_EPM2_EPM2 = glob.glob(Base_Path + "/recaptured/RecapturedImages/EPM2/*EPM2*EPM2*.txt")
    recaptured_path_list_EPM2_RX100 = glob.glob(Base_Path + "/recaptured/RecapturedImages/EPM2/*RX100*.txt")
    recaptured_path_list_EPM2_TZ7 = glob.glob(Base_Path + "/recaptured/RecapturedImages/EPM2/*TZ7*.txt")
    recaptured_path_list_EPM2_V550B = glob.glob(Base_Path + "/recaptured/RecapturedImages/EPM2/*V550B*.txt")
    recaptured_path_list_EPM2_V550S = glob.glob(Base_Path + "/recaptured/RecapturedImages/EPM2/*V550S*.txt")
    recaptured_path_list_EPM2_V610 = glob.glob(Base_Path + "/recaptured/RecapturedImages/EPM2/*V610*.txt")
    print("single captured found for EPM2 with D40: ", len(recaptured_path_list_EPM2_D40))
    print("single captured found for EPM2 with D70S: ", len(recaptured_path_list_EPM2_D70S))
    print("single captured found for EPM2 with EOS600D: ", len(recaptured_path_list_EPM2_EOS600D))
    print("single captured found for EPM2 with EPM2: ", len(recaptured_path_list_EPM2_EPM2))
    print("single captured found for EPM2 with RX100: ", len(recaptured_path_list_EPM2_RX100))
    print("single captured found for EPM2 with TZ7: ", len(recaptured_path_list_EPM2_TZ7))
    print("single captured found for EPM2 with V550B: ", len(recaptured_path_list_EPM2_V550B))
    print("single captured found for EPM2 with V550S: ", len(recaptured_path_list_EPM2_V550S))
    print("single captured found for EPM2 with V610: ", len(recaptured_path_list_EPM2_V610))
    recaptured_EPM2_D40_train, recaptured_EPM2_D40_test = train_test_split(recaptured_path_list_EPM2_D40, train_size=3, test_size=None, shuffle= True)
    recaptured_EPM2_D70S_train, recaptured_EPM2_D70S_test = train_test_split(recaptured_path_list_EPM2_D70S, train_size=3, test_size=None, shuffle= True)
    recaptured_EPM2_EOS600D_train, recaptured_EPM2_EOS600D_test = train_test_split(recaptured_path_list_EPM2_EOS600D, train_size=3, test_size=None, shuffle= True)
    recaptured_EPM2_EPM2_train, recaptured_EPM2_EPM2_test = train_test_split(recaptured_path_list_EPM2_EPM2, train_size=3, test_size=None, shuffle= True)
    recaptured_EPM2_RX100_train, recaptured_EPM2_RX100_test = train_test_split(recaptured_path_list_EPM2_RX100, train_size=3, test_size=None, shuffle= True)
    recaptured_EPM2_TZ7_train, recaptured_EPM2_TZ7_test = train_test_split(recaptured_path_list_EPM2_TZ7, train_size=3, test_size=None, shuffle= True)
    recaptured_EPM2_V550B_train, recaptured_EPM2_V550B_test = train_test_split(recaptured_path_list_EPM2_V550B, train_size=3, test_size=None, shuffle= True)
    recaptured_EPM2_V550S_train, recaptured_EPM2_V550S_test = train_test_split(recaptured_path_list_EPM2_V550S, train_size=3, test_size=None, shuffle= True)
    recaptured_EPM2_V610_train, recaptured_EPM2_V610_test = train_test_split(recaptured_path_list_EPM2_V610, train_size=3, test_size=None, shuffle= True)

    recaptured_EPM2_train = recaptured_EPM2_D40_train + recaptured_EPM2_D70S_train + recaptured_EPM2_EOS600D_train + recaptured_EPM2_EPM2_train + recaptured_EPM2_RX100_train + recaptured_EPM2_TZ7_train + recaptured_EPM2_V550B_train + recaptured_EPM2_V550S_train + recaptured_EPM2_V610_train
    recaptured_EPM2_test = recaptured_EPM2_D40_test + recaptured_EPM2_D70S_test + recaptured_EPM2_EOS600D_test + recaptured_EPM2_EPM2_test + recaptured_EPM2_RX100_test + recaptured_EPM2_TZ7_test + recaptured_EPM2_V550B_test + recaptured_EPM2_V550S_test + recaptured_EPM2_V610_test 

    #camera RX100
    recaptured_path_list_RX100_D40 = glob.glob(Base_Path + "/recaptured/RecapturedImages/RX100/*D40*.txt")
    recaptured_path_list_RX100_D70S = glob.glob(Base_Path + "/recaptured/RecapturedImages/RX100/*D70S*.txt")
    recaptured_path_list_RX100_EOS600D = glob.glob(Base_Path + "/recaptured/RecapturedImages/RX100/*EOS600D*.txt")
    recaptured_path_list_RX100_EPM2 = glob.glob(Base_Path + "/recaptured/RecapturedImages/RX100/*EPM2*.txt")
    recaptured_path_list_RX100_RX100 = glob.glob(Base_Path + "/recaptured/RecapturedImages/RX100/*RX100*RX100*.txt")
    recaptured_path_list_RX100_TZ7 = glob.glob(Base_Path + "/recaptured/RecapturedImages/RX100/*TZ7*.txt")
    recaptured_path_list_RX100_V550B = glob.glob(Base_Path + "/recaptured/RecapturedImages/RX100/*V550B*.txt")
    recaptured_path_list_RX100_V550S = glob.glob(Base_Path + "/recaptured/RecapturedImages/RX100/*V550S*.txt")
    recaptured_path_list_RX100_V610 = glob.glob(Base_Path + "/recaptured/RecapturedImages/RX100/*V610*.txt")
    print("single captured found for RX100 with D40: ", len(recaptured_path_list_RX100_D40))
    print("single captured found for RX100 with D70S: ", len(recaptured_path_list_RX100_D70S))
    print("single captured found for RX100 with EOS600D: ", len(recaptured_path_list_RX100_EOS600D))
    print("single captured found for RX100 with EPM2: ", len(recaptured_path_list_RX100_EPM2))
    print("single captured found for RX100 with RX100: ", len(recaptured_path_list_RX100_RX100))
    print("single captured found for RX100 with TZ7: ", len(recaptured_path_list_RX100_TZ7))
    print("single captured found for RX100 with V550B: ", len(recaptured_path_list_RX100_V550B))
    print("single captured found for RX100 with V550S: ", len(recaptured_path_list_RX100_V550S))
    print("single captured found for RX100 with V610: ", len(recaptured_path_list_RX100_V610))
    recaptured_RX100_D40_train, recaptured_RX100_D40_test = train_test_split(recaptured_path_list_RX100_D40, train_size=3, test_size=None, shuffle= True)
    recaptured_RX100_D70S_train, recaptured_RX100_D70S_test = train_test_split(recaptured_path_list_RX100_D70S, train_size=3, test_size=None, shuffle= True)
    recaptured_RX100_EOS600D_train, recaptured_RX100_EOS600D_test = train_test_split(recaptured_path_list_RX100_EOS600D, train_size=3, test_size=None, shuffle= True)
    recaptured_RX100_EPM2_train, recaptured_RX100_EPM2_test = train_test_split(recaptured_path_list_RX100_EPM2, train_size=3, test_size=None, shuffle= True)
    recaptured_RX100_RX100_train, recaptured_RX100_RX100_test = train_test_split(recaptured_path_list_RX100_RX100, train_size=3, test_size=None, shuffle= True)
    recaptured_RX100_TZ7_train, recaptured_RX100_TZ7_test = train_test_split(recaptured_path_list_RX100_TZ7, train_size=3, test_size=None, shuffle= True)
    recaptured_RX100_V550B_train, recaptured_RX100_V550B_test = train_test_split(recaptured_path_list_RX100_V550B, train_size=3, test_size=None, shuffle= True)
    recaptured_RX100_V550S_train, recaptured_RX100_V550S_test = train_test_split(recaptured_path_list_RX100_V550S, train_size=3, test_size=None, shuffle= True)
    recaptured_RX100_V610_train, recaptured_RX100_V610_test = train_test_split(recaptured_path_list_RX100_V610, train_size=3, test_size=None, shuffle= True)

    recaptured_RX100_train = recaptured_RX100_D40_train + recaptured_RX100_D70S_train + recaptured_RX100_EOS600D_train + recaptured_RX100_EPM2_train + recaptured_RX100_RX100_train + recaptured_RX100_TZ7_train + recaptured_RX100_V550B_train + recaptured_RX100_V550S_train + recaptured_RX100_V610_train
    recaptured_RX100_test = recaptured_RX100_D40_test + recaptured_RX100_D70S_test + recaptured_RX100_EOS600D_test + recaptured_RX100_EPM2_test + recaptured_RX100_RX100_test + recaptured_RX100_TZ7_test + recaptured_RX100_V550B_test + recaptured_RX100_V550S_test + recaptured_RX100_V610_test 

    #camera TZ7
    recaptured_path_list_TZ7_D40 = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ7/*D40*.txt")
    recaptured_path_list_TZ7_D70S = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ7/*D70S*.txt")
    recaptured_path_list_TZ7_EOS600D = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ7/*EOS600D*.txt")
    recaptured_path_list_TZ7_EPM2 = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ7/*EPM2*.txt")
    recaptured_path_list_TZ7_RX100 = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ7/*RX100*.txt")
    recaptured_path_list_TZ7_TZ7 = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ7/*TZ7*TZ7*.txt")
    recaptured_path_list_TZ7_V550B = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ7/*V550B*.txt")
    recaptured_path_list_TZ7_V550S = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ7/*V550S*.txt")
    recaptured_path_list_TZ7_V610 = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ7/*V610*.txt")
    print("single captured found for TZ7 with D40: ", len(recaptured_path_list_TZ7_D40))
    print("single captured found for TZ7 with D70S: ", len(recaptured_path_list_TZ7_D70S))
    print("single captured found for TZ7 with EOS600D: ", len(recaptured_path_list_TZ7_EOS600D))
    print("single captured found for TZ7 with EPM2: ", len(recaptured_path_list_TZ7_EPM2))
    print("single captured found for TZ7 with RX100: ", len(recaptured_path_list_TZ7_RX100))
    print("single captured found for TZ7 with TZ7: ", len(recaptured_path_list_TZ7_TZ7))
    print("single captured found for TZ7 with V550B: ", len(recaptured_path_list_TZ7_V550B))
    print("single captured found for TZ7 with V550S: ", len(recaptured_path_list_TZ7_V550S))
    print("single captured found for TZ7 with V610: ", len(recaptured_path_list_TZ7_V610))
    recaptured_TZ7_D40_train, recaptured_TZ7_D40_test = train_test_split(recaptured_path_list_TZ7_D40, train_size=3, test_size=None, shuffle= True)
    recaptured_TZ7_D70S_train, recaptured_TZ7_D70S_test = train_test_split(recaptured_path_list_TZ7_D70S, train_size=3, test_size=None, shuffle= True)
    recaptured_TZ7_EOS600D_train, recaptured_TZ7_EOS600D_test = train_test_split(recaptured_path_list_TZ7_EOS600D, train_size=3, test_size=None, shuffle= True)
    recaptured_TZ7_EPM2_train, recaptured_TZ7_EPM2_test = train_test_split(recaptured_path_list_TZ7_EPM2, train_size=3, test_size=None, shuffle= True)
    recaptured_TZ7_RX100_train, recaptured_TZ7_RX100_test = train_test_split(recaptured_path_list_TZ7_RX100, train_size=3, test_size=None, shuffle= True)
    recaptured_TZ7_TZ7_train, recaptured_TZ7_TZ7_test = train_test_split(recaptured_path_list_TZ7_TZ7, train_size=3, test_size=None, shuffle= True)
    recaptured_TZ7_V550B_train, recaptured_TZ7_V550B_test = train_test_split(recaptured_path_list_TZ7_V550B, train_size=3, test_size=None, shuffle= True)
    recaptured_TZ7_V550S_train, recaptured_TZ7_V550S_test = train_test_split(recaptured_path_list_TZ7_V550S, train_size=3, test_size=None, shuffle= True)
    recaptured_TZ7_V610_train, recaptured_TZ7_V610_test = train_test_split(recaptured_path_list_TZ7_V610, train_size=3, test_size=None, shuffle= True)

    recaptured_TZ7_train = recaptured_TZ7_D40_train + recaptured_TZ7_D70S_train + recaptured_TZ7_EOS600D_train + recaptured_TZ7_EPM2_train + recaptured_TZ7_RX100_train + recaptured_TZ7_TZ7_train + recaptured_TZ7_V550B_train + recaptured_TZ7_V550S_train + recaptured_TZ7_V610_train
    recaptured_TZ7_test = recaptured_TZ7_D40_test + recaptured_TZ7_D70S_test + recaptured_TZ7_EOS600D_test + recaptured_TZ7_EPM2_test + recaptured_TZ7_RX100_test + recaptured_TZ7_TZ7_test + recaptured_TZ7_V550B_test + recaptured_TZ7_V550S_test + recaptured_TZ7_V610_test 

    #camera TZ10
    recaptured_path_list_TZ10_D40 = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ10/*D40*.txt")
    recaptured_path_list_TZ10_D70S = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ10/*D70S*.txt")
    recaptured_path_list_TZ10_EOS600D = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ10/*EOS600D*.txt")
    recaptured_path_list_TZ10_EPM2 = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ10/*EPM2*.txt")
    recaptured_path_list_TZ10_RX100 = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ10/*RX100*.txt")
    recaptured_path_list_TZ10_TZ7 = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ10/*TZ7*.txt")
    recaptured_path_list_TZ10_V550B = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ10/*V550B*.txt")
    recaptured_path_list_TZ10_V550S = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ10/*V550S*.txt")
    recaptured_path_list_TZ10_V610 = glob.glob(Base_Path + "/recaptured/RecapturedImages/TZ10/*V610*.txt")
    print("single captured found for TZ10 with D40: ", len(recaptured_path_list_TZ10_D40))
    print("single captured found for TZ10 with D70S: ", len(recaptured_path_list_TZ10_D70S))
    print("single captured found for TZ10 with EOS600D: ", len(recaptured_path_list_TZ10_EOS600D))
    print("single captured found for TZ10 with EPM2: ", len(recaptured_path_list_TZ10_EPM2))
    print("single captured found for TZ10 with RX100: ", len(recaptured_path_list_TZ10_RX100))
    print("single captured found for TZ10 with TZ7: ", len(recaptured_path_list_TZ10_TZ7))
    print("single captured found for TZ10 with V550B: ", len(recaptured_path_list_TZ10_V550B))
    print("single captured found for TZ10 with V550S: ", len(recaptured_path_list_TZ10_V550S))
    print("single captured found for TZ10 with V610: ", len(recaptured_path_list_TZ10_V610))
    recaptured_TZ10_D40_train, recaptured_TZ10_D40_test = train_test_split(recaptured_path_list_TZ10_D40, train_size=3, test_size=None, shuffle= True)
    recaptured_TZ10_D70S_train, recaptured_TZ10_D70S_test = train_test_split(recaptured_path_list_TZ10_D70S, train_size=3, test_size=None, shuffle= True)
    recaptured_TZ10_EOS600D_train, recaptured_TZ10_EOS600D_test = train_test_split(recaptured_path_list_TZ10_EOS600D, train_size=3, test_size=None, shuffle= True)
    recaptured_TZ10_EPM2_train, recaptured_TZ10_EPM2_test = train_test_split(recaptured_path_list_TZ10_EPM2, train_size=3, test_size=None, shuffle= True)
    recaptured_TZ10_RX100_train, recaptured_TZ10_RX100_test = train_test_split(recaptured_path_list_TZ10_RX100, train_size=3, test_size=None, shuffle= True)
    recaptured_TZ10_TZ7_train, recaptured_TZ10_TZ7_test = train_test_split(recaptured_path_list_TZ10_TZ7, train_size=3, test_size=None, shuffle= True)
    recaptured_TZ10_V550B_train, recaptured_TZ10_V550B_test = train_test_split(recaptured_path_list_TZ10_V550B, train_size=3, test_size=None, shuffle= True)
    recaptured_TZ10_V550S_train, recaptured_TZ10_V550S_test = train_test_split(recaptured_path_list_TZ10_V550S, train_size=3, test_size=None, shuffle= True)
    recaptured_TZ10_V610_train, recaptured_TZ10_V610_test = train_test_split(recaptured_path_list_TZ10_V610, train_size=3, test_size=None, shuffle= True)

    recaptured_TZ10_train = recaptured_TZ10_D40_train + recaptured_TZ10_D70S_train + recaptured_TZ10_EOS600D_train + recaptured_TZ10_EPM2_train + recaptured_TZ10_RX100_train + recaptured_TZ10_TZ7_train + recaptured_TZ10_V550B_train + recaptured_TZ10_V550S_train + recaptured_TZ10_V610_train
    recaptured_TZ10_test = recaptured_TZ10_D40_test + recaptured_TZ10_D70S_test + recaptured_TZ10_EOS600D_test + recaptured_TZ10_EPM2_test + recaptured_TZ10_RX100_test + recaptured_TZ10_TZ7_test + recaptured_TZ10_V550B_test + recaptured_TZ10_V550S_test + recaptured_TZ10_V610_test 

    recaptured_train = recaptured_60D_train + recaptured_600D_train + recaptured_D70S_train + recaptured_D3200_train + recaptured_EPM2_train + recaptured_RX100_train + recaptured_TZ7_train + recaptured_TZ10_train
    recaptured_test = recaptured_60D_test + recaptured_600D_test + recaptured_D70S_test + recaptured_D3200_test + recaptured_EPM2_test + recaptured_RX100_test + recaptured_TZ7_test + recaptured_TZ10_test 

    #save on text file all the paths
    with open(Base_Path + "/training_set_paths/recaptured_train.txt", 'w') as f:
        for p in recaptured_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/recaptured_60D_train.txt", 'w') as f:
        for p in recaptured_60D_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/recaptured_600D_train.txt", 'w') as f:
        for p in recaptured_600D_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/recaptured_D70S_train.txt", 'w') as f:
        for p in recaptured_D70S_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/recaptured_D3200_train.txt", 'w') as f:
        for p in recaptured_D3200_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/recaptured_EPM2_train.txt", 'w') as f:
        for p in recaptured_EPM2_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/recaptured_RX100_train.txt", 'w') as f:
        for p in recaptured_RX100_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/recaptured_TZ7_train.txt", 'w') as f:
        for p in recaptured_TZ7_train:
            f.write(str(p) + '\n')
    with open(Base_Path + "/training_set_paths/recaptured_TZ10_train.txt", 'w') as f:
        for p in recaptured_TZ10_train:
            f.write(str(p) + '\n')
    
    with open(Base_Path + "/test_set_paths/recaptured_test.txt", 'w') as f:
        for p in recaptured_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/recaptured_60D_test.txt", 'w') as f:
        for p in recaptured_60D_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/recaptured_600D_test.txt", 'w') as f:
        for p in recaptured_600D_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/recaptured_D70S_test.txt", 'w') as f:
        for p in recaptured_D70S_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/recaptured_D3200_test.txt", 'w') as f:
        for p in recaptured_D3200_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/recaptured_EPM2_test.txt", 'w') as f:
        for p in recaptured_EPM2_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/recaptured_RX100_test.txt", 'w') as f:
        for p in recaptured_RX100_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/recaptured_TZ7_test.txt", 'w') as f:
        for p in recaptured_TZ7_test:
            f.write(str(p) + '\n')
    with open(Base_Path + "/test_set_paths/recaptured_TZ10_test.txt", 'w') as f:
        for p in recaptured_TZ10_test:
            f.write(str(p) + '\n')
    print("all recaptured path saved")

    return 1

if __name__ == '__main__':
	output = main()