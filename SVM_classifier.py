import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay 
from block_functions import approximation_error as ape
from block_functions import lambda_calculus as lc
from joblib import dump, load
import dataframe_image as dfi

#constant declaration
LABEL_SC = 0
LABEL_RC = 1

#Base_Path = "./../../dataset/complete_dataset"
Base_Path = "./complete_dataset"

def write2file (data2write):
    res_str = np.empty((data2write.shape[0],), dtype=object)
    for i in range (data2write.shape[0]):
        if data2write[i] == 1:
            res_str[i] = 'Recaptured'
        else :
            res_str[i] = 'Single Captured'
    df = pd.DataFrame(res_str, columns=['Predicted label'])
    return df

def initialize():
    data = {
        'img_path' : [],
        'label' : [],
        'lambda_avg' : [],
        'approx_error' : []
    } 
    return data

def extracting_feature(test, label):
    data = initialize()
    Dsc_T = np.loadtxt(Base_Path +"/dictionaries/Dsc_T.txt")
    Drc_T = np.loadtxt(Base_Path +"/dictionaries/Drc_T.txt")

    #orthogonal matching pursuit
    for x in test:
        Qi = np.loadtxt(x)
        if np.any(Qi): 
            Esc = ape.approximation_error(Qi, Dsc_T)
            Erc = ape.approximation_error(Qi, Drc_T)
            Ed = Esc - Erc
            lambda_avg = lc.find_lambda_average(Qi)
            data["img_path"].append(x)
            data['label'].append(label)
            data['approx_error'].append(Ed[0,0])
            data['lambda_avg'].append(lambda_avg)
        else:
            print("No Qi found for: ", x)
    return data

def classificate(svm, paths, label, camera):
    time1 = time.time()
    data = extracting_feature(paths, label)
    df = pd.DataFrame(data)
    X = df[['lambda_avg','approx_error']].values
    Y = df[['label']].values
    res = svm.predict(X)
    CM = confusion_matrix(Y, res)
    dataframe_svm = write2file(res) 
    items = [df,dataframe_svm]
    df_res = pd.concat(items, axis=1)
    time2 = time.time()
    print("Tempo impiegato per classificazione", camera, ": --- %s seconds ---" % (time2 - time1))
    CM_new = np.zeros((2,2))
    if CM.shape[0] == 1:
        if label == 1:
            CM_new[1][1] = CM[0][0]
        else:
            CM_new[0][0] = CM[0][0]
    else:
        CM_new[0:CM.shape[0], 0:CM.shape[1]] = CM
    print(f'Confusion matrix : \n {CM_new}, \n TN : {CM_new[0][0]}, FN : {CM_new[0][1]}, TP : {CM_new[1][1]}, FP : {CM_new[0][1]}')
    dati = (camera, df_res, X, Y, res, CM_new, CM_new[0][0], CM_new[1][0], CM_new[1][1], CM_new[0][1], label)
    df_res.to_csv(Base_Path + "/test_set_paths/" + camera + "table.csv")

    return dati

#for each column highlight the max value
def highlight_max(s, props=''):
    return np.where(s == np.nanmax(s.values), props, '')
def highlight_min(s, props=''):
    return np.where(s == np.nanmin(s.values), props, '')

def report_results(results_list):
    data_recaptured = []
    data_single_captured = []
    TN_r = 0
    FN_r = 0
    TP_r = 0
    FP_r = 0
    TN_s = 0
    FN_s = 0
    TP_s = 0
    FP_s = 0
    with open(Base_Path + "/report/result_report.txt", 'w') as f:
        for res in results_list:
            TN = res[6]
            FN = res[7]
            TP = res[8]
            FP = res[9]
            if res[10] == 1:
                TN_r += TN
                FN_r += FN
                TP_r += TP
                FP_r += FP
            else:
                TN_s += TN
                FN_s += FN
                TP_s += TP
                FP_s += FP
            accuracy = (TN + TP)/(TN + TP + FN + FP)
            print("Stats for camera: " + res[0])
            f.write("Stats for camera: " + res[0] + '\n')
            print("TN: ", TN)
            f.write("TN: " + np.array2string(TN) + '\n')
            print("FN: ", FN)
            f.write("FN: " + np.array2string(FN) + '\n')
            print("TP: ", TP)
            f.write("TP: " + np.array2string(TP) + '\n')
            print("FP: ", FP)
            f.write("FP: " + np.array2string(FP) + '\n')
            f.write("Accuracy(%): " + np.array2string(accuracy, precision = 2) + '\n')
            f.write('\n')
            if 'rec' in str(res):
                data_recaptured.append([TP+FN+FP+TN, FN, TP, accuracy])
            else:
                data_single_captured.append([TP+FN+FP+TN, FP, TN, accuracy])

        accuracy_rc = (TN_r + TP_r)/(TN_r + TP_r + FN_r + FP_r)
        accuracy_sc = (TN_s + TP_s)/(TN_s + TP_s + FN_s + FP_s)
        accuracy_tot = (TN_r + TP_r + TN_s + TP_s)/(TN_r + TN_s + TP_r + TP_s + FN_r + FN_s + FP_r + FP_s)
        f.write("Accuracy(%) for recaptured: " + np.array2string(accuracy_rc, precision = 2) + '\n')
        f.write("Accuracy(%) for single captured: " + np.array2string(accuracy_sc, precision = 2) + '\n')
        f.write("Accuracy(%) tot: " + np.array2string(accuracy_tot, precision = 2) + '\n')
        print(f'Data recaptured : {data_recaptured}')
        print(f'Data single captured : {data_single_captured}')
    table_making(data_recaptured, accuracy_rc, 'r')
    table_making(data_single_captured, accuracy_sc, 's')


def table_making(data, accuracy, c):

    index_single = pd.Index(['D40','D70S','EOS600D','EPM2','RX100','TZ7','V550B','V550S','V610'], name='Camera models')
    index_recaptured= pd.Index(['60D', '600D', 'D70S','D3200','EPM2','RX100','TZ7','TZ10'], name='Camera models')

    # order -> FN+TP,FN,TP,ACCURACY
    # print(f'Confusion matrix : \n {CM_new}, \n TN : {CM_new[0][0]}, FN : {CM_new[0][1]}, TP : {CM_new[1][0]}, FP : {CM_new[1][1]}')
    # dati = (camera, df_res, X, Y, res, CM_new, CM_new[0][0], CM_new[1][0], CM_new[1][1], CM_new[0][1], label)   
    cols=(['Classification Results'],['Accuracy'])
    if c == 'r':
        columns=pd.MultiIndex.from_product([['Classification Results'],['No. of Images','Wrong', 'Correct','Accuracy']], names=['','Recaptured'])
        df = pd.DataFrame(data, index_recaptured,columns=columns)

    else :
        columns=pd.MultiIndex.from_product([['Classification Results'],['No. of Images','Wrong', 'Correct','Accuracy']], names=['','Single Recaptured'])
        df = pd.DataFrame(data, index_single,columns=columns)
        
    df.loc[('Average'),cols] = accuracy
    s = df.style.format(formatter={('Classification Results','Accuracy'):'{:.2f}',
                               ('Classification Results','No. of Images'):'{:.0f}',
                               ('Classification Results','Wrong'):'{:.0f}',
                               ('Classification Results','Correct'):'{:.0f}'},na_rep='-')
    if c == 'r' :
        subset=(index_recaptured[:-1], [('Classification Results','Correct')])
        subset2=(index_recaptured[:-1], [('Classification Results','Wrong')])
        subset3=(index_recaptured[:-1], [('Classification Results','Accuracy')])
        s.set_caption("Table of recapture images")
    else :
        s.set_caption('Table of original images')
        subset=(index_single[:-1], [('Classification Results','Correct')])
        subset2=(index_single[:-1], [('Classification Results','Wrong')])
        subset3=(index_single[:-1], [('Classification Results','Accuracy')])

    s.set_table_styles([{'selector': 'th.col_heading', 'props': 'text-align: center; '},
                        {'selector': 'th.col_heading.level0', 'props': 'font-size: 1.5em;background-color: #000066; color: white;'},
                        {'selector': 'td', 'props': 'text-align: center; font-weight: bold; border-left: 2px solid #000066 '},], 
                        overwrite=False, axis=0)

    s.apply(highlight_max, props='color:white;background-color:green', axis=0,subset=subset)
    s.apply(highlight_max, props='color:white;background-color:red', axis=0,subset=subset2)
    s.apply(highlight_max, props='color:white;background-color:green', axis=0,subset=subset3)
    s.apply(highlight_min, props='color:white;background-color:red', axis=0,subset=subset3)
    if c == 's':
        dfi.export(s, Base_Path + '/report/table_single_captured.png')
    else :
        dfi.export(s, Base_Path + '/report/table_recaptured.png')


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_results(results_list, svm):
    rows = 2
    tot = len(results_list)
    cols = tot // rows
    cols += tot % rows

    fig = plt.figure()
    ax = plt.subplot()

    fig.suptitle('Classification results', fontsize = 16)
    index = 1
    for i in results_list:
        X0 = i[1][['lambda_avg']].values
        X1 = i[1][['approx_error']].values
        xx, yy = make_meshgrid(X0, X1)
        ax = fig.add_subplot(rows, cols, index)
        plot_contours(ax, svm , xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        for color in ['S.C. : blue [0]', 'R:C. : red [1]']:
            scatter = ax.scatter(X0, X1, c = i[1][['label']].values, cmap=plt.cm.coolwarm, s=20, label=color, edgecolors='k')
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
        ax.add_artist(legend1)
        # ax.scatter(X0, X1, c = i[1][['label']].values, cmap=plt.cm.coolwarm, s=20, edgecolors='k') 
        ax.set_title(i[0])
        ax.set_xlabel("Lambda Average")
        ax.set_ylabel("Approx Error")
        ax.set_xticks(())
        ax.set_yticks(())
        ax.legend()
        index += 1 
    fig.tight_layout()
    plt.show()


def main():
    #We are going to test recaptured and then single captured for every single camera, at the and we put all togheter
    #but according to the paper is important to have separated statistics for every camera
    #Note: the training set is general, is not fitted for a particular camera

    #acquisizione modello svm
    svm = load(Base_Path + "/svm/model.joblib")
    #acquisizione path di test
    with open(Base_Path + "/test_set_paths/recaptured_60D_test.txt", 'r') as f:
        recaptured_60D_test = [line.rstrip('\n') for line in f]
    with open(Base_Path + "/test_set_paths/recaptured_600D_test.txt", 'r') as f:
        recaptured_600D_test = [line.rstrip('\n') for line in f]
    with open(Base_Path + "/test_set_paths/recaptured_D70S_test.txt", 'r') as f:
        recaptured_D70S_test = [line.rstrip('\n') for line in f]
    with open(Base_Path + "/test_set_paths/recaptured_D3200_test.txt", 'r') as f:
        recaptured_D3200_test = [line.rstrip('\n') for line in f]
    with open(Base_Path + "/test_set_paths/recaptured_EPM2_test.txt", 'r') as f:
        recaptured_EPM2_test = [line.rstrip('\n') for line in f]
    with open(Base_Path + "/test_set_paths/recaptured_RX100_test.txt", 'r') as f:
        recaptured_RX100_test = [line.rstrip('\n') for line in f]
    with open(Base_Path + "/test_set_paths/recaptured_TZ7_test.txt", 'r') as f:
        recaptured_TZ7_test = [line.rstrip('\n') for line in f]
    with open(Base_Path + "/test_set_paths/recaptured_TZ10_test.txt", 'r') as f:
        recaptured_TZ10_test = [line.rstrip('\n') for line in f]

    with open(Base_Path + "/test_set_paths/single_captured_D40_test.txt", 'r') as f:
        single_captured_D40_test = [line.rstrip('\n') for line in f]
    with open(Base_Path + "/test_set_paths/single_captured_D70S_test.txt", 'r') as f:
        single_captured_D70S_test = [line.rstrip('\n') for line in f]
    with open(Base_Path + "/test_set_paths/single_captured_EOS600D_test.txt", 'r') as f:
        single_captured_EOS600D_test = [line.rstrip('\n') for line in f]
    with open(Base_Path + "/test_set_paths/single_captured_EPM2_test.txt", 'r') as f:
        single_captured_EPM2_test = [line.rstrip('\n') for line in f]
    with open(Base_Path + "/test_set_paths/single_captured_RX100_test.txt", 'r') as f:
        single_captured_RX100_test = [line.rstrip('\n') for line in f]
    with open(Base_Path + "/test_set_paths/single_captured_TZ7_test.txt", 'r') as f:
        single_captured_TZ7_test = [line.rstrip('\n') for line in f]
    with open(Base_Path + "/test_set_paths/single_captured_V550B_test.txt", 'r') as f:
        single_captured_V550B_test = [line.rstrip('\n') for line in f]
    with open(Base_Path + "/test_set_paths/single_captured_V550S_test.txt", 'r') as f:
        single_captured_V550S_test = [line.rstrip('\n') for line in f]
    with open(Base_Path + "/test_set_paths/single_captured_V610_test.txt", 'r') as f:
        single_captured_V610_test = [line.rstrip('\n') for line in f]
        
    #classificazione 

    rec_60D = classificate(svm, recaptured_60D_test, LABEL_RC, "recaptured_60D")
    rec_600D = classificate(svm, recaptured_600D_test, LABEL_RC, "recaptured_600D")
    rec_D70S = classificate(svm, recaptured_D70S_test, LABEL_RC, "recaptured_D70S")
    rec_D3200 = classificate(svm, recaptured_D3200_test, LABEL_RC, "recaptured_D3200")
    rec_EPM2 = classificate(svm, recaptured_EPM2_test, LABEL_RC, "recaptured_EPM2")
    rec_RX100 = classificate(svm, recaptured_RX100_test, LABEL_RC, "recaptured_RX100")
    rec_TZ7 = classificate(svm, recaptured_TZ7_test, LABEL_RC, "recaptured_TZ7")
    rec_TZ10 = classificate(svm, recaptured_TZ10_test, LABEL_RC, "recaptured_TZ10")

    sc_D40 = classificate(svm, single_captured_D40_test, LABEL_SC, "single_captured_D40")
    sc_D70S = classificate(svm, single_captured_D70S_test, LABEL_SC, "single_captured_D70S")
    sc_EOS600D = classificate(svm, single_captured_EOS600D_test, LABEL_SC, "single_captured_EOS600D")
    sc_EPM2 = classificate(svm, single_captured_EPM2_test, LABEL_SC, "single_captured_EPM2")
    sc_RX100 = classificate(svm, single_captured_RX100_test, LABEL_SC, "single_captured_RX100")
    sc_TZ7 = classificate(svm, single_captured_TZ7_test, LABEL_SC, "single_captured_TZ7")
    sc_V550B = classificate(svm, single_captured_V550B_test, LABEL_SC, "single_captured_V550B")
    sc_V550S = classificate(svm, single_captured_V550S_test, LABEL_SC, "single_captured_V550S")
    sc_V610 = classificate(svm, single_captured_V610_test, LABEL_SC, "single_captured_V610")

    results_list = [rec_60D, rec_600D, rec_D70S, rec_D3200, rec_EPM2, rec_RX100, rec_TZ7, rec_TZ10, sc_D40, sc_D70S, sc_EOS600D, sc_EPM2, sc_RX100, sc_TZ7, sc_V550B, sc_V550S, sc_V610]
    rec_results_list = [rec_60D, rec_600D, rec_D70S, rec_D3200, rec_EPM2, rec_RX100, rec_TZ7, rec_TZ10]
    sc_results_list = [sc_D40, sc_D70S, sc_EOS600D, sc_EPM2, sc_RX100, sc_TZ7, sc_V550B, sc_V550S, sc_V610]
    
    report_results(results_list)
    plot_results(rec_results_list, svm)
    plot_results(sc_results_list, svm)

    return 1


if __name__ == '__main__':
	output = main()