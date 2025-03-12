import numpy as np
import pandas as pd
from pandas import read_csv
import math
import matplotlib.pyplot as plt
import glob
import sys
import os

directory = "./debug_plots/"
if not os.path.exists(directory):
    os.makedirs(directory)

sensor_geom = '50x12P5_0fb'
output_file = open(directory+"output"+sensor_geom+".txt", "w")

for thresh_iter in [0.1,0.15,0.2,0.3,0.4,0.5]:
    threshold = thresh_iter
    print("Producing datasets for thresh = ",thresh_iter)
    # Global variables
    noise_threshold = 600
    train_dataset_name = 'dataset_9s' # for train datasets
    test_dataset_name = 'dataset_8s' # for location of test (physical pT) datasets
    # dataset_savedir = 'dataset_9s'
    dataset_savedir = f'dataset_9s_{noise_threshold}NoiseThresh' # for save loc of final datasets

    output_file.write("=========="+"\n")
    output_file.write("Train dataset prod. beginning: "+str(threshold)+" thresh"+"\n")
    dirtrain = '/location/of/parquets/smartpixels/'+train_dataset_name+'/'+train_dataset_name+'_'+sensor_geom+'_parquets/unflipped/'
    # /location/of/parquets/smartpixels/dataset_2s/dataset_2s_50x12P5_parquets/unflipped
    trainlabels = []
    trainrecons = []

    iter=0
    suffix = 16400
    for filepath in glob.iglob(dirtrain+'labels*.parquet'):
        iter+=3
    output_file.write(str(iter)+" files present in directory."+"\n")
    for i in range(int(iter/3)):
            trainlabels.append(pd.read_parquet(dirtrain+'labels_d'+str(suffix+i+1)+'.parquet'))
            trainrecons.append(pd.read_parquet(dirtrain+'recon2D_d'+str(suffix+i+1)+'.parquet'))
    trainlabels_csv = pd.concat(trainlabels, ignore_index=True)
    trainrecons_csv = pd.concat(trainrecons, ignore_index=True)

    iter_0, iter_1, iter_2 = 0, 0, 0
    iter_rem = 0
    for iter, row in trainlabels_csv.iterrows():
        if(abs(row['pt'])>threshold):
            iter_0+=1
        elif(-1*threshold<=row['pt']<0):
            iter_1+=1
        elif(0<row['pt']<=threshold):
            iter_2+=1
        else:
            iter_rem+=1
    output_file.write("iter_0: "+str(iter_0)+"\n")
    output_file.write("iter_1: "+str(iter_1)+"\n")
    output_file.write("iter_2: "+str(iter_2)+"\n")
    output_file.write("iter_rem: "+str(iter_rem)+"\n")

    plt.hist(trainlabels_csv['pt'], bins=100)
    plt.title('pT of all events')
    plt.savefig(directory+"train_pt_all_"+sensor_geom+".png")
    plt.close()

    plt.hist(trainlabels_csv[abs(trainlabels_csv['pt'])>threshold]['pt'], bins=100)
    plt.title('pT of Class 0 events')
    plt.savefig(directory+"train_pt_cls0_"+sensor_geom+".png")
    plt.close()

    plt.hist(trainlabels_csv[(0<=trainlabels_csv['pt'])&(trainlabels_csv['pt']<=threshold)]['pt'], bins=50)
    plt.hist(trainlabels_csv[(-1*threshold<=trainlabels_csv['pt'])& (trainlabels_csv['pt']<0)]['pt'], bins=50)
    plt.title('pT of Class 1+2 events')
    plt.savefig(directory+"train_pt_cls12_"+sensor_geom+".png")
    plt.close()

    number_of_events = (min(iter_1, iter_2)//1000)*1000
    if(number_of_events*2>iter_0):
        number_of_events = (iter_0//1000)*1000/2
    number_of_events = int(number_of_events)
    output_file.write("Number of events: "+str(number_of_events)+"\n")

    def sumRow(X):
        X = np.where(X < noise_threshold, 0, X)
        sum1 = 0
        sumList = []
        for i in X:
            sum1 = np.sum(i,axis=0)
            sumList.append(sum1)
            b = np.array(sumList)
        return b
    trainlist1, trainlist2 = [], []
    hist_temp=[]
    for (index1, row1), (index2, row2) in zip(trainrecons_csv.iterrows(), trainlabels_csv.iterrows()):
        rowSum = 0.0
        X = row1.values
        X = np.reshape(X,(13,21))
        rowSum = sumRow(X)
        hist_temp.append(np.sum(rowSum>0))
        trainlist1.append(rowSum)
        cls = -1
        if(abs(row2['pt'])>threshold):
            cls=0
        elif(-1*threshold<=row2['pt']<0):
            cls=1
        elif(0<=row2['pt']<=threshold):
            cls=2
        trainlist2.append([row2['y-local'], cls, row2['pt']])

    plt.hist(hist_temp, bins=14,  range=[0, 14], histtype='step', fill=False, density=True)
    plt.savefig(directory+"y_profile_afterThreshold_"+sensor_geom+".png")
    plt.close()

    traindf_all = pd.concat([pd.DataFrame(trainlist1), pd.DataFrame(trainlist2 , columns=['y-local', 'cls', 'pt'])], axis=1)

    totalsize = number_of_events
    random_seed0 = 10#11
    random_seed1 = 13#14
    random_seed2 = 19#20

    traindf_all = traindf_all.sample(frac=1, random_state=random_seed0).reset_index(drop=True)
    # traindf_all.to_csv(dataset_savedir+'/'+'/FullTrainData_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
    traindfcls0 = traindf_all.loc[traindf_all['cls']==0]
    traindfcls1 = traindf_all.loc[traindf_all['cls']==1]
    traindfcls2 = traindf_all.loc[traindf_all['cls']==2]
    output_file.write(str(traindfcls0.shape)+"\n")
    output_file.write(str(traindfcls1.shape)+"\n")
    output_file.write(str(traindfcls2.shape)+"\n")
    traindfcls0 = traindfcls0.iloc[:2*totalsize]
    traindfcls1 = traindfcls1.iloc[:totalsize]
    traindfcls2 = traindfcls2.iloc[:totalsize]

    traincls0 = traindfcls0.sample(frac = 1, random_state=random_seed1)
    traincls1 = traindfcls1.sample(frac = 1, random_state=random_seed1)
    traincls2 = traindfcls2.sample(frac = 1, random_state=random_seed1)
    train = pd.concat([traincls0, traincls1, traincls2], axis=0)

    train = train.sample(frac=1, random_state=random_seed2)

    output_file.write(str(traincls0.shape)+"\n")
    output_file.write(str(traincls1.shape)+"\n")
    output_file.write(str(traincls2.shape)+"\n")
    output_file.write(str(train.shape)+"\n")

    trainlabel = train['cls']
    trainpt = train['pt']
    train = train.drop(['cls', 'pt'], axis=1)

    output_file.write(str(train.shape)+"\n")
    output_file.write(str(trainlabel.shape)+"\n")
    output_file.write(str(trainpt.shape)+"\n")

    train.to_csv(dataset_savedir+'/FullPrecisionInputTrainSet_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
    trainlabel.to_csv(dataset_savedir+'/TrainSetLabel_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
    trainpt.to_csv(dataset_savedir+'/TrainSetPt_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)

    # TEST DATASET PROD
    output_file.write("=========="+"\n")
    output_file.write("Test dataset prod. beginning: "+str(threshold)+" thresh"+"\n")
    dirtest = '/location/of/parquets/smartpixels/'+test_dataset_name+'/'+test_dataset_name+'_'+sensor_geom+'_parquets/unflipped/'
    # /location/of/parquets/smartpixels/dataset_2s/dataset_2s_50x12P5_parquets/unflipped
    testlabels = []
    testrecons = []

    iter=0
    suffix = 16400
    for filepath in glob.iglob(dirtest+'labels*.parquet'):
        iter+=3
    output_file.write(str(iter)+" files present in directory."+"\n")
    for i in range(int(iter/3)):
            testlabels.append(pd.read_parquet(dirtest+'labels_d'+str(suffix+i+1)+'.parquet'))
            testrecons.append(pd.read_parquet(dirtest+'recon2D_d'+str(suffix+i+1)+'.parquet'))
    testlabels_csv = pd.concat(testlabels, ignore_index=True)
    testrecons_csv = pd.concat(testrecons, ignore_index=True)

    iter_0, iter_1, iter_2 = 0, 0, 0
    iter_rem = 0
    for iter, row in testlabels_csv.iterrows():
        if(abs(row['pt'])>threshold):
            iter_0+=1
        elif(-1*threshold<=row['pt']<0):
            iter_1+=1
        elif(0<row['pt']<=threshold):
            iter_2+=1
        else:
            iter_rem+=1
    output_file.write("iter_0: "+str(iter_0)+"\n")
    output_file.write("iter_1: "+str(iter_1)+"\n")
    output_file.write("iter_2: "+str(iter_2)+"\n")
    output_file.write("iter_rem: "+str(iter_rem)+"\n")

    plt.hist(testlabels_csv['pt'], bins=100)
    plt.title('pT of all events')
    plt.savefig(directory+"test_pt_all_"+sensor_geom+".png")
    plt.close()

    plt.hist(testlabels_csv[abs(testlabels_csv['pt'])>threshold]['pt'], bins=100)
    plt.title('pT of Class 0 events')
    plt.savefig(directory+"test_pt_cls0_"+sensor_geom+".png")
    plt.close()

    plt.hist(testlabels_csv[(0<=testlabels_csv['pt'])&(testlabels_csv['pt']<=threshold)]['pt'], bins=50)
    plt.hist(testlabels_csv[(-1*threshold<=testlabels_csv['pt'])& (testlabels_csv['pt']<0)]['pt'], bins=50)
    plt.title('pT of Class 1+2 events')
    plt.savefig(directory+"test_pt_cls12_"+sensor_geom+".png")
    plt.close()

    number_of_events = (min(iter_1, iter_2)//1000)*1000
    if(number_of_events*2>iter_0):
        number_of_events = (iter_0//1000)*1000/2
    number_of_events = int(number_of_events)
    output_file.write("Number of events: "+str(number_of_events)+"\n")

    testlist1, testlist2 = [], []

    for (index1, row1), (index2, row2) in zip(testrecons_csv.iterrows(), testlabels_csv.iterrows()):
        rowSum = 0.0
        X = row1.values
        X = np.reshape(X,(13,21))
        rowSum = sumRow(X)
        testlist1.append(rowSum)
        cls = -1
        if(abs(row2['pt'])>threshold):
            cls=0
        elif(-1*threshold<=row2['pt']<0):
            cls=1
        elif(0<=row2['pt']<=threshold):
            cls=2
        testlist2.append([row2['y-local'], cls, row2['pt']])
    testdf_all = pd.concat([pd.DataFrame(testlist1), pd.DataFrame(testlist2 , columns=['y-local', 'cls', 'pt'])], axis=1)

    totalsize = number_of_events#227000
    random_seed0 = 10#11
    random_seed1 = 13#14
    random_seed2 = 19#20

    testdf_all = testdf_all.sample(frac=1, random_state=random_seed0).reset_index(drop=True)
    testdf_all.to_csv(dataset_savedir+'/'+'/FullTestData_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
    # testdfcls0 = testdf_all.loc[testdf_all['cls']==0]
    # testdfcls1 = testdf_all.loc[testdf_all['cls']==1]
    # testdfcls2 = testdf_all.loc[testdf_all['cls']==2]
    # output_file.write(testdfcls0.shape+"\n")
    # output_file.write(testdfcls1.shape+"\n")
    # output_file.write(testdfcls2.shape+"\n")
    # output_file.write(testdfcls2.head()+"\n")
    # testdfcls0 = testdfcls0.iloc[:2*totalsize]
    # testdfcls1 = testdfcls1.iloc[:totalsize]
    # testdfcls2 = testdfcls2.iloc[:totalsize]
    # output_file.write(testdfcls2.head()+"\n")

    # testcls0 = testdfcls0.sample(frac = 1, random_state=random_seed1)
    # testcls1 = testdfcls1.sample(frac = 1, random_state=random_seed1)
    # testcls2 = testdfcls2.sample(frac = 1, random_state=random_seed1)
    # test = pd.concat([testcls0, testcls1, testcls2], axis=0)

    # test = test.sample(frac=1, random_state=random_seed2)
    test=testdf_all
    # output_file.write(testcls0.shape+"\n")
    # output_file.write(testcls1.shape+"\n")
    # output_file.write(testcls2.shape+"\n")
    output_file.write(str(test.shape)+"\n")

    testlabel = test['cls']
    testpt = test['pt']
    test = test.drop(['cls', 'pt'], axis=1)

    output_file.write(str(test.shape)+"\n")
    output_file.write(str(testlabel.shape)+"\n")
    output_file.write(str(testpt.shape)+"\n")

    test.to_csv(dataset_savedir+'/FullPrecisionInputTestSet_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
    testlabel.to_csv(dataset_savedir+'/TestSetLabel_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
    testpt.to_csv(dataset_savedir+'/TestSetPt_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)

output_file.close()