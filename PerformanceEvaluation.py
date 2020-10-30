from IrisMatching import *


from scipy import spatial
import tabulate
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


def GetCRR_table(CRR_original, CRR_reduced):
  '''
  This is to get the CRR table for Different Similarity Measures
  -trainX_ld: trainX with reduced dim 
  -testX_ld: testX with reduced dim 
  '''
 # BestCRR = GetCRR(trainX_ld,trainY,testX_ld,testY)
  print("     Recognition Result Using Different Similarity Measures")
  print ("           Correct recognition rate (%)")
  print(tabulate.tabulate([['L1 distance measure',CRR_original[0]*100,CRR_reduced[0]*100 ],['L2 distance measure', CRR_original[1]*100,CRR_reduced[1]*100 ],
	 	['Cosine similarity measure', CRR_original[2]*100,CRR_reduced[2]*100 ]],
	  	headers=['Similartiy measure','Original feature set(%)', "Reduced feature set(%)"]))


def GetCRR_dim_fig(DimDict):
  '''
  This is to get the fig on CRR for different dims 
  '''
  reduced_dim = [30,40,50,60,70,80,90,100,107]
 # reduced_dim,DimDict = GetDimDict(trainX,trainY,testX,testY)
  plt.figure()
  plt.plot(reduced_dim,DimDict)
  plt.xlabel('Dimensionality of the feature vector')
  plt.ylabel('Correct recognition rate (%)')
  plt.title('CRR with different dimensionality')
  plt.savefig('figure_1.png')
  plt.show()


def ROC_Metrics(trainY,trainX_ld,testY,testX_ld):
  '''
  This is to get cos_sim matric with shape of (N,M), where N is the len of test data 
  M is the len of train data 

  -trainX_ld: trainX with reduced dim 
  -testX_ld: testX with reduced dim 
  '''
  N = len(testY)
  M = len(trainY)
  cos_sim = np.zeros((N,M))
  for i in range(len(testY)):
    for j in range(len(trainY)):
      tmp = 1- spatial.distance.cosine(trainX_ld[j,:], testX_ld[i,:])
      cos_sim[i,j] = tmp
  return cos_sim


def roc(cos_sim,trainY,testY,T):
  '''
  This is to get the FMR and FNMR for test data 
  -cos_sim: parameter from ROC_Metrics function 
  -T: threshold 
  '''
  (N,M) = cos_sim.shape
  fp = 0
  fn = 0
  tp = 0
  tn = 0 
  for i in range(N):
    for j in range(M):
      if cos_sim[i,j] < T:
        if trainY[j] == testY[i]:
          fn += 1        
        else:
          tn += 1 
      elif cos_sim[i,j] >= T:  
        if trainY[j] != testY[i]:
          fp += 1          
        else:
          tp += 1 
  FMR=fp/(fp+tn)
  FNMR= fn/(fn+tp)
  return FMR,FNMR

def getFMR_FNMR_table(cos_sim,trainY,testY):
  '''
  This is to get FMR(false match rate) and FNMR(false non-match rate)
  '''

  T = [0.446,0.472,0.502]
  FMR_list = []
  FNMR_list = []
  [FMR_list.append(roc(cos_sim,trainY,testY,t)[0]) for t in T]
  [FNMR_list.append(roc(cos_sim,trainY,testY,t)[1]) for t in T] 
  print("False Match and False Nonmatch Rates with Different Threshold Values")
  print(tabulate.tabulate([[T[0], FMR_list[0],FNMR_list[0]], 
                    [T[1], FMR_list[1],FNMR_list[1]],
                    [T[2], FMR_list[2],FNMR_list[2]]],
                    headers=['Threshold', 'False match rate',"False non-match rate"]))


def getROC_fig(cos_sim,trainY,testY):
  '''
  This is to get the roc figure 
  '''
  T_new = np.linspace(0.4,0.5,10)
  FMR_list_new = []
  FNMR_list_new = []
  [FMR_list_new.append(roc(cos_sim,trainY,testY,t)[0]) for t in T_new]
  [FNMR_list_new.append(roc(cos_sim,trainY,testY,t)[1]) for t in T_new]
  plt.figure()
  plt.plot(FMR_list_new,FNMR_list_new)
  plt.xlabel('False Match Rate')
  plt.ylabel('False Non_match Rate')
  plt.title('ROC Curve')
  plt.savefig('figure_2.png')
  plt.show()












