import numpy as np
import cv2
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors.nearest_centroid import NearestCentroid
import matplotlib.pyplot as plt



def DimensionReduction(trainX,trainY,testX,testY,n=107):
  '''
  This is get the reduced feature vector on both test data and train data 
  -trainX:training feature
  -trainY:training label
  -testX:testing feature
  -testY:testing label
  -n:control the dimension of the reduced feature vector
  '''
  lda = LDA(n_components = n)
  lda.fit(trainX,trainY)
  trainX_ld = lda.transform(trainX)
  testX_ld = lda.transform(testX)
  return trainX_ld,testX_ld



def IrisMatch(trainX,trainY,testX,testY,d):
  '''
  -trainX:training feature
  -trainY:training label
  -testX:testing feature
  -testY:testing label
  -d: it is the metric used for nearest neingbor classifer 
  '''

  if d==1:
    metric = 'l1'
  elif d==2:
    metric = 'l2'
  elif d==3:
    metric = 'cosine'
  else:
    raise Exception('d should in [1,2,3]. The value of d was: {}'.format(d))
  clf = NearestCentroid(metric=metric)
  clf.fit(trainX,trainY)
  return clf.score(testX,testY)

def GetCRR(trainX_ld,trainY,testX_ld,testY):
  '''
  This is to compute the CRR
  -trainX_ld:training feature
  -trainY:training label
  -testX_ld:testing feature
  -testY:testing label 
  '''
  BestCRR = []
  for i in range(1,4):
    BestCRR.append(IrisMatch(trainX_ld,trainY,testX_ld,testY,d=i))
  return BestCRR

def GetDimDict(trainX,trainY,testX,testY):
  '''
  This is to get the CRR with diferent dims
  -trainX:training feature
  -trainY:training label
  -testX:testing feature
  -testY:testing label 
  '''
  reduced_dim = [30,40,50,60,70,80,90,100,107]
  DimDict = []
  for i in reduced_dim:
    trainX_ld,testX_ld = DimensionReduction(trainX,trainY,testX,testY,n=i)
    DimDict.append(IrisMatch(trainX_ld,trainY,testX_ld,testY,d=3)*100)
  return  DimDict




	



  
