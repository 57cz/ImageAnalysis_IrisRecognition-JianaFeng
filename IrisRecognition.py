from IrisLocalization import IrisLocalization
from IrisNormalization import *
from ImageEnhancement import *
from IrisMatching import *
from IrisNormalization import *
from FeatureExtraction import * 
from PerformanceEvaluation import *

import numpy as np
import cv2
import pickle
import os 



def DataProcess(img,training=True):
  '''
  This is the implement all the preprocessing part
  -img:original image of iris
  -training:control whether rotation is needed
  '''
  img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
  pupil_circle,iris_circle =IrisLocalization(img)
  img_normal = IrisNormalization(img,pupil_circle,iris_circle)
  if training:
    #After experiment, this degree could imporve our performance
    degree = [-4,-3,-2,-1,0,1,2,3,4]
    output = np.zeros((9,1536))
    for i in range(len(degree)):
      img_rotate = rotation(img_normal,degree[i])
      #Since we found IrisEnhancement would decrease the CRR,we do not use this function actually
      img_enhance = IrisEnhancement(img_rotate)
      feature_extracted = FeatureExtraction(img_enhance)
      output[i] = feature_extracted


  else:
    img_enhance = IrisEnhancement(img_normal)
    feature_extracted = FeatureExtraction(img_enhance)
    output = feature_extracted

  return output


def DataLoader():
  '''
  This is to get the preprocessed dataset for training and testing
  '''
  path = 'datasets/CASIA Iris Image Database (version 1.0)'
  folder = [1,2]
  trainX = np.zeros((3*9*108,1536))
  testX = np.zeros((4*108,1536))
  trainY = np.repeat(np.arange(1,109),3*9)
  testY = np.repeat(np.arange(1,109),4)
  for sample in range(1,109):
    sample_str = "%03d"%sample
    print('Processing {}/108'.format(sample_str))
    for i in folder:
      if i ==1:
        for j in range(1,4):
          img_filePath = os.path.join(path,sample_str,str(i),sample_str+'_'+str(i)+'_'+str(j)+'.bmp')
          img = cv2.imread(img_filePath)
          feature_vector = DataProcess(img,training=True)
          trainX[(sample-1)*27+(j-1)*9:(sample-1)*27+j*9] = feature_vector
      if i ==2:
        for j in range(1,5):
          img_filePath = os.path.join(path,sample_str,str(i),sample_str+'_'+str(i)+'_'+str(j)+'.bmp')
          img = cv2.imread(img_filePath)
          feature_vector = DataProcess(img,training=False)
          testX[(sample-1)*4+(j-1)] = feature_vector
  return (trainX,trainY),(testX,testY)

def data2pickle(trainX,trainY,testX,testY):
  '''
  This is to save the training and testing data since it would take long time to preprocess data  
  '''
  pickle_file = 'ProcessedIris.pickle'
  if not os.path.isfile(pickle_file): 
      print('Saving data to pickle file...')
      try:
	       with open('ProcessedIris.pickle', 'wb') as pfile:
	            pickle.dump(
	                {
	                    'train_dataset': trainX,
	                    'train_labels': trainY,
	                    'test_dataset': testX,
	                    'test_labels': testY,
	                },
	                pfile, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
	        print('Unable to save data to', pickle_file, ':', e)
	        raise
  print('Data cached in pickle file.')



(trainX,trainY),(testX,testY) = DataLoader()
data2pickle(trainX,trainY,testX,testY)

#Using the original feature set to compute CRR
CRR_original = GetCRR(trainX,trainY,testX,testY)
#Using the reduced feature set to compute CRR
(trainX_ld,testX_ld) = DimensionReduction(trainX,trainY,testX,testY,n=107)
CRR_reduced = GetCRR(trainX_ld,trainY,testX_ld,testY)

GetCRR_table(CRR_original, CRR_reduced)
DimDict = GetDimDict(trainX,trainY,testX,testY)
GetCRR_dim_fig(DimDict)

cos_sim = ROC_Metrics(trainY,trainX_ld,testY,testX_ld)
getFMR_FNMR_table(cos_sim,trainY,testY)
getROC_fig(cos_sim,trainY,testY)



