import numpy as np
import cv2
import pandas as pd


#input is a 64*512 array
#The function output the background of 16*16
def ImgBackground(img):
  '''
  -img: normalized image whch is a 64*512 array
  '''
  nrow,ncol = int(img.shape[0]/16),int(img.shape[1]/16)
  img_output = np.zeros((nrow,ncol))
  for i in range(ncol):
    for j in range(nrow):
      #get the mean from 16*16 region
      img_output[j,i] = np.mean(img[j*16:(j+1)*16,i*16:(i+1)*16])
  #do interpolation to make the size of background the same as normalized image
  img_output = cv2.resize(img_output,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_CUBIC)
  return img_output


def SubtractBackground(img,background):
  '''
  -img:normalized image
  -background:background for normalized image
  '''
  output = img-background
  return output



def HistogramEqualization(img):
  '''
  -img: normalized image whch is a 64*512 array
  '''
  img_output = np.zeros(img.shape)
  #img = np.array(img,dtype=np.uint8)
  nrow,ncol = int(img.shape[0]/32),int(img.shape[1]/32)
  for i in range(ncol):
    for j in range(nrow):
      img_output[j*32:(j+1)*32,i*32:(i+1)*32] = cv2.equalizeHist(np.uint8(img[j*32:(j+1)*32,i*32:(i+1)*32]))
  return img_output


#Since we found that implementing algorithm without Image Enhancement could ahcieve higher CRR
#We do not use this part
def IrisEnhancement(img):
  '''
  -img: normalized image whch is a 64*512 array
  '''
  #bg = ImgBackground(img)
  #img = SubtractBackground(img,bg)
  #img = HistogramEqualization(img)
  return img