import numpy as np
import scipy.signal


def ROI(img):
  '''
  This is to get the upper portion of one normalized image  
  -img: norml image 
  '''
  img_new = img[0:48,]
  return img_new


def M1(x,y,sgmY):
  '''
  -(x,y): point from image
  -sgmY: sigmaY 
  '''
  f = 1/sgmY 
  m = np.cos(2*np.pi*f*np.sqrt(x**2 + y**2))
  return m 

def G1(x,y,sgmX,sgmY):
  '''
  -(x,y):point from image 
  -sgmX: sigmaX 
  -sgmY: sigmaY 
  '''
  g = (1/(2*np.pi*sgmX*sgmY) * np.exp(-0.5 * (x ** 2 / sgmX ** 2 + y ** 2 / sgmY ** 2)) * M1(x, y, sgmY))
  return g

def kernal(sgmX,sgmY):
  '''
  This is to get the Gabor Filter matrix
  -sgmX: sigmaX
  -sgmY: sigmaY 
  '''

  kernal = np.zeros((9,9))
  for i in range(9):
    for j in range(9):
      kernal[i,j] = G1((-4+i),(-4+j),sgmX,sgmY)
  return kernal

def filtered_image(img,sgmX,sgmY):
  '''
  This is to get the filterd image
  -img: normalized image 
  -sgmX:sigmaX
  -sgmY:sigmaY 
  '''
  roi = ROI(img)
  ker = kernal(sgmX,sgmY)
  img_filter = scipy.signal.convolve2d(roi,ker,mode='same')
  return img_filter


def FeatureExtraction(img):
  '''
  To get the feature vector 

  -img: normalized image 
  '''
  # channels values from LiMa essay 
  sigmaX1 = 3
  sigmaY1 = 1.5
  sigmaX2 = 4.5
  sigmaY2 = 1.5
  #get two filter images 
  img1 = filtered_image(img, sigmaX1,sigmaY1)
  img2 = filtered_image(img, sigmaX2,sigmaY2)
  #get feature vector 
  r = int(img1.shape[0]/8 )             #r =48/8 = 6
  c = int(img2.shape[1]/8 )             #c =512/8 = 64
  vector = np.zeros(r*c*2*2)            #len is 6*64*2*2=1536
  #for two channels
  for i in range(2):
    image = [img1,img2][i]
    for row in range(r):
      for col in range(c):
        mean = np.mean( np.abs(image[row*8: (row+1) * 8,col*8: (col +1) * 8] ))
        sd = np.sum(abs(image[row*8: (row+1) * 8,col*8: (col +1) * 8] - mean))/ (8*8)
        vector[i*768 + 2*row*c + 2*col] = mean
        vector[i*768 + 2*row*c + 2*col + 1] = sd
  return vector

