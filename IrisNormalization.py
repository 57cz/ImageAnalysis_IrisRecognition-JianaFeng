import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
 

def getxy(X,Y,pupil_circle,iris_circle): 
    '''
    -(X,Y): points from rectangle image 
    -pupil_circle: center point and raidus from pupil 
    -iris_circle: center point and raidus from outter circle 
    '''  

    #dimension of rectangle on iris image 
    (M, N) = (64, 512)
    theta = 2 * np.pi * X / N 
    # get the cneters and radius from pupil and iris for outside circle 
    (xp,yp,rp) = pupil_circle
    (xi,yi,ri) = iris_circle
    #if xi = xp, compute the average iris radius 
    if xi==xp:
      d = np.sqrt((xp-xi)**2 + (yp-yi)**2)
      a = np.pi/2
      R = (2*d*np.cos(a) + np.sqrt( abs( (2*d*np.cos(a))**2 - 4*(d*d - ri*ri) ))) / 2 
    #compute the average iris raidus for outside circle 
    else: 
      d = np.sqrt((xp-xi)**2 + (yp-yi)**2)
      a = np.arctan((yi-yp)/(xi-xp))
      R = (2*d*np.cos(a) + np.sqrt( abs( (2*d*np.cos(a))**2 - 4*(d*d - ri*ri) ))) / 2

    #(xIn yIn) is th center of inner circle 
    xIn = xp + rp * np.cos(theta)
    yIn = yp + rp * np.sin(theta)
    #(xOu yOu) is th center of outter circle  
    xOu = xp + R * np.cos(theta)
    yOu = yp + R * np.sin(theta)
     
    #Using the formula to get the corresponing point  
    x = int(xIn + (xOu - xIn) * Y / M)
    y = int(yIn + (yOu - yIn) * Y / M)
      
    x = min(319,x) or max(0,x)
    y = min(279,y) or max(0,y)
    
    return(x, y)

def IrisNormalization(img,pupil_circle,iris_circle):
    '''
    -img: gray image 
    -pupil_circle: center point and raidus from pupil 
    -iris_circle: center point and raidus from outter circle 
    '''

    #create the matrix with shape of (64,512)
    img_normal = np.zeros((64,512))

    #Using the loop to output the normal image 
    for Y in np.arange(64):
        for X in np.arange(512):
            (x, y) = getxy(X, Y,pupil_circle,iris_circle)
            img_normal[Y, 511- X] = img[y, x]
    return img_normal


def rotation(img_normal,degree):
  '''
  -img_normal: normal image 
  -degree: rotation degree 
  '''

  #get the rotation image 
  pixel = int(degree/360*512)
  img_rot = np.hstack((img_normal[:,pixel:],img_normal[:,:pixel]))
  return img_rot





