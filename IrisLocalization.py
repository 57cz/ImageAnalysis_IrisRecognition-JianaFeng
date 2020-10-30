import numpy as np
import cv2
import pandas as pd



def IrisLocalization(img_gray):
    '''
    -img_gray: the gray image for Irir
    '''

    #step 1
    #Estimate the center of the pupil by sum by vertical and horizontal
    #To imporve the finding result,working on a subimage
    sub_img_1= img_gray[60:240,100:220]
    yp = np.argmin(sub_img_1.sum(axis=1))+60
    xp = np.argmin(sub_img_1.sum(axis=0))+100

    #Step 2
    #using estimated center of pupil to find the more accuracte position of pupil center
    sub_img_2 = img_gray[yp-60:yp+60,xp-60:xp+60]
    _,sub_img_2_binary = cv2.threshold(sub_img_2,64,65,cv2.THRESH_BINARY)
    sub_img_2_binary = np.where(sub_img_2_binary>0,1,0)

    #calculate the radius of the pupil
    s_circle = 120*120-np.sum(sub_img_2_binary)
    radius = np.sqrt(s_circle/np.pi)

    yp2 = np.argmin(sub_img_2_binary.sum(axis=1))
    xp2 = np.argmin(sub_img_2_binary.sum(axis=0))

    xp3 = xp2+xp-60
    yp3 = yp2+yp-60



    #Step 3
    #Working on sub image to imporve the result
    #a region with 120*120 to find pupil
    #a region with 230*270 to find iris
    sub_img_pupil = img_gray[(yp3-60):min(279, yp3+60),(xp3-60):min(319,xp3+60)]
    sub_img_iris = img_gray[np.arange(yp3-120, min(279, yp3+110)),:][:,np.arange(xp3-135,min(319,xp3+135))]
    img_output = img_gray.copy()
    #find the pupil circle whose center is most close to the one from step2
    for i in range(1,5):
        pupil_circles = cv2.HoughCircles(sub_img_pupil,cv2.HOUGH_GRADIENT,1,250,param1=50,param2=10,
                                   minRadius=int(radius-i),maxRadius=int(radius+i))
        if type(pupil_circles) != type(None):
            break
        else:
            pass
    pupil_circles = np.uint16(np.around(pupil_circles))
    for i in pupil_circles[0,:]:
        cv2.circle(img_output,(i[0]+xp3-60,i[1]+yp3-60),i[2],(0,255,0),2)
        cv2.circle(img_output,(i[0]+xp3-60,i[1]+yp3-60),2,(0,0,255),3)
        pupil_circles = [i[0] + xp3 - 60,i[1] + yp3 -60 ,i[2]]
        
    iris_circles = cv2.HoughCircles(sub_img_iris,cv2.HOUGH_GRADIENT,1,250,param1=30,param2=10,
                                  minRadius=98,maxRadius=118)
  
    iris_circles = np.uint16(np.around(iris_circles))
    #print(iris_circles)
    for i in iris_circles[0,:]:
        # draw the outer circle
        cv2.circle(img_output,(i[0]+xp3-135,i[1]+yp3-120),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img_output,(i[0]+xp3-135,i[1]+xp3-120),i[2],(0,0,255),3) 
        iris_circles =  [i[0]+xp3-135 ,i[1]+yp3-120 ,i[2]]
  
    return (pupil_circles,iris_circles)

  