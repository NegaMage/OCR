#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:28:36 2019
@author: adithya+NegaMage+Vignesh+Rahul
"""

# Import the modules
from __future__ import print_function
import cv2
from sklearn.externals import joblib
import numpy as np
from PIL import Image

# Load the classifier
clf = joblib.load("digits_cls.pkl")

#read
im1 = cv2.imread("photo_1.jpg")

# grayscaling and gaussian filtering
im_gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im_gray2 = cv2.GaussianBlur(im_gray1, (5, 5), 0)

# Thresholding
ret, im_th = cv2.threshold(im_gray2, 90, 255, cv2.THRESH_BINARY_INV)

#cropping the image using histograms
k=0
height=np.size(im_th, 0)
width=np.size(im_th, 1)

#We have the dimensions of height and width. 
#We can find what fraction of each row is filled with white pixels.
#and then we can choose those rows where the fraction is correspondingly accurate.

plottedx=[]
#plottedx is the list that will contain what fraction of each row is filled with white pixels. 
#Remember that this is for the inverted image, not for the original image.

for x in range(height):
    k=0
    for y in range(width):
        k+=im_th[x][y]
    
    k/=height
    plottedx.append(k)

plottedy=[]
#plottedy is the equivalent in columns

for x in range(width):
    k=0
    for y in range(height):
        k+=im_th[y][x]
    
    k/=width
    plottedy.append(k)
    
#Now we choose the part of plottedx and plottedy that  are greater than 1 in the final testing image.
    
listx=[]
k=0
for x in range(height):
    
    if k==0 and plottedx[x]>1:
        listx.append(x)
        k=1
        continue
    elif k==1 and plottedx[x]==0.0:
        listx.append(x-1)
        k=0

listy=[]
k=0
for x in range(width):
    
    if k==0 and plottedy[x]>1:
        listy.append(x)
        k=1
        continue
    elif k==1 and plottedy[x]==0.0:
        listy.append(x-1)
        k=0

#Number of objects
no_of_objects=int(len(listy)/2)
print(no_of_objects)


idx = 0

#image process and recognise for each detected object
for i in range(0, no_of_objects):
    #coordinates
    print(listx[0], listy[2*i], listx[1], listy[2*i+1])
    #cropping
    roi=im_th[listx[0]:listx[1],listy[2*i]:listy[2*i+1]]
    
    #padding for top and bottom cropping
    BLACK = [0, 0, 0]
    roi= cv2.copyMakeBorder(roi,1,1,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    
    #same previous method on repeat
    height1 = np.size(roi, 0)
    width1 = np.size(roi, 1)
    plottedx1 = []
    for x in range(height1):
        k=0
        for y in range(width1):
            k+=roi[x][y]
        k/=height1
        plottedx1.append(k)
        
    listx1=[]
    k=0
    for x in range(height1):
        if k==0 and plottedx1[x]>1:
            listx1.append(x)
            k=1
            continue
        elif k==1 and plottedx1[x]==0.0:
            listx1.append(x-1)
            k=0
    
    #final crop
    roi=roi[listx1[0]:listx1[1],:]
    
    #calculating width and height
    w = listy[2*i+1] - listy[2*i]
    h = listx1[1] - listx1[0]
    
    #creating blank image of max dim    
    blank_image = np.zeros((max(w, h), max (w, h), 3), np.uint8)
    
    #for PIL
    cv2.imwrite('D:\\NITK\\Python\\crop\\temp\\' + 'roi.jpg', roi)
    cv2.imwrite('D:\\NITK\\Python\\crop\\temp\\' + 'blank.jpg', blank_image)
    
    #PIL Lazy Operation
    img1 = Image.open('D:\\NITK\\Python\\crop\\temp\\roi.jpg')
    img2 = Image.open('D:\\NITK\\Python\\crop\\temp\\blank.jpg')
    
    #pasting ROIs into blank for retaining aspect ratio
    img2.paste(img1, (int((max(w, h) - w) / 2), int((max(w, h) - h) / 2)))
    
    #coverting object into array
    img3 = np.asarray(img2)
    
    #resizing to 20*20
    img4 = cv2.resize(img3, (20, 20), interpolation = cv2.INTER_AREA)
    
    #padding with 4 pixels each side to make 28*28
    BLACK = [0, 0, 0]
    img4 = cv2.copyMakeBorder(img4,4,4,4,4,cv2.BORDER_CONSTANT,value=BLACK)
    
    #for next process
    cv2.imwrite('D:\\NITK\\Python\\crop\\temp\\' + 'roi.jpg', img4)
    
    #loading in the form of grayscalled flattened 2D image
    fim = Image.open("D:\\NITK\\Python\\crop\\temp\\roi.jpg").convert('LA')
    fim = np.asarray(fim)[:,:,0]
    
    fim.reshape((1, 784))
    
    #print final image to be recognised
    print(fim)
    
    #making it single row
    fim = np.reshape(fim, (1,np.product(fim.shape)))
    
    # grab the image and classify it
    image = fim[[idx]]
    prediction = clf.predict(image)[0]
    
    #image = image.reshape((8, 8)).astype("uint8")
    #image = exposure.rescale_intensity(image, out_range=(0, 255))
    #image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

    #Show the prediction
    print("I think that digit is: {}".format(prediction))
    cv2.imshow("Image", img4)
    cv2.imwrite('D:\\NITK\\Python\\crop\\' + str(idx) + '.jpg', fim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()