import copy
import pickle

import cv2
from sklearn import svm

import matplotlib.pyplot as plt
import numpy as np


def getFeatures(im,backGround='Black',isBinary=False):
        
#     # If input image is not a binary image we will threshold it here to separate back ground and foreground
#     if backGround=='White':Method = cv2.THRESH_BINARY_INV
#     else:Method = cv2.THRESH_BINARY
#             
#     if not isBinary:        
#         _,im = cv2.threshold(im,225,255,Method)
        
#     kernel = np.ones((7,7), np.uint8)
#     im = cv2.dilate(im,kernel,iterations=1) 
    #cv2.imshow("",im),cv2.waitKey()   
    
    imshape = np.shape(im)    
    
    #We will use shape information to determine whether the image is a color image or a binary one.
    if len(imshape)>2:
        
        #If image is a color image it will have 3 Channels so we will convert it into single channel image.
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
    im_ = im.copy()
    im_ = im_.astype("float32")/255.0
    
    im = cv2.GaussianBlur(im,(5,5),0)                            
    edgeIm = cv2.Canny(im,30,150)    
    
    _, contours,_= cv2.findContours(edgeIm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #cv2.imshow("",cntIm),cv2.waitKey()
    idx =0 
    features = []
    sx = 28;sy =28
    
    for cnt in contours:                
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h>25:        
            idx+=1
            roi=im_[y:y+h,x:x+w]
            roi = cv2.resize(roi,(sx,sy))
            roi_array = np.reshape(roi, (sx*sy,))
            features.append(roi_array)
            #cv2.imshow(str(idx),roi),cv2.waitKey()
            
    return np.array(features),contours
    

def CreateDataset():
    
    path2read = 'E:/GoogleDrive/TechDocuments/Webinars/SVM/Alphabets/'
    images2read = ['a.png','b.png','c.png','d.png','e.png','f.png','g.png','h.png',
                   'i.png','j.png','k.png','l.png','m.png','n.png','o.png','p.png',
                   'q.png','r.png','s.png','t.png','u.png','v.png','w.png','x.png','y.png','z.png']
    
    
    labels = []
    npoints = 784    
    data = np.array(np.zeros((1,npoints),dtype='uint16'));
    class_var = 0
    for image_name in images2read: 
        im = cv2.imread(path2read+image_name)
        features,_ = getFeatures(im, backGround='White')
        data = np.append(data,features,axis=0)            
        lab = [class_var for i in range(len(features))]
        class_var+=1            
        labels = np.append(labels,lab)
    
    data = np.delete(data, (0), axis=0)
    dataMean = np.mean(data, axis=0)
    #data = data-dataMean
    return np.array(data),np.array(labels),np.array(dataMean)

def main():
    
    sign,labels,mean = CreateDataset()
    model = svm.LinearSVC()
    model.fit(sign, labels)    
    predictions = model.predict(sign)    
        
    accuracy = getAccuracy(labels, predictions)
    print("Accuracy: %.2f"%(accuracy))
    model_path = 'E:/GoogleDrive/TechDocuments/Webinars/SVM/alphabet.sav'
    np.save('E:/GoogleDrive/TechDocuments/Webinars/SVM/dMean.npy',mean)
    pickle.dump(model,open(model_path,'wb'))

def test():
    
    #testPath = 'D:/testim/Ovaries/Code/datachallenge_cods2016/shapes.png'
    testPath = 'E:/GoogleDrive/TechDocuments/Webinars/SVM/ocrtest.jpg' 
    model_path = 'E:/GoogleDrive/TechDocuments/Webinars/SVM/alphabet.sav'
    
    im = cv2.imread(testPath) 
    im = cv2.resize(im,(512,512));        
    features,contours = getFeatures(im,backGround='White')
     
    datamean = np.load('E:/GoogleDrive/TechDocuments/Webinars/SVM/dMean.npy')
    #features = features-datamean          
    model = pickle.load(open(model_path,'rb'))
    predictions = model.predict(features)
             
    for i in range(len(predictions)):      
        
        #centx,centy = np.int16(centers[i])        
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)        
        
        cv2.putText(im,str(w*h),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        im = cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),1)
    #cv2.imwrite('E:/GoogleDrive/TechDocuments/Webinars/SVM/alphabets_annot.png',im)
    plt.show()
    cv2.imshow(" ",im)
    cv2.waitKey()

def getAccuracy(actual,predicted):
    count = 0
    for i in range(len(actual)):
        if actual[i]==predicted[i]:
            count+=1
    accuracy = 100*count/len(actual)
    
    return accuracy

def LookUp(num):
    if num==0:text='a'
    if num==1:text='b'
    if num==2:text='c'
    if num==3:text='d'
    if num==4:text='e'
    if num==5:text='f'
    if num==6:text='g'
    if num==7:text='h'
    if num==8:text='i'
    if num==9:text='j'
    if num==10:text='k'
    if num==11:text='l'
    if num==12:text='m'
    if num==13:text='n'
    if num==14:text='o'
    if num==15:text='p'
    if num==16:text='q'
    if num==17:text='r'
    if num==18:text='s'
    if num==19:text='t'
    if num==20:text='u'
    if num==21:text='v'
    if num==22:text='w'
    if num==23:text='x'
    if num==24:text='y'
    if num==25:text='z'
    return text
        
#main()
test()            