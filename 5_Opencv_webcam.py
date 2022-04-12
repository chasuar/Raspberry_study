import cv2
import numpy as np
from keras.models import load_model

cap =cv2.VideoCapture(0)
model = load_model('Predict_Model.h5')
while (cap.isOpened()):
   ret, img=cap.read()
   if(ret) :

    if cv2.waitKey(1) & 0xFF == 27: # esc 키를 누르면 닫음
            break
    
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur=cv2.GaussianBlur(img_gray, (5,5),0)
    ret, img_th=cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, hierachy=cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    img_c=img.copy()
    img_class=[]

    if len(contours)>0:
        for ctr in contours:
            if cv2.contourArea(ctr)>1000 and cv2.contourArea(ctr)<15000:
                rect = cv2.boundingRect(ctr)
                img_class=img_th[rect[1] : rect[1]+rect[3], rect[0]: rect[0]+rect[2]]
                test=cv2.resize(img_class,(28,28))
                test=255-test
                test=test/255.0
                test=test.reshape((1,28,28,1))
                
                predict=model.predict_on_batch(test)
                mypred=np.argmax(predict, axis=1)
                
                cv2.rectangle(img_c, (rect[0], rect[1]),(rect[0]+rect[2], rect[1]+rect[3]),(0,255,0),5)
                font=cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_c, str(mypred),(rect[0],rect[1]),font, 1, (0,0,255),1)



                 
    cv2.imshow('Frame',img_th)
    cv2.imshow("Conours", img_c)
