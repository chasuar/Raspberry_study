import cv2
import numpy as np
from keras.models import load_model



cap =cv2.VideoCapture(0)
model = load_model('ox_Model.h5')
while (cap.isOpened()):
    
    ret, img=cap.read()

   
    if cv2.waitKey(1) & 0xFF == 27: # esc 키를 누르면 닫음
            break
    if(ret) :
        gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur=cv2.GaussianBlur(gray, (5,5),0)
        ret, th=cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        contours, hierachy=cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours)>0:
            for ctr in contours:
                if cv2.contourArea(ctr)>1000 and cv2.contourArea(ctr)<20000:
                    rect = cv2.boundingRect(ctr)
                    img_class=th[rect[1] : rect[1]+rect[3], rect[0]: rect[0]+rect[2]]
                    test=cv2.resize(img_class,(100,100))
                    test=255-test
                    test=test/255.0
                    test=test.reshape((1,100,100,1))
                    predict=model.predict_on_batch(test)
                    mypred=np.argmax(predict, axis=1)
                    score=round(100*np.max(predict[0]),2)
                    if score>95:
                        cv2.rectangle(img, (rect[0], rect[1]),(rect[0]+rect[2], rect[1]+rect[3]),(0,255,0),5)

                        if mypred==0:
                            s="O :"+str(score)+"%"
                        elif mypred==1:
                            s="X :"+str(score)+"%"
                
                        font=cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img, s,(rect[0],rect[1]),font, 1, (0,0,255),1)
        
        cv2.imshow('Frame',img)
        cv2.imshow('binary',th)
        
cv2.waitKey()
cv2.destroyAllWindows()

