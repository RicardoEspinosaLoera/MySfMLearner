import os
import numpy as np
import cv2

i = 0
path = "/workspace/dataset/"
arr= ["Sequence_11-8-53-796","Sequence_12-18-52-88","Sequence_12-59-29-990"]
for i in arr:    
    listd = os.listdir(path+i)
    for x in listd:
        lista = os.listdir(path+i+"/"+x)
        for idx,y in enumerate(lista):
            name = y.split(".")[0]
            # Load .png image
            image = cv2.imread(path+i+"/"+x+"/"+y)
            # Save .jpg image
            cv2.imwrite(path+i+"/"+x+"/"+name+'.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            os.remove(path+i+"/"+x+"/"+y)
            print(y)
