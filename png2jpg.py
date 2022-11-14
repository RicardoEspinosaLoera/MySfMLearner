import os
import numpy as np
import cv2


path = "/workspace/dataset/"
arr= ["Sequence_11-8-53-796","Sequence_12-18-52-88","Sequence_12-59-29-990"]
id = 0
for i in arr:    
    listd = os.listdir(path+i)
    #listd.sort()
    for x in listd:
        print(x)
        lista = os.listdir(path+i+"/"+x)
        for idx,y in enumerate(lista):
            name = y.split(".")[0]
            if "png" in y: 
                # Load .png image
                image = cv2.imread(path+i+"/"+x+"/"+y)
                # Save .jpg image
                cv2.imwrite(path+i+"/"+x+"/"+str(id+1)+'.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                os.remove(path+i+"/"+x+"/"+y)
                id = id+1
                #print((idx+1))
                #os.rename(path+i+"/"+x+"/"+y,path+i+"/"+x+"/"+str(idx+1)+'.jpg')
            
