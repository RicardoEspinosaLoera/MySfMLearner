import os
import numpy as np

i = 0
path = "/workspace/dataset/"
arr= ["Sequence_11-8-53-796","Sequence_12-18-52-88","Sequence_12-59-29-990"]
for i in arr:    
    listd = os.listdir(path+i)
    for x in listd:
        lista = os.listdir(path+i+"/"+x)
        for idx,y in enumerate(lista):
            print(y)