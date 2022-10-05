import os
import shutil
from PIL import Image

i = 0
folder = 1
path = "D:/Phd_Research/Datasets/Real_dominique/Sequence_12-18-52-88"
listd = os.listdir(path)
for folder in listd:
    p = os.path.join(path,folder)
    lista = os.listdir(p)
    for idx,img in enumerate(lista):
        if "png" in img:
            im1 = Image.open(os.path.join(p,img))
            save = os.path.join(p,str(idx+1)+".jpg")
            im1.save(save)
            os.remove(os.path.join(p,img))