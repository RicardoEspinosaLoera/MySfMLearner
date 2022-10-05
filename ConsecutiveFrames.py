import os
import shutil
from PIL import Image

i = 0
folder = 1
path = "/workspace/dataset/Sequence_12-59-29-990"
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