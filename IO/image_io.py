from PIL import Image
import os
import numpy as np

def load_imgs(img_dir):
    Imgs = []
    img_lst = os.listdir(img_dir)
    img_lst.sort(reverse=True)
    for filename in img_lst:
        Imgs.append(np.array(Image.open(img_dir+filename)))
        
    return np.asarray(Imgs)