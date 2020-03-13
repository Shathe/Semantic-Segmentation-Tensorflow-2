
import numpy as np
import random
import cv2
import os
import argparse
import glob
from PIL import Image, ImageOps

random.seed(os.urandom(9))

from glob import glob

for file in glob("./labels/*/*"):

    print(file)
    

    im = Image.open(file)
    old_size = im.size  # old_size[0] is in (width, height) format
    print(old_size)

    
    '''
    new_size = (old_size[0]*2,old_size[1]*2)
    # use thumbnail() or resize() method to resize the input image
    # thumbnail is a in-place operation
    # im.thumbnail(new_size, Image.ANTIALIAS)
    im = im.resize(new_size, Image.NEAREST)# NEAREST BILINEAR
    # create a new image and paste the resized on it
   
    im.save(file) 
    '''
