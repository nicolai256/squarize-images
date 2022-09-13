#import base64, os
#from base64 import b64decode
#import matplotlib.pyplot as plt
#from IPython.display import clear_output
import torch
import numpy as np
import cv2
import tqdm
from tqdm import tqdm
import natsort
from natsort import natsorted
import glob,shutil
import subprocess
from PIL import Image
import os, sys
import numpy
import PIL
print("images resized")
source1 = 'tmpsquarize/resizedimages'
destination1 = 'tmpsquarize/Imagesfortrainingdark/'
allfiles = os.listdir(source1)
 
# iterate on all files to move them to destination folder
for f in allfiles:
    src_path = os.path.join(source1, f)
    dst_path = os.path.join(destination1, f)
    shutil.copy(src_path, dst_path)

path_to_dark_input_images="tmpsquarize/Imagesfortrainingdark/"
path2=path_to_dark_input_images
dirs = os.listdir( path2 )
outpathdark = "tmpsquarize/Imagesfortrainingdark/"

from PIL import Image
from PIL import Image, ImageEnhance
def darken():
    for item in dirs:
        if os.path.isfile(path2+item):
          im = Image.open(path2+item)
          enhancer = ImageEnhance.Brightness(im)
          factor = 0 #darkens the image
          im_output = enhancer.enhance(factor)
          im_output.save(path2+item)
darken()
print("images for mask made")
def extraborder():
    subprocess.run(['magick', 'mogrify', '-path', "C:/deepdream-test/stable/stable-diffusion-2/tmpsquarize/resizedimages", '-bordercolor', 'white', '-border', '10', '-format', 'png', "C:/deepdream-test/stable/stable-diffusion-2/tmpsquarize/resizedimages/*"])
    subprocess.run(['magick', 'mogrify', '-path', "C:/deepdream-test/stable/stable-diffusion-2/tmpsquarize/Imagesfortrainingdark", '-bordercolor', 'white', '-border', '10', '-format', 'png', "C:/deepdream-test/stable/stable-diffusion-2/tmpsquarize/Imagesfortrainingdark/*"])

#   subprocess.run(['magick', 'mogrify', '-path', "C:/deepdream-test/stable/stable-diffusion-2/tmpsquarize/padded_imgs-masks", '-resize', '512x512', '-format', 'png', "C:/deepdream-test/stable/stable-diffusion-2/tmpsquarize/padded_imgs-masks/*"])
extraborder()
