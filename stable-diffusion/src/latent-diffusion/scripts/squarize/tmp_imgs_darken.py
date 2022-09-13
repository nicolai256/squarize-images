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
import argparse
print("images resized")
parser = argparse.ArgumentParser()
parser.add_argument("--voc",type=int,default=False,help="activate edgedetection",)
opt = parser.parse_args()

if opt.voc:
    visions_path = os.path.expanduser('~\Documents\\visions of chaos')
else:
    visions_path = os.path.expanduser('~\Documents')
    
source1 = visions_path+"/tmpsquarize/resizedimages/"
source2 = visions_path+"/tmpsquarize/resizedimages/*"
destination1 = visions_path+"/tmpsquarize/Imagesfortrainingdark/"
allfiles = os.listdir(source1)
 
# iterate on all files to move them to destination folder
for f in allfiles:
    src_path = os.path.join(source1, f)
    dst_path = os.path.join(destination1, f)
    shutil.copy(src_path, dst_path)

path_to_dark_input_images=visions_path+"/tmpsquarize/Imagesfortrainingdark/"
path2=path_to_dark_input_images
dirs = os.listdir( path2 )
outpathdark = visions_path+"/tmpsquarize/Imagesfortrainingdark/"
outpathdark2 = visions_path+"/tmpsquarize/Imagesfortrainingdark/*"

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
    subprocess.run(['magick', 'mogrify', '-path', source1, '-bordercolor', 'white', '-border', '10', '-format', 'png', source2])
    subprocess.run(['magick', 'mogrify', '-path', outpathdark, '-bordercolor', 'white', '-border', '10', '-format', 'png', outpathdark2])

#   subprocess.run(['magick', 'mogrify', '-path', "C:/deepdream-test/stable/stable-diffusion-2/tmpsquarize/padded_imgs-masks", '-resize', '512x512', '-format', 'png', "C:/deepdream-test/stable/stable-diffusion-2/tmpsquarize/padded_imgs-masks/*"])
extraborder()
