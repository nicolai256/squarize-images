#@title pad squarize images
import base64, os
import os
import shutil
extract_from="folder"
path_to_input_images="tmpsquarize/resizedimages/"
path=path_to_input_images

if not os.path.exists(path): raise Exception("folder doesn't exist, please check it.")
#if extract_from=="zip":
#  shutil.unpack_archive("/content/inputs","zip",path)
#  path="/content/inputs/"
else:
  if not path.endswith("/"): path+="/"

import math
from math import ceil
import glob,shutil
import natsort
from natsort import natsorted
import tqdm
from tqdm import tqdm
import cv2
import numpy as np
#from IPython.display import clear_output

from PIL import Image

c=0
files=glob.glob(path+"*")
files=natsorted(files)
output_folder="outputs/training_imgs_squarized/" 
if not output_folder.endswith("/"): output_folder+="/"
if not os.path.exists(output_folder):
  os.mkdir(output_folder) 
print("Preprocessing images")

if not os.path.exists(output_folder):
  os.mkdir(output_folder)

from PIL import Image
import os, sys
import PIL
dirs = os.listdir( path )

for image in tqdm(files):
  
  i=cv2.imread(image)
  size=i.shape
  h,w=size[0],size[1]
  if w==h:
    shutil.move(os.path.join(image),output_folder)
  elif w==h-1:
    shutil.move(os.path.join(image),output_folder)
  elif w==h+1:
    shutil.move(os.path.join(image),output_folder)
  elif w-1==h:
    shutil.move(os.path.join(image),output_folder)
  elif w+1==h:
    shutil.move(os.path.join(image),output_folder)
  if w>h:
    newim=np.zeros((w,w,3),dtype=np.uint8)
    newim[:,:]=255
    pad=int((w-h)/2)
    padpad=int(w*0.04)
    newim[pad:w-pad,0:w]=0
    newim[:,:]=255
    #cv2.imwrite("/content/temp/"+str(c)+"_mask.png",newim)
    if pad>((w-h)/2):
      newim[pad-1:w-pad,0:w]=i[:,:]
    elif pad<((h-w)/2):
      newim[pad+1:w-pad,0:w]=i[:,:]
    #else:
     # newim[pad+1:w-pad,0:w]=i[:,:]
    else:
      newim[pad:w-pad,0:w]=i[:,:]
    cv2.imwrite("tmpsquarize/padded_imgs-masks/"+str(c)+".png",newim)
  else:
    newim=np.zeros((h,h,3),dtype=np.uint8)
    newim[:,:]=255
    pad=int((h-w)/2)
    newim[0:h,pad:h-pad]=0
    #cv2.imwrite("/content/temp/"+str(c)+"_mask.png",newim)
    newim[:,:]=255
    if pad>((h-w)/2):
      newim[0:h,pad-1:h-pad]=i[:,:]
    elif pad<((h-w)/2):
      newim[0:h,pad+1:h-pad]=i[:,:]
    else:
      newim[0:h,pad:h-pad]=i[:,:]
    cv2.imwrite("tmpsquarize/padded_imgs-masks/"+str(c)+".png",newim)
  c=c+1
  Exception("hi")
print("masks half done...")