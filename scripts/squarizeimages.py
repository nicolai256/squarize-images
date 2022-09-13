#this script was modified by https://github.com/nicolai256 to make square training images squared without cropping

import argparse
import cv2
import time
import os, sys
import shutil
from pathlib import Path


doc_path = os.path.expanduser('~\Documents')
visions_path = os.path.expanduser('~\Documents\\visions of chaos')

import subprocess
import random
parser = argparse.ArgumentParser()
#inpaint
#parser.add_argument("--mask", type=str, help="thickness of the mask for seamless inpainting",choices=["thinnest", "thin", "medium", "thick", "thickest"],default="medium")
#u can change these
parser.add_argument("--input",type=str,nargs="?",default="tmp360/tiled_image/",help="input image",)
parser.add_argument("--steps",type=int,default=50,help="number of ddim sampling steps",)
parser.add_argument("--projectname",type=str,default="project1",help="foldername of the export",)
parser.add_argument("--edgedetection",type=str,default="40%",help="strength of edgedetection, percentage strength depends on how much border your images have ",)
#u could change this one
parser.add_argument("--outdir",type=str,nargs="?",default="outputs/training_imgs_squarized/",help="dir to write results to",)

opt = parser.parse_args()  



##first pass of inpainting
import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import natsort
from natsort import natsorted
path=opt.input
dirs = os.listdir( path )
path=opt.input

def rmdirs():
	zero1="tmpsquarize"
	one1="tmpsquarize/resizedimages/"
	two1="tmpsquarize/Imagesfortrainingdark/"
	three1="tmpsquarize/padded_imgs-masks/"
	five1="tmpsquarize/images/"

	import shutil

	
	if not os.path.exists(zero1):
		shutil.rmtree(zero1, ignore_errors=True)
	if not os.path.exists(one1):
		shutil.rmtree(one1, ignore_errors=True)
	if not os.path.exists(two1):
		shutil.rmtree(two1, ignore_errors=True)
	if not os.path.exists(three1):
		shutil.rmtree(three1, ignore_errors=True)
	if not os.path.exists(five1):
		shutil.rmtree(five1, ignore_errors=True)
	import subprocess
	from pathlib import Path

	#using pathlib.Path
	path = Path(zero1)
	subprocess.run(["rm", "-rf", str(path)])

	#using strings
	path = zero1
	subprocess.run(["rm", "-rf", path])
rmdirs()

def mkdirs():
	zero="tmpsquarize"
	one="tmpsquarize/resizedimages/"
	two="tmpsquarize/Imagesfortrainingdark/"
	three="tmpsquarize/padded_imgs-masks/"
	five="tmpsquarize/images/"
	four="outputs/training_imgs_squarized/"+opt.projectname+"/"
	six="tmpsquarize/trimmedborders/"
	if not os.path.exists(zero):
		os.mkdir(zero)
	if not os.path.exists(one):
		os.mkdir(one)
	if not os.path.exists(two):
		os.mkdir(two)
	if not os.path.exists(three):
		os.mkdir(three)
	if not os.path.exists(four):
		os.mkdir(four)
	if not os.path.exists(five):
		os.mkdir(five)
	if not os.path.exists(six):
		os.mkdir(six)
mkdirs()

def move ():
	source1 = opt.input
	destination1 = "tmpsquarize/images/"
	allfiles = os.listdir(source1)
 
	# iterate on all files to move them to destination folder
	for f in allfiles:
		src_path = os.path.join(source1, f)
		dst_path = os.path.join(destination1, f)
		shutil.copy(src_path, dst_path)
move()

def edgeremoval():
	print('removing edges for improved inpainting')
	edgedetection = opt.edgedetection
	subprocess.run(['magick', 'mogrify', '-path', "C:/deepdream-test/stable/stable-diffusion-2/tmpsquarize/resizedimages", '-fuzz', edgedetection, '-trim', '-format', 'png', "C:/deepdream-test/stable/stable-diffusion-2/tmpsquarize/images/*"])
	
edgeremoval()

def crop():
	path_to_input_images="tmpsquarize/resizedimages/"
	path=path_to_input_images
	dirs = os.listdir( path )
	subprocess.call(['python.exe','scripts/squarize/tmp_imgs_crop.py'])
	print('cropping 5%')
	for item in dirs:
		if os.path.isfile(path+item):
			im = Image.open(path+item).convert("RGB")
			w, h = im.size
 
			# Setting the points for cropped image
			left =  w-w+10
			top =  h-h+10
			right =  w-10
			bottom = h-10
 
			# Cropped image of above dimension
			# (It will not change original image)
			im1 = im.crop((left, top, right, bottom))
			im1.save(path+item)
	print('cropped')
crop()


def square():
	path_to_input_images="tmpsquarize/resizedimages/"
	path=path_to_input_images
	files=glob.glob(path+"*")
	files=natsorted(files)
	for image in tqdm(files):
		head_tail = os.path.basename(image)
		i=cv2.imread(image)
		size=i.shape
		h,w=size[0],size[1]
		output_folder = opt.outdir + "/" + opt.projectname + "/" + head_tail
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
square()

def padding():
	subprocess.call(['python.exe','scripts/squarize/tmp_imgs_darken.py'])
	subprocess.call(['python.exe','scripts/squarize/tmp_imgs_dark.py'])
	subprocess.call(['python.exe','scripts/squarize/tmp_imgs.py'])
	subprocess.call(['python.exe','scripts/squarize/pad_imgs.py'])
	subprocess.call(['python.exe','scripts/squarize/pad_masks.py'])
padding()

def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch

def run():
	#for item in dirs:
	#	if os.path.isfile(path+item):
			import ntpath
			global base_count
			indir = "tmpsquarize/padded_imgs-masks/"
			masks = sorted(glob.glob(os.path.join(indir, "*_mask.png")))
			images = [x.replace("_mask.png", ".png") for x in masks]
			#basename = ntpath.basename(images)
			outdir = opt.outdir + "/" + opt.projectname + "/"
			print(f"Found {len(masks)} inputs.")

			config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
			model = instantiate_from_config(config.model)
			#'MachineLearning/Super Resolution Latent-Diffusion/last.ckpt'
			model.load_state_dict(torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"],strict=False)

			device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
			model = model.to(device)
			sampler = DDIMSampler(model)

			os.makedirs(opt.outdir, exist_ok=True)
			with torch.no_grad():
				with model.ema_scope():
					for image, mask in tqdm(zip(images, masks)):
						outpath = os.path.join(outdir, os.path.split(image)[1])
						batch = make_batch(image, mask, device=device)
						global base_count
						# encode masked image and concat downsampled mask
						c = model.cond_stage_model.encode(batch["masked_image"])
						cc = torch.nn.functional.interpolate(batch["mask"],size=c.shape[-2:])
						c = torch.cat((c, cc), dim=1)

						shape = (c.shape[1]-1,)+c.shape[2:]
						samples_ddim, _ = sampler.sample(S=opt.steps,conditioning=c,batch_size=c.shape[0],shape=shape,verbose=False)
						x_samples_ddim = model.decode_first_stage(samples_ddim)

						image = torch.clamp((batch["image"]+1.0)/2.0,min=0.0, max=1.0)
						mask = torch.clamp((batch["mask"]+1.0)/2.0,min=0.0, max=1.0)
						predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,min=0.0, max=1.0)

						inpainted = (1-mask)*image+mask*predicted_image
						inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
						base_count = 0
						base_count += 1
						Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
						base_count += 1

			
run()

def rmdirs():
	zero1="tmpsquarize"
	one1="tmpsquarize/resizedimages/"
	two1="tmpsquarize/Imagesfortrainingdark/"
	three1="tmpsquarize/padded_imgs-masks/"
	five1="tmpsquarize/images/"

	import shutil

	
	if not os.path.exists(zero1):
		shutil.rmtree(zero1, ignore_errors=True)
	if not os.path.exists(one1):
		shutil.rmtree(one1, ignore_errors=True)
	if not os.path.exists(two1):
		shutil.rmtree(two1, ignore_errors=True)
	if not os.path.exists(three1):
		shutil.rmtree(three1, ignore_errors=True)
	if not os.path.exists(five1):
		shutil.rmtree(five1, ignore_errors=True)
	import subprocess
	from pathlib import Path

	#using pathlib.Path
	path = Path(zero1)
	subprocess.run(["rm", "-rf", str(path)])

	#using strings
	path = zero1
	subprocess.run(["rm", "-rf", path])
rmdirs()