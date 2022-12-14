# squarize-images

#

activate SD environment

```pip install natsort==8.1.0```  on top of your environment

place src folder in stable-diffusion folder

Download the pre-trained weights

```
wget -O src/latent-diffusion/models/ldm/inpainting_big/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
```
to run the script

```
python src\\latent-diffusion\\squarizeimages.py --input "C:\Users\Gebruiker\Documents\Visions of Chaos\Movies\00000000/" --steps "50" --projectname "square1" --edgeremoval --edgedetection "40%" --extra_crop --outdir "whatever/dir/is/possible"
```
```--input "path/to/input/images/folder/"``` the input folder

```--steps "50"``` the amount of steps the inpainting does

```--edgeremoval "1"``` activates --edgedetection

```--edgedetection "40%"``` the percentage of how much borderdetection is going on

```--extra_crop "1"``` adds an extra 10px crop on each side 

```--outdir "path/to/output/images/folder"``` the output folder

```--voc "1"``` if you have voc installed it will use the folders in voc automatically and you won't need to pass the latent diffusion location

```--latent_diffusion_path "path/to/Stable Diffusion/src/latent-diffusion"``` **only pass this if you're not using the voc version**

![](demo.png)

this is an example of a combination of ```--edgeremoval "1" --edgedetection "40%" --extra_crop "1"``` to remove borders

![](demo2.png)
