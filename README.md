# squarize-images

place the scripts folder in your stable diffusion directory

Download the pre-trained weights
```
wget -O models/ldm/inpainting_big/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
```

**run squarize images**

```
python scripts/squarizeimages.py --input "path/to/input/folder/" --steps "50" --projectname "squarize1" --edgedetection "40%" --outdir path/to/output
```

![](demo.png)
