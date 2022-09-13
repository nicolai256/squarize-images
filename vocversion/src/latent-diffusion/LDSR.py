#@title Clone repos and install requirements
#%cd '/content'
#!git clone https://github.com/CompVis/latent-diffusion.git
#!git clone https://github.com/CompVis/taming-transformers
#!pip install -e ./taming-transformers
#!pip install ipywidgets omegaconf>=2.0.0 pytorch-lightning>=1.0.8 torch-fidelity einops

import sys
#import ipywidgets as widgets
import os
import gc
from tabnanny import check
#from IPython import display
sys.path.append(".")
sys.path.append('./taming-transformers')
from taming.models import vqgan # checking correct import from taming
from torchvision.datasets.utils import download_url
#%cd '/content/latent-diffusion'
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import ismap
#%cd '/content'
import torch
#from google.colab import files
#from IPython.display import Image as ipyimg
#import ipywidgets as widgets
#import resampling from PIL
import tempfile

from PIL import Image
from numpy import asarray
from einops import rearrange, repeat
import torch, torchvision
import time
from omegaconf import OmegaConf
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#create class LDSR
class LDSR():
    #init function
    def __init__(self, modelPath,yamlPath):
        self.modelPath = modelPath
        self.yamlPath = yamlPath
        #self.model = self.load_model_from_config()
        #print(self.load_model_from_config(OmegaConf.load(yamlPath), modelPath))
        #self.print_current_directory()
    #get currennt directory

    '''
    def check_model_exists(self):
        #check if model and yaml exist
        path = self.pathInput + "/models/ldm/ld_sr".replace('\\',os.sep).replace('/',os.sep)
        model = self.modelName
        yaml = self.yamlName
        if os.path.exists(path):
            #check if yaml exists
            if os.path.exists(os.path.join(path,yaml)):
                print('YAML found')
                #check if ckpt exists
                if os.path.exists(os.path.join(path,model)):
                    print('Model found')
                    return os.path.join(path,model), os.path.join(path,yaml)
                else:
                    return False
        #return onlyfiles
    '''
    def load_model_from_config(self):
        #print(f"Loading model from {self.modelPath}")
        pl_sd = torch.load(self.modelPath, map_location="cpu")
        global_step = pl_sd["global_step"]
        sd = pl_sd["state_dict"]
        config = OmegaConf.load(self.yamlPath)
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        model.cuda()
        model.eval()
        return {"model": model}#, global_step

    '''
    def get_model(self):
        check = self.check_model_exists()
        if check != False:
            path_ckpt = check[0]
            path_conf = check[1]
        else:
            print('Model not found, please run the bat file to download the model')
        config = OmegaConf.load(path_conf)
        model, step = self.load_model_from_config(config, path_ckpt)
        return model

    
    def get_custom_cond(mode):
        dest = "data/example_conditioning"

        if mode == "superresolution":
            uploaded_img = files.upload()
            filename = next(iter(uploaded_img))
            name, filetype = filename.split(".") # todo assumes just one dot in name !
            os.rename(f"{filename}", f"{dest}/{mode}/custom_{name}.{filetype}")

        elif mode == "text_conditional":
            #w = widgets.Text(value='A cake with cream!', disabled=True)
            w = 'Empty Test'
            display.display(w)

            with open(f"{dest}/{mode}/custom_{w.value[:20]}.txt", 'w') as f:
                f.write(w.value)

        elif mode == "class_conditional":
            #w = widgets.IntSlider(min=0, max=1000)
            w = 1000
            display.display(w)
            with open(f"{dest}/{mode}/custom.txt", 'w') as f:
                f.write(w.value)

        else:
            raise NotImplementedError(f"cond not implemented for mode{mode}")
    '''

    def get_cond_options(self,mode):
        path = "data/example_conditioning"
        path = os.path.join(path, mode)
        onlyfiles = [f for f in sorted(os.listdir(path))]
        return path, onlyfiles

    '''
    def select_cond_path(mode):
        path = "data/example_conditioning"  # todo
        path = os.path.join(path, mode)
        onlyfiles = [f for f in sorted(os.listdir(path))]

        selected = widgets.RadioButtons(
            options=onlyfiles,
            description='Select conditioning:',
            disabled=False
        )
        display.display(selected)
        selected_path = os.path.join(path, selected.value)
        return selected_path
    '''

    

    '''
    # Google Collab stuff
    def visualize_cond_img(path):
        display.display(ipyimg(filename=path))
    '''

    def run(self,model, selected_path, task, custom_steps, eta, resize_enabled=False, classifier_ckpt=None, global_step=None):
        def make_convolutional_sample(batch, model, mode="vanilla", custom_steps=None, eta=1.0, swap_mode=False, masked=False,
                              invert_mask=True, quantize_x0=False, custom_schedule=None, decode_interval=1000,
                              resize_enabled=False, custom_shape=None, temperature=1., noise_dropout=0., corrector=None,
                              corrector_kwargs=None, x_T=None, save_intermediate_vid=False, make_progrow=True,ddim_use_x0_pred=False):
            log = dict()

            z, c, x, xrec, xc = model.get_input(batch, model.first_stage_key,
                                                return_first_stage_outputs=True,
                                                force_c_encode=not (hasattr(model, 'split_input_params')
                                                                    and model.cond_stage_key == 'coordinates_bbox'),
                                                return_original_cond=True)

            log_every_t = 1 if save_intermediate_vid else None

            if custom_shape is not None:
                z = torch.randn(custom_shape)
                # print(f"Generating {custom_shape[0]} samples of shape {custom_shape[1:]}")

            z0 = None

            log["input"] = x
            log["reconstruction"] = xrec

            if ismap(xc):
                log["original_conditioning"] = model.to_rgb(xc)
                if hasattr(model, 'cond_stage_key'):
                    log[model.cond_stage_key] = model.to_rgb(xc)

            else:
                log["original_conditioning"] = xc if xc is not None else torch.zeros_like(x)
                if model.cond_stage_model:
                    log[model.cond_stage_key] = xc if xc is not None else torch.zeros_like(x)
                    if model.cond_stage_key =='class_label':
                        log[model.cond_stage_key] = xc[model.cond_stage_key]

            with model.ema_scope("Plotting"):
                t0 = time.time()
                img_cb = None

                sample, intermediates = convsample_ddim(model, c, steps=custom_steps, shape=z.shape,
                                                        eta=eta,
                                                        quantize_x0=quantize_x0, img_callback=img_cb, mask=None, x0=z0,
                                                        temperature=temperature, noise_dropout=noise_dropout,
                                                        score_corrector=corrector, corrector_kwargs=corrector_kwargs,
                                                        x_T=x_T, log_every_t=log_every_t)
                t1 = time.time()

                if ddim_use_x0_pred:
                    sample = intermediates['pred_x0'][-1]

            x_sample = model.decode_first_stage(sample)

            try:
                x_sample_noquant = model.decode_first_stage(sample, force_not_quantize=True)
                log["sample_noquant"] = x_sample_noquant
                log["sample_diff"] = torch.abs(x_sample_noquant - x_sample)
            except:
                pass

            log["sample"] = x_sample
            log["time"] = t1 - t0

            return log
        def convsample_ddim(model, cond, steps, shape, eta=1.0, callback=None, normals_sequence=None,
                    mask=None, x0=None, quantize_x0=False, img_callback=None,
                    temperature=1., noise_dropout=0., score_corrector=None,
                    corrector_kwargs=None, x_T=None, log_every_t=None
                    ):

            ddim = DDIMSampler(model)
            bs = shape[0]  # dont know where this comes from but wayne
            shape = shape[1:]  # cut batch dim
            print(f"Sampling with eta = {eta}; steps: {steps}")
            samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, conditioning=cond, callback=callback,
                                                normals_sequence=normals_sequence, quantize_x0=quantize_x0, eta=eta,
                                                mask=mask, x0=x0, temperature=temperature, verbose=False,
                                                score_corrector=score_corrector,
                                                corrector_kwargs=corrector_kwargs, x_T=x_T)

            return samples, intermediates
        # global stride
        def get_cond(mode, selected_path):
            example = dict()
            if mode == "superresolution":
                up_f = 4
                #visualize_cond_img(selected_path)

                c = selected_path.convert('RGB')
                c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
                c_up = torchvision.transforms.functional.resize(c, size=[up_f * c.shape[2], up_f * c.shape[3]], antialias=True)
                c_up = rearrange(c_up, '1 c h w -> 1 h w c')
                c = rearrange(c, '1 c h w -> 1 h w c')
                c = 2. * c - 1.

                c = c.to(torch.device("cuda"))
                example["LR_image"] = c
                example["image"] = c_up

            return example
        example = get_cond(task, selected_path)

        save_intermediate_vid = False
        n_runs = 1
        masked = False
        guider = None
        ckwargs = None
        mode = 'ddim'
        ddim_use_x0_pred = False
        temperature = 1.
        eta = eta
        make_progrow = True
        custom_shape = None

        height, width = example["image"].shape[1:3]
        split_input = height >= 128 and width >= 128

        if split_input:
            ks = 128
            stride = 64
            vqf = 4  #
            model.split_input_params = {"ks": (ks, ks), "stride": (stride, stride),
                                        "vqf": vqf,
                                        "patch_distributed_vq": True,
                                        "tie_braker": False,
                                        "clip_max_weight": 0.5,
                                        "clip_min_weight": 0.01,
                                        "clip_max_tie_weight": 0.5,
                                        "clip_min_tie_weight": 0.01}
        else:
            if hasattr(model, "split_input_params"):
                delattr(model, "split_input_params")

        invert_mask = False

        x_T = None
        for n in range(n_runs):
            if custom_shape is not None:
                x_T = torch.randn(1, custom_shape[1], custom_shape[2], custom_shape[3]).to(model.device)
                x_T = repeat(x_T, '1 c h w -> b c h w', b=custom_shape[0])

            logs = make_convolutional_sample(example, model,
                                         mode=mode, custom_steps=custom_steps,
                                         eta=eta, swap_mode=False , masked=masked,
                                         invert_mask=invert_mask, quantize_x0=False,
                                         custom_schedule=None, decode_interval=10,
                                         resize_enabled=resize_enabled, custom_shape=custom_shape,
                                         temperature=temperature, noise_dropout=0.,
                                         corrector=guider, corrector_kwargs=ckwargs, x_T=x_T, save_intermediate_vid=save_intermediate_vid,
                                         make_progrow=make_progrow,ddim_use_x0_pred=ddim_use_x0_pred
                                         )
        return logs


    @torch.no_grad()
    


    @torch.no_grad()
    
    def superResolution(self,image,ddimSteps=100,preDownScale='None',postDownScale='None'):
        diffMode = 'superresolution'
        model = self.load_model_from_config()
        #@title Import location
        #@markdown ***File height and width should be multiples of 64, or image will be padded.***

        #@markdown *To change upload settings without adding more, run and cancel upload*
        #import_method = 'Directory' #@param ['Google Drive', 'Upload']
        #output_subfolder_name = 'processed' #@param {type: 'string'}

        #@markdown Drive method options:
        #drive_directory = '/content/drive/MyDrive/upscaleTest' #@param {type: 'string'}

        #@markdown Upload method options:
        #remove_previous_uploads = False #@param {type: 'boolean'}
        #save_output_to_drive = False #@param {type: 'boolean'}
        #zip_if_not_drive = False #@param {type: 'boolean'}
        '''
        os.makedirs(pathInput+'/content/input'.replace('\\',os.sep).replace('/',os.sep), exist_ok=True)
        output_directory = os.getcwd()+f'/content/output/{output_subfolder_name}'.replace('\\',os.sep).replace('/',os.sep)
        os.makedirs(output_directory, exist_ok=True)
        uploaded_img = pathInput+'/content/input/'.replace('\\',os.sep).replace('/',os.sep)
        pathInput, dirsInput, filesInput = next(os.walk(pathInput+'/content/input').replace('\\',os.sep).replace('/',os.sep))
        file_count = len(filesInput)
        print(f'Found {file_count} files total')
        '''


        #Run settings

        diffusion_steps = int(ddimSteps) #@param [25, 50, 100, 250, 500, 1000]
        eta = 1.0 #@param  {type: 'raw'}
        stride = 0 #not working atm

        # ####Scaling options:
        # Downsampling to 256px first will often improve the final image and runs faster.
        
        # You can improve sharpness without upscaling by upscaling and then downsampling to the original size (i.e. Super Resolution)
        pre_downsample = preDownScale #@param ['None', '1/2', '1/4']

        post_downsample = postDownScale #@param ['None', 'Original Size', '1/2', '1/4']

        # Nearest gives sharper results, but may look more pixellated. Lancoz is much higher quality, but result may be less crisp.
        downsample_method = 'Lanczos' #@param ['Nearest', 'Lanczos']


        overwrite_prior_runs = True #@param {type: 'boolean'}

        #pathProcessed, dirsProcessed, filesProcessed = next(os.walk(output_directory))

        #for img in filesInput:
        #    if img in filesProcessed and overwrite_prior_runs is False:
        #        print(f'Skipping {img}: Already processed')
        #        continue
        gc.collect()
        torch.cuda.empty_cache()
        #dir = pathInput
        #filepath = os.path.join(dir, img).replace('\\',os.sep).replace('/',os.sep)

        im_og = image
        width_og, height_og = im_og.size

        #Downsample Pre
        if pre_downsample == '1/2':
            downsample_rate = 2
        elif pre_downsample == '1/4':
            downsample_rate = 4
        else:
            downsample_rate = 1
        # get system temp directory
        #dir = tempfile.gettempdir()
        width_downsampled_pre = width_og//downsample_rate
        height_downsampled_pre = height_og//downsample_rate
        if downsample_rate != 1:
            print(f'Downsampling from [{width_og}, {height_og}] to [{width_downsampled_pre}, {height_downsampled_pre}]')
            im_og = im_og.resize((width_downsampled_pre, height_downsampled_pre), Image.LANCZOS)
            #os.makedirs(dir, exist_ok=True)
            #im_og.save(dir + '/ldsr/temp.png'.replace('\\',os.sep).replace('/',os.sep))
            #filepath = dir + '/ldsr/temp.png'.replace('\\',os.sep).replace('/',os.sep)

        logs = self.run(model["model"], im_og, diffMode, diffusion_steps, eta)

        sample = logs["sample"]
        sample = sample.detach().cpu()
        sample = torch.clamp(sample, -1., 1.)
        sample = (sample + 1.) / 2. * 255
        sample = sample.numpy().astype(np.uint8)
        sample = np.transpose(sample, (0, 2, 3, 1))
        #print(sample.shape)
        a = Image.fromarray(sample[0])

        #Downsample Post
        if post_downsample == '1/2':
            downsample_rate = 2
        elif post_downsample == '1/4':
            downsample_rate = 4
        else:
            downsample_rate = 1

        width, height = a.size
        width_downsampled_post = width//downsample_rate
        height_downsampled_post = height//downsample_rate

        if downsample_method == 'Lanczos':
            aliasing = Image.LANCZOS
        else:
            aliasing = Image.NEAREST

        if downsample_rate != 1:
            print(f'Downsampling from [{width}, {height}] to [{width_downsampled_post}, {height_downsampled_post}]')
            a = a.resize((width_downsampled_post, height_downsampled_post), aliasing)
        elif post_downsample == 'Original Size':
            print(f'Downsampling from [{width}, {height}] to Original Size [{width_og}, {height_og}]')
            a = a.resize((width_og, height_og), aliasing)

        #display.display(a)
        #a.save(f'{output_directory}/{img}')
        del model
        gc.collect()
        torch.cuda.empty_cache()
        '''
        if import_method != 'Google Drive' and zip_if_not_drive is True:
        print('Zipping files')
        current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
        output_zip_name = 'output'+str(current_time)+'.zip'
        #!zip -r {output_zip_name} {output_directory}
        print(f'Zipped outputs in {output_zip_name}')
        '''
        print(f'Processing finished!')
        return a