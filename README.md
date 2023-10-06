# NF-ULA: Normalizing flow-based unadjusted Langevin algorithm for imaging inverse problems


This repository provides four numerical examples from the paper [1] available at https://arxiv.org/abs/2304.08342. The examples include 
motion deblurring in Section 4.1, 
inpainting in Section 4.2, 
CT reconstruction with Gaussian noise in Section 4.3, and
CT reconstruction with Poisson noise in Appendix.B. 
Please cite this paper, if you use this code. 

This repository uses the networks codes from other repositories. Please download and put those repositories in the directory if you use it.

Please contact Ziruo Cai (ziruocai@gmail.com) for any issues or bug reports.



## 1. REQUIREMENTS

The code requires several Python packages. If you use conda, you can  import from `NFULA.yaml` to create a new conda env:  
`conda env create -n NFULA --file NFULA.yml`, or install the packages one by one.

We tested the code with Python 3.9.16 and the following package versions:

- numpy                     1.24.1
- scipy                     1.10.0
- pytorch                   1.13.1
- torchvision               0.14.1
- tqdm                      4.64.1
- matplotlib                3.6.3
- statsmodels               0.13.5


The code is also compatible with the packages from some other versions.

Some networks are using the codes from other repositories:

#### Deblurring:

- Glow from [Glow-pytorch](https://github.com/rosinality/glow-pytorch),
- RealSN-DnCNN from [Provable_Plug_and_Play](https://github.com/uclaopt/Provable_Plug_and_Play),
- Standard DnCNN from [KAIR](https://github.com/cszn/KAIR),
- DRUnet from [DPIR](https://github.com/cszn/DPIR).

#### Inpainting:

- Glow from [Glow-pytorch](https://github.com/rosinality/glow-pytorch),
- RealSN-DnCNN from [Provable_Plug_and_Play](https://github.com/uclaopt/Provable_Plug_and_Play).

#### CT reconstruction:

- PatchNR from [PatchNR](https://github.com/FabianAltekrueger/patchNR),
- RealSN-DnCNN from [Provable_Plug_and_Play](https://github.com/uclaopt/Provable_Plug_and_Play).

Since the original realSN-DnCNN from [Provable_Plug_and_Play](https://github.com/uclaopt/Provable_Plug_and_Play) supports grayscale images by default and does not support 3-channels images, we made some small changes on it. The user can read `color_realSN-DnCNN.txt` to use or train realSN-DnCNN on 3-channels images.
For CT, we use fan beam geometry implemented by [torch_radon](https://github.com/matteo-ronchetti/torch-radon).
We use wavelet basis [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets) and [statsmodels](https://pypi.org/project/statsmodels/) to calculate the autocorrelation function (ACF) of the samples.


## 2. USAGE AND EXAMPLES

The algorithms NF-ULA and PnP-ULA using different networks are in the file `ULA_NF.py`. You can train the above mentioned normalizing flows or PnP denoisers by yourserf, or download the pre-trained models in the [Google drive download link](https://drive.google.com/drive/folders/1-op8CJHA8ZAguFbxye-Vm05lQsqFabl3?usp=sharing).

### Deblurring

The script `deblurring.py` is the implementation of motion deblurring examples in [1, Section 4.1]. The training data are from [Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)[2] excluding the first 20 faces. The testing images (face1, face2, face3, face4) are randomly chosen from the first 20 face of FFHQ.

### Inpainting 

The script `inpainting.py` is the implementation of the inpainting example in [1, Section 4.2]. The training and testing data are the same as **Deblurring** above. The file `inpainting/mask.pt` is the pre-generated mask of the inpainting example. Please do not change it if you want to recover from the existing `observation.pt`.


### CT reconstruction

The scripts `CT_Gaussian_noise.py` and `CT_Poisson_noise.py` are the implementations of the CT reconstruction with Gaussian noise example in [1, Section 4.3] and the CT reconstruction with Poisson noise example in [1, Appendix.B]. The training and testing set of patchNR and realSN-DnCNN is from LoDoPab[3].



## 3. REFERENCES

[1] Cai, Z., Tang, J., Mukherjee, S., Li, J., Schönlieb, C. B., & Zhang, X. (2023). 
NF-ULA: Langevin Monte Carlo with Normalizing Flow Prior for Imaging Inverse Problems. 
arXiv preprint arXiv:2304.08342.

[2] Karras T, Laine S, Aila T. 
A style-based generator architecture for generative adversarial networks[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 4401-4410.


[3] Leuschner, J., Schmidt, M., Baguer, D. O., & Maass, P. (2021). 
LoDoPaB-CT, a benchmark dataset for low-dose computed tomography reconstruction. 
Scientific Data, 8(1), 109.
